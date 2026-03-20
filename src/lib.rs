use fancy_regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy)]
struct Node {
    token_id: u32,
    prev: Option<u32>,
    next: Option<u32>,
}

struct Chain {
    nodes: Vec<Option<Node>>,
    head: Option<u32>,
}

impl Chain {
    pub fn new(bytes: &[u8]) -> Self {
        let n = bytes.len();
        Self {
            head: if bytes.is_empty() { None } else { Some(0) },
            nodes: bytes
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    Some(Node {
                        token_id: b as u32,
                        prev: if i > 0 { Some(i as u32 - 1) } else { None },
                        next: if i + 1 < n { Some(i as u32 + 1) } else { None },
                    })
                })
                .collect(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = (u32, Node)> + '_ {
        let mut cur = self.head;
        std::iter::from_fn(move || {
            let pos = cur?;
            let node = self.nodes[pos as usize].unwrap();
            cur = node.next;
            Some((pos, node))
        })
    }

    /// Replaces the [left, right] pair with a new merged node, returning the new node's position.
    pub fn splice(&mut self, left: u32, right: u32, new_token_id: u32) -> u32 {
        let prev = self.nodes[left as usize].as_ref().unwrap().prev;
        let next = self.nodes[right as usize].as_ref().unwrap().next;
        let pos = self.nodes.len() as u32;

        if let Some(p) = prev {
            self.nodes[p as usize].as_mut().unwrap().next = Some(pos);
        } else {
            self.head = Some(pos);
        }
        if let Some(n) = next {
            self.nodes[n as usize].as_mut().unwrap().prev = Some(pos);
        }

        self.nodes.push(Some(Node {
            token_id: new_token_id,
            prev,
            next,
        }));
        self.nodes[left as usize] = None;
        self.nodes[right as usize] = None;
        pos
    }
}

pub struct BPE {
    split_pattern: Regex,
    vocab: HashMap<u32, Vec<u8>>,        // id -> bytes
    merge_map: HashMap<(u32, u32), u32>, // pair -> merged id

    // training-time state
    chains: Vec<Chain>,
    count_to_pairs: BTreeMap<u32, BTreeSet<(u32, u32)>>, // frequency -> pairs
    pair_counts: HashMap<(u32, u32), u32>,               // pair -> frequency
    pair_locs: HashMap<(u32, u32), BTreeSet<(usize, u32)>>, // pair -> (chain_idx, node_pos)
}

impl BPE {
    pub fn new(split_pattern: impl AsRef<str>) -> Self {
        Self {
            split_pattern: Regex::new(split_pattern.as_ref()).expect("invalid split regex"),
            vocab: (0u32..256).map(|b| (b, vec![b as u8])).collect(),
            merge_map: HashMap::new(),

            chains: Vec::new(),
            count_to_pairs: BTreeMap::new(),
            pair_counts: HashMap::new(),
            pair_locs: HashMap::new(),
        }
    }

    /// Increment (`delta = 1`) or decrement (`delta = -1`) a pair's frequency and location set.
    fn adjust(&mut self, pair: (u32, u32), seq: usize, pos: u32, delta: i32) {
        let old = self.pair_counts.get(&pair).copied().unwrap_or(0);
        let new = (old as i32 + delta) as u32;

        // Remove pair from its old frequency bucket.
        if old > 0 {
            let bucket = self.count_to_pairs.get_mut(&old).unwrap();
            bucket.remove(&pair);
            if bucket.is_empty() {
                self.count_to_pairs.remove(&old);
            }
        }

        if new == 0 {
            // Pair has been fully merged away — drop all tracking state.
            self.pair_counts.remove(&pair);
            self.pair_locs.remove(&pair);
        } else {
            self.pair_counts.insert(pair, new);
            self.count_to_pairs.entry(new).or_default().insert(pair);
            let locs = self.pair_locs.entry(pair).or_default();
            if delta > 0 {
                locs.insert((seq, pos));
            } else {
                locs.remove(&(seq, pos));
            }
        }
    }

    pub fn train(&mut self, vocab_size: u32, docs: impl IntoIterator<Item = impl AsRef<str>>) {
        for doc in docs {
            self.chains.extend(self.split(doc));
        }

        let mut chains = std::mem::take(&mut self.chains);

        // Populate initial pair counts from all chains.
        for (si, chain) in chains.iter().enumerate() {
            let nodes: Vec<(u32, Node)> = chain.iter().collect();
            for window in nodes.windows(2) {
                let (left_pos, left_node) = window[0];
                let (_, right_node) = window[1];
                self.adjust((left_node.token_id, right_node.token_id), si, left_pos, 1);
            }
        }

        // Iteratively merge the most frequent pair until the target vocab size is reached.
        for merged_id in 256..vocab_size {
            // Pick the highest-frequency pair (ties broken by BTreeSet ordering).
            let best = match self
                .count_to_pairs
                .iter()
                .next_back()
                .and_then(|(_, bucket)| bucket.iter().next())
                .copied()
            {
                Some(p) => p,
                None => break, // no more pairs to merge
            };

            let new_bytes = [
                self.vocab[&best.0].as_slice(),
                self.vocab[&best.1].as_slice(),
            ]
            .concat();
            self.vocab.insert(merged_id, new_bytes);
            self.merge_map.insert(best, merged_id);

            // Collect all locations where this pair appears before mutating chains.
            let locs: Vec<(usize, u32)> = self
                .pair_locs
                .get(&best)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();

            for (si, left) in locs {
                // The node may already have been consumed by an earlier splice this round.
                let left_node = match chains[si].nodes[left as usize] {
                    Some(n) => n,
                    None => continue,
                };
                let right = match left_node.next {
                    Some(r) => r,
                    None => continue,
                };
                let right_node = match chains[si].nodes[right as usize] {
                    Some(n) => n,
                    None => continue,
                };
                if left_node.token_id != best.0 || right_node.token_id != best.1 {
                    continue;
                }

                let prev = left_node.prev;
                let next = right_node.next;
                let prev_id = prev.map(|p| chains[si].nodes[p as usize].unwrap().token_id);
                let next_id = next.map(|n| chains[si].nodes[n as usize].unwrap().token_id);

                let new_pos = chains[si].splice(left, right, merged_id);

                // Remove the consumed pair and fix up the pairs that straddle the merge point.
                self.adjust(best, si, left, -1);
                if let (Some(pid), Some(p)) = (prev_id, prev) {
                    self.adjust((pid, best.0), si, p, -1);
                    self.adjust((pid, merged_id), si, p, 1);
                }
                if let Some(nid) = next_id {
                    self.adjust((best.1, nid), si, right, -1);
                    self.adjust((merged_id, nid), si, new_pos, 1);
                }
            }
        }

        self.chains = chains;
    }

    pub fn encode(&self, doc: impl AsRef<str>) -> Vec<u32> {
        let mut chains = self.split(doc);
        let mut encoded = Vec::new();

        for chain in &mut chains {
            // Repeatedly apply the lowest-ranked (earliest) applicable merge.
            loop {
                let mut best: Option<(u32, u32, u32)> = None; // (merge_id, left_pos, right_pos)
                let mut prev: Option<(u32, u32)> = None; // (pos, token_id)

                for (pos, node) in chain.iter() {
                    if let Some((prev_pos, prev_id)) = prev {
                        let pair = (prev_id, node.token_id);
                        if let Some(&merge_id) = self.merge_map.get(&pair) {
                            if best.map_or(true, |(best_id, _, _)| merge_id < best_id) {
                                best = Some((merge_id, prev_pos, pos));
                            }
                        }
                    }
                    prev = Some((pos, node.token_id));
                }

                match best {
                    None => break,
                    Some((merge_id, left_pos, right_pos)) => {
                        chain.splice(left_pos, right_pos, merge_id);
                    }
                }
            }
            encoded.extend(chain.iter().map(|(_, node)| node.token_id));
        }
        encoded
    }

    pub fn decode(&self, token_ids: Vec<u32>) -> Vec<u8> {
        token_ids
            .iter()
            .flat_map(|id| self.vocab.get(id).into_iter().flatten().copied())
            .collect()
    }

    fn split(&self, doc: impl AsRef<str>) -> Vec<Chain> {
        self.split_pattern
            .find_iter(doc.as_ref())
            .filter_map(|m| m.ok())
            .map(|m| Chain::new(m.as_str().as_bytes()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_splice_updates_links_for_middle_pair() {
        let mut chain = Chain::new(b"abcd");

        let merged_pos = chain.splice(1, 2, 999);
        let nodes: Vec<(u32, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_pos, 4);
        assert_eq!(nodes, vec![(0, b'a' as u32), (4, 999), (3, b'd' as u32)]);
        assert_eq!(chain.head, Some(0));
        assert_eq!(chain.nodes[0].unwrap().next, Some(4));
        assert_eq!(chain.nodes[3].unwrap().prev, Some(4));
    }

    #[test]
    fn train_learns_most_frequent_pair_and_roundtrips() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(257, ["abababa"]);

        assert_eq!(bpe.vocab.get(&256), Some(&b"ab".to_vec()));
        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&256));

        let encoded = bpe.encode("abababa");
        assert_eq!(encoded, vec![256, 256, 256, b'a' as u32]);
        assert_eq!(bpe.decode(encoded), b"abababa");
    }

    #[test]
    fn training_handles_overlapping_pairs_without_corrupting_state() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(258, ["aaaa"]);

        assert_eq!(bpe.vocab.get(&256), Some(&b"aa".to_vec()));
        assert_eq!(bpe.vocab.get(&257), Some(&b"aaaa".to_vec()));
        assert_eq!(bpe.encode("aaaa"), vec![257]);
        assert_eq!(bpe.decode(vec![257]), b"aaaa");
    }

    #[test]
    fn split_pattern_keeps_merges_scoped_to_each_match() {
        let mut bpe = BPE::new("\\S+");
        bpe.train(257, ["go go", "go stop"]);

        assert_eq!(bpe.vocab.get(&256), Some(&b"go".to_vec()));
        assert_eq!(
            bpe.encode("go stop go"),
            vec![256, b's' as u32, b't' as u32, b'o' as u32, b'p' as u32, 256]
        );
        assert_eq!(
            bpe.decode(vec![
                256,
                b's' as u32,
                b't' as u32,
                b'o' as u32,
                b'p' as u32,
                256
            ]),
            b"gostopgo"
        );
    }

    #[test]
    fn empty_inputs_and_unknown_token_ids_are_handled() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(300, std::iter::empty::<&str>());

        assert!(bpe.encode("").is_empty());
        assert!(bpe.decode(Vec::new()).is_empty());
        assert_eq!(bpe.decode(vec![999_999]), Vec::<u8>::new());
        assert!(bpe.merge_map.is_empty());
    }

    #[test]
    fn requesting_base_vocab_size_keeps_byte_level_encoding() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(256, ["banana"]);

        let encoded = bpe.encode("banana");
        assert_eq!(
            encoded,
            b"banana".iter().map(|&b| b as u32).collect::<Vec<_>>()
        );
        assert_eq!(bpe.decode(encoded), b"banana");
        assert!(bpe.merge_map.is_empty());
    }
}
