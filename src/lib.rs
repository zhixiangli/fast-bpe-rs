use fancy_regex::Regex;
use pyo3::exceptions::{PyUnicodeDecodeError, PyValueError};
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const BASE_VOCAB_SIZE: u32 = 256;

type TokenId = u32;
type NodePos = u32;
type Pair = (TokenId, TokenId);
type ChainIndex = usize;
type PairLocations = BTreeSet<(ChainIndex, NodePos)>;

#[derive(Clone, Copy, Debug)]
struct Node {
    token_id: TokenId,
    prev: Option<NodePos>,
    next: Option<NodePos>,
}

#[derive(Debug)]
struct Chain {
    nodes: Vec<Option<Node>>,
    head: Option<NodePos>,
}

impl Chain {
    fn new(bytes: &[u8]) -> Self {
        let len = bytes.len();
        let nodes = bytes
            .iter()
            .enumerate()
            .map(|(index, &byte)| {
                let pos = index as NodePos;
                Some(Node {
                    token_id: TokenId::from(byte),
                    prev: pos.checked_sub(1),
                    next: (index + 1 < len).then_some(pos + 1),
                })
            })
            .collect();

        Self {
            nodes,
            head: (!bytes.is_empty()).then_some(0),
        }
    }

    fn iter(&self) -> impl Iterator<Item = (NodePos, Node)> + '_ {
        let mut current = self.head;
        std::iter::from_fn(move || {
            let pos = current?;
            let node = self.nodes[pos as usize].expect("chain iterator visited a removed node");
            current = node.next;
            Some((pos, node))
        })
    }

    /// Replaces the `[left, right]` pair with a new merged node, returning the new node's position.
    fn splice(&mut self, left: NodePos, right: NodePos, new_token_id: TokenId) -> NodePos {
        let left_node = self.nodes[left as usize].expect("left splice node must exist");
        let right_node = self.nodes[right as usize].expect("right splice node must exist");
        debug_assert_eq!(
            left_node.next,
            Some(right),
            "splice requires adjacent nodes"
        );

        let prev = left_node.prev;
        let next = right_node.next;
        let pos = self.nodes.len() as NodePos;

        if let Some(prev_pos) = prev {
            self.nodes[prev_pos as usize]
                .as_mut()
                .expect("previous splice node must exist")
                .next = Some(pos);
        } else {
            self.head = Some(pos);
        }

        if let Some(next_pos) = next {
            self.nodes[next_pos as usize]
                .as_mut()
                .expect("next splice node must exist")
                .prev = Some(pos);
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

#[derive(Debug)]
pub struct BPE {
    split_pattern: Regex,
    vocab: HashMap<TokenId, Vec<u8>>,  // id -> bytes
    merge_map: HashMap<Pair, TokenId>, // pair -> merged id

    // training-time state
    chains: Vec<Chain>,
    count_to_pairs: BTreeMap<u32, BTreeSet<Pair>>, // frequency -> pairs
    pair_counts: HashMap<Pair, u32>,               // pair -> frequency
    pair_locs: HashMap<Pair, PairLocations>,       // pair -> (chain_idx, node_pos)
}

impl BPE {
    pub fn new(split_pattern: impl AsRef<str>) -> Self {
        Self::try_new(split_pattern).expect("invalid split regex")
    }

    pub fn try_new(split_pattern: impl AsRef<str>) -> Result<Self, fancy_regex::Error> {
        Ok(Self {
            split_pattern: Regex::new(split_pattern.as_ref())?,
            vocab: (0..BASE_VOCAB_SIZE)
                .map(|byte| (byte, vec![byte as u8]))
                .collect(),
            merge_map: HashMap::new(),
            chains: Vec::new(),
            count_to_pairs: BTreeMap::new(),
            pair_counts: HashMap::new(),
            pair_locs: HashMap::new(),
        })
    }

    /// Increment (`delta = 1`) or decrement (`delta = -1`) a pair's frequency and location set.
    fn adjust(&mut self, pair: Pair, chain_index: ChainIndex, pos: NodePos, delta: i32) {
        let old_count = self.pair_counts.get(&pair).copied().unwrap_or_default();
        let new_count = (old_count as i32 + delta) as u32;

        if old_count > 0 {
            let bucket = self
                .count_to_pairs
                .get_mut(&old_count)
                .expect("existing pair count bucket must exist");
            bucket.remove(&pair);
            if bucket.is_empty() {
                self.count_to_pairs.remove(&old_count);
            }
        }

        if new_count == 0 {
            self.pair_counts.remove(&pair);
            self.pair_locs.remove(&pair);
            return;
        }

        self.pair_counts.insert(pair, new_count);
        self.count_to_pairs
            .entry(new_count)
            .or_default()
            .insert(pair);
        let locations = self.pair_locs.entry(pair).or_default();
        if delta > 0 {
            locations.insert((chain_index, pos));
        } else {
            locations.remove(&(chain_index, pos));
        }
    }

    pub fn train(&mut self, vocab_size: TokenId, docs: impl IntoIterator<Item = impl AsRef<str>>) {
        self.reset_training_state();

        for doc in docs {
            self.chains.extend(self.split(doc));
        }

        let mut chains = std::mem::take(&mut self.chains);

        for (chain_index, chain) in chains.iter().enumerate() {
            let nodes: Vec<_> = chain.iter().collect();
            for window in nodes.windows(2) {
                let (left_pos, left_node) = window[0];
                let (_, right_node) = window[1];
                self.adjust(
                    (left_node.token_id, right_node.token_id),
                    chain_index,
                    left_pos,
                    1,
                );
            }
        }

        for merged_id in BASE_VOCAB_SIZE..vocab_size {
            let Some(best_pair) = self
                .count_to_pairs
                .iter()
                .next_back()
                .and_then(|(_, bucket)| bucket.iter().next())
                .copied()
            else {
                break;
            };

            let new_bytes = [
                self.vocab[&best_pair.0].as_slice(),
                self.vocab[&best_pair.1].as_slice(),
            ]
            .concat();
            self.vocab.insert(merged_id, new_bytes);
            self.merge_map.insert(best_pair, merged_id);

            let locations: Vec<_> = self
                .pair_locs
                .get(&best_pair)
                .map(|set| set.iter().copied().collect())
                .unwrap_or_default();

            for (chain_index, left_pos) in locations {
                let Some(left_node) = chains[chain_index].nodes[left_pos as usize] else {
                    continue;
                };
                let Some(right_pos) = left_node.next else {
                    continue;
                };
                let Some(right_node) = chains[chain_index].nodes[right_pos as usize] else {
                    continue;
                };
                if (left_node.token_id, right_node.token_id) != best_pair {
                    continue;
                }

                let prev = left_node.prev;
                let next = right_node.next;
                let prev_id = prev.map(|pos| {
                    chains[chain_index].nodes[pos as usize]
                        .expect("previous node must exist")
                        .token_id
                });
                let next_id = next.map(|pos| {
                    chains[chain_index].nodes[pos as usize]
                        .expect("next node must exist")
                        .token_id
                });

                let new_pos = chains[chain_index].splice(left_pos, right_pos, merged_id);

                self.adjust(best_pair, chain_index, left_pos, -1);
                if let (Some(prev_id), Some(prev_pos)) = (prev_id, prev) {
                    self.adjust((prev_id, best_pair.0), chain_index, prev_pos, -1);
                    self.adjust((prev_id, merged_id), chain_index, prev_pos, 1);
                }
                if let Some(next_id) = next_id {
                    self.adjust((best_pair.1, next_id), chain_index, right_pos, -1);
                    self.adjust((merged_id, next_id), chain_index, new_pos, 1);
                }
            }
        }

        self.chains = chains;
    }

    pub fn encode(&self, doc: impl AsRef<str>) -> Vec<TokenId> {
        let mut chains = self.split(doc);
        let mut encoded = Vec::new();

        for chain in &mut chains {
            loop {
                let mut best: Option<(TokenId, NodePos, NodePos)> = None;
                let mut previous: Option<(NodePos, TokenId)> = None;

                for (pos, node) in chain.iter() {
                    if let Some((prev_pos, prev_id)) = previous {
                        let pair = (prev_id, node.token_id);
                        if let Some(&merge_id) = self.merge_map.get(&pair)
                            && best.is_none_or(|(best_id, _, _)| merge_id < best_id)
                        {
                            best = Some((merge_id, prev_pos, pos));
                        }
                    }
                    previous = Some((pos, node.token_id));
                }

                let Some((merge_id, left_pos, right_pos)) = best else {
                    break;
                };
                chain.splice(left_pos, right_pos, merge_id);
            }

            encoded.extend(chain.iter().map(|(_, node)| node.token_id));
        }

        encoded
    }

    pub fn decode(&self, token_ids: impl IntoIterator<Item = TokenId>) -> Vec<u8> {
        token_ids
            .into_iter()
            .filter_map(|id| self.vocab.get(&id))
            .flat_map(|bytes| bytes.iter().copied())
            .collect()
    }

    fn split(&self, doc: impl AsRef<str>) -> Vec<Chain> {
        self.split_pattern
            .find_iter(doc.as_ref())
            .filter_map(Result::ok)
            .map(|matched| Chain::new(matched.as_str().as_bytes()))
            .collect()
    }

    fn reset_training_state(&mut self) {
        self.vocab.retain(|token_id, _| *token_id < BASE_VOCAB_SIZE);
        self.merge_map.clear();
        self.chains.clear();
        self.count_to_pairs.clear();
        self.pair_counts.clear();
        self.pair_locs.clear();
    }
}

#[pyclass(name = "BPE")]
pub struct PyBPE {
    inner: BPE,
}

#[pymethods]
impl PyBPE {
    #[new]
    #[pyo3(signature = (split_pattern))]
    fn py_new(split_pattern: &str) -> PyResult<Self> {
        let inner = BPE::try_new(split_pattern)
            .map_err(|err| PyValueError::new_err(format!("invalid split regex: {err}")))?;
        Ok(Self { inner })
    }

    fn train(&mut self, vocab_size: TokenId, docs: Vec<String>) {
        self.inner
            .train(vocab_size, docs.iter().map(String::as_str));
    }

    fn encode(&self, doc: &str) -> Vec<TokenId> {
        self.inner.encode(doc)
    }

    fn decode(&self, token_ids: Vec<TokenId>) -> Vec<u8> {
        self.inner.decode(token_ids)
    }

    fn decode_to_string<'py>(
        &self,
        py: Python<'py>,
        token_ids: Vec<TokenId>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bytes = self.inner.decode(token_ids);
        match String::from_utf8(bytes) {
            Ok(text) => Ok(text.into_pyobject(py)?.into_any()),
            Err(err) => Err(PyUnicodeDecodeError::new_err(err.to_string())),
        }
    }
}

#[pymodule]
fn fast_bpe_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBPE>()?;
    Ok(())
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
        assert_eq!(chain.nodes[0].expect("node 0 should exist").next, Some(4));
        assert_eq!(chain.nodes[3].expect("node 3 should exist").prev, Some(4));
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
        bpe.train(BASE_VOCAB_SIZE, ["banana"]);

        let encoded = bpe.encode("banana");
        assert_eq!(
            encoded,
            b"banana"
                .iter()
                .map(|&byte| TokenId::from(byte))
                .collect::<Vec<_>>()
        );
        assert_eq!(bpe.decode(encoded), b"banana");
        assert!(bpe.merge_map.is_empty());
    }

    #[test]
    fn retraining_replaces_previous_merge_state() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(257, ["aaaa"]);
        let first_encoding = bpe.encode("aaaa");
        assert_eq!(first_encoding, vec![256, 256]);

        bpe.train(257, ["bbbb"]);

        assert_eq!(bpe.vocab.get(&256), Some(&b"bb".to_vec()));
        assert_eq!(bpe.encode("bbbb"), vec![256, 256]);
        assert_eq!(bpe.encode("aaaa"), vec![b'a' as u32; 4]);
    }

    #[test]
    fn try_new_returns_error_for_invalid_regex() {
        assert!(BPE::try_new("(").is_err());
    }
}
