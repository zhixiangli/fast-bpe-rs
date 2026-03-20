use fancy_regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy)]
struct Node {
    id: u32,
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
                        id: b as u32,
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

    pub fn splice(&mut self, left: u32, right: u32, new_id: u32) -> u32 {
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
            id: new_id,
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
    vocab: HashMap<u32, Vec<u8>>,        // id  -> bytes
    merge_map: HashMap<(u32, u32), u32>, // token_ids pair -> merged new token id

    // training-time state
    chains: Vec<Chain>,

    count_to_pairs: BTreeMap<u32, BTreeSet<(u32, u32)>>, // frequency -> token_ids pair
    pair_counts: HashMap<(u32, u32), u32>,               // token_ids pair -> frequency
    pair_locs: HashMap<(u32, u32), BTreeSet<(u32, u32)>>,
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

    pub fn train<I, S>(&mut self, vocab_size: u32, docs: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for doc in docs {
            self.chains.extend(self.split(doc));
        }

        for i in 0..self.chains.len() {
            // let chain = self.chains[i];
            // let mut last = None;
            // for (pos, node) in chain.iter() {
            // curr = node;
            // if last is not None {
            // let pair = (last.id, curr.id);
            // TODO
            // increase the frequencies
            // update the locations
            // }
            // last = curr;
            // }
        }

        let mut next_id = 256;
        for _ in 0..vocab_size - 256 {
            let best_pair = match self.count_to_pairs.iter().next_back() {
                None => break,
                Some((_, bucket)) => *bucket.iter().next().unwrap(),
            };
            // let new_token = FastToken::new({
            //     token_id: next_id;
            //     bytes: // merge
            // });
            // TODO: replace all best pair with new_token

            next_id += 1;
        }
    }

    pub fn encode(&self, doc: impl AsRef<str>) -> Vec<u32> {
        let mut chains = self.split(doc);
        let mut encoded = Vec::new();
        for chain in &mut chains {
            loop {
                let mut best: Option<(u32, u32, u32)> = None; // (merge_id, left_pos, right_pos)
                let mut prev: Option<(u32, u32)> = None; // (pos, id)

                for (pos, node) in chain.iter() {
                    if let Some((prev_pos, prev_id)) = prev {
                        let pair = (prev_id, node.id);
                        if let Some(&merge_id) = self.merge_map.get(&pair) {
                            if best.map_or(true, |(best_id, _, _)| merge_id < best_id) {
                                best = Some((merge_id, prev_pos, pos));
                            }
                        }
                    }
                    prev = Some((pos, node.id));
                }

                match best {
                    None => break, // no more mergeable pairs → done
                    Some((merge_id, left_pos, right_pos)) => {
                        chain.splice(left_pos, right_pos, merge_id);
                    }
                }
            }
            encoded.extend(chain.iter().map(|(_, node)| node.id));
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
