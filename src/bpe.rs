use crate::chain::Chain;
use crate::types::{BASE_VOCAB_SIZE, ChainIndex, NodePos, Pair, PairLocations, TokenId};
use fancy_regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug)]
pub struct BPE {
    split_pattern: Regex,
    pub(crate) vocab: HashMap<TokenId, Vec<u8>>, // id -> bytes
    pub(crate) merge_map: HashMap<Pair, TokenId>, // pair -> merged id

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
                let Some(left_node) = chains[chain_index].nodes[left_pos] else {
                    continue;
                };
                let Some(right_pos) = left_node.next else {
                    continue;
                };
                let Some(right_node) = chains[chain_index].nodes[right_pos] else {
                    continue;
                };
                if (left_node.token_id, right_node.token_id) != best_pair {
                    continue;
                }

                let prev = left_node.prev;
                let next = right_node.next;
                let prev_id = prev.map(|pos| {
                    chains[chain_index].nodes[pos]
                        .expect("previous node must exist")
                        .token_id
                });
                let next_id = next.map(|pos| {
                    chains[chain_index].nodes[pos]
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
