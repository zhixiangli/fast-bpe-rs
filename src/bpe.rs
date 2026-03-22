use crate::chain::Chain;
use crate::types::{BASE_VOCAB_SIZE, ChainIndex, NodePos, Pair, PairLocations, TokenId};
use fancy_regex::{Regex, escape};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

/// One unique training chunk plus the number of corpus occurrences it represents.
#[derive(Debug)]
struct WeightedChain {
    chain: Chain,
    frequency: u32,
}

/// Errors that can occur while constructing a [`BPE`] instance.
#[derive(Debug)]
pub enum BPEError {
    /// The caller supplied an invalid regex for splitting documents into chunks.
    InvalidSplitRegex(fancy_regex::Error),
    /// A special token attempted to reuse an id reserved for raw byte values.
    SpecialTokenIdOverlapsBaseVocab { token: String, token_id: TokenId },
    /// Two different special tokens were assigned the same token id.
    DuplicateSpecialTokenId { token_id: TokenId },
}

impl fmt::Display for BPEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSplitRegex(err) => write!(f, "invalid split regex: {err}"),
            Self::SpecialTokenIdOverlapsBaseVocab { token, token_id } => write!(
                f,
                "special token {token:?} uses reserved token id {token_id}; special token ids must be >= {BASE_VOCAB_SIZE}"
            ),
            Self::DuplicateSpecialTokenId { token_id } => {
                write!(f, "special token id {token_id} is assigned more than once")
            }
        }
    }
}

impl std::error::Error for BPEError {}

impl From<fancy_regex::Error> for BPEError {
    fn from(value: fancy_regex::Error) -> Self {
        Self::InvalidSplitRegex(value)
    }
}

/// Byte Pair Encoding model with optional special-token support.
///
/// The implementation keeps training-time state in linked-list-like chains so that merges can be
/// applied without rebuilding every token sequence from scratch.
#[derive(Debug)]
pub struct BPE {
    /// Regex used to split ordinary text into independently-mergeable spans.
    split_pattern: Regex,
    /// Regex that spots special tokens before normal splitting is applied.
    special_split_pattern: Option<Regex>,
    /// Mapping from literal special-token text to the externally-visible token id.
    special_tokens: HashMap<String, TokenId>,
    /// Vocabulary entries materialized as raw bytes for decode/inspection.
    pub(crate) vocab: HashMap<TokenId, Vec<u8>>, // id -> bytes
    /// Learned merge rules keyed by `(left, right)` token pairs.
    pub(crate) merge_map: HashMap<Pair, TokenId>, // pair -> merged id

    // Training-time index structures.
    //
    // `chains` stores each unique training chunk once alongside its corpus frequency.
    // `pair_counts`, `count_to_pairs`, and `pair_locs` together act like an indexed priority queue
    // that answers two questions efficiently:
    //   1. "Which pair is currently most frequent?"
    //   2. "Where does that pair appear so we can rewrite those locations?"
    chains: Vec<WeightedChain>,
    count_to_pairs: Vec<BTreeSet<Pair>>,     // frequency -> pairs
    max_pair_count: u32,                     // largest non-empty frequency bucket
    pair_counts: HashMap<Pair, u32>,         // pair -> frequency
    pair_locs: HashMap<Pair, PairLocations>, // pair -> (chain_idx, node_pos)
}

impl BPE {
    /// Constructs a model and panics if the split regex is invalid.
    pub fn new(split_pattern: impl AsRef<str>) -> Self {
        Self::try_new(split_pattern).expect("invalid split regex")
    }

    /// Constructs a model with special tokens and panics on invalid configuration.
    pub fn new_with_special_tokens(
        split_pattern: impl AsRef<str>,
        special_tokens: impl IntoIterator<Item = (impl Into<String>, TokenId)>,
    ) -> Self {
        Self::try_new_with_special_tokens(split_pattern, special_tokens)
            .expect("invalid split regex or special token configuration")
    }

    /// Fallible constructor without special tokens.
    pub fn try_new(split_pattern: impl AsRef<str>) -> Result<Self, BPEError> {
        Self::try_new_with_special_tokens(split_pattern, std::iter::empty::<(String, TokenId)>())
    }

    /// Fallible constructor with special tokens.
    ///
    /// Special-token ids must live above the byte vocabulary so the base 0..=255 range remains a
    /// lossless representation of raw bytes.
    pub fn try_new_with_special_tokens(
        split_pattern: impl AsRef<str>,
        special_tokens: impl IntoIterator<Item = (impl Into<String>, TokenId)>,
    ) -> Result<Self, BPEError> {
        let split_pattern = Regex::new(split_pattern.as_ref()).map_err(BPEError::from)?;
        let mut vocab: HashMap<TokenId, Vec<u8>> = (0..BASE_VOCAB_SIZE)
            .map(|byte| (byte, vec![byte as u8]))
            .collect();
        let mut special_token_map = HashMap::new();
        let mut used_special_ids = HashSet::new();

        for (token, token_id) in special_tokens {
            let token = token.into();
            if token_id < BASE_VOCAB_SIZE {
                return Err(BPEError::SpecialTokenIdOverlapsBaseVocab { token, token_id });
            }
            if !used_special_ids.insert(token_id) {
                return Err(BPEError::DuplicateSpecialTokenId { token_id });
            }
            vocab.insert(token_id, token.as_bytes().to_vec());
            special_token_map.insert(token, token_id);
        }

        let special_split_pattern = Self::build_special_token_pattern(&special_token_map)
            .as_deref()
            .map(Regex::new)
            .transpose()
            .map_err(BPEError::from)?;

        Ok(Self {
            split_pattern,
            special_split_pattern,
            special_tokens: special_token_map,
            vocab,
            merge_map: HashMap::new(),
            chains: Vec::new(),
            count_to_pairs: vec![BTreeSet::new()],
            max_pair_count: 0,
            pair_counts: HashMap::new(),
            pair_locs: HashMap::new(),
        })
    }

    /// Builds an alternation regex that matches special tokens from longest to shortest.
    ///
    /// Ordering by length prevents a short token like `<e>` from stealing the prefix of a longer
    /// token like `<eos>`.
    fn build_special_token_pattern(special_tokens: &HashMap<String, TokenId>) -> Option<String> {
        if special_tokens.is_empty() {
            return None;
        }

        let mut special_tokens: Vec<_> = special_tokens.keys().collect();
        special_tokens
            .sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
        Some(
            special_tokens
                .into_iter()
                .map(|token| escape(token))
                .collect::<Vec<_>>()
                .join("|"),
        )
    }

    /// Updates the frequency tables and occurrence set for one adjacent token pair.
    ///
    /// Conceptually the bookkeeping looks like this:
    ///
    /// ```text
    /// pair_counts[(A, B)] = 3
    /// count_to_pairs[3]   = {(A, B), ...}
    /// pair_locs[(A, B)]   = {(chain 0, pos 4), (chain 2, pos 1), ...}
    /// ```
    ///
    /// When a merge removes or creates a pair we update all three views together so the next
    /// training step can still find the globally most frequent pair in `O(log n)` map time.
    fn adjust(&mut self, pair: Pair, chain_index: ChainIndex, pos: NodePos, delta: i64) {
        let old_count = self.pair_counts.get(&pair).copied().unwrap_or_default();
        let new_count_i64 = i64::from(old_count) + delta;
        debug_assert!(new_count_i64 >= 0, "pair counts must stay non-negative");
        let new_count =
            u32::try_from(new_count_i64).expect("pair counts must fit in u32 after adjustment");

        if old_count > 0 {
            let bucket = self
                .count_to_pairs
                .get_mut(old_count as usize)
                .expect("existing pair count bucket must exist");
            bucket.remove(&pair);
            if self.max_pair_count == old_count && bucket.is_empty() {
                while self.max_pair_count > 0
                    && self.count_to_pairs[self.max_pair_count as usize].is_empty()
                {
                    self.max_pair_count -= 1;
                }
            }
        }

        if new_count == 0 {
            self.pair_counts.remove(&pair);
            self.pair_locs.remove(&pair);
            return;
        }

        self.pair_counts.insert(pair, new_count);
        if self.count_to_pairs.len() <= new_count as usize {
            self.count_to_pairs
                .resize_with(new_count as usize + 1, BTreeSet::new);
        }
        self.count_to_pairs[new_count as usize].insert(pair);
        self.max_pair_count = self.max_pair_count.max(new_count);

        let locations = self.pair_locs.entry(pair).or_default();
        if delta > 0 {
            locations.insert((chain_index, pos));
        } else {
            locations.remove(&(chain_index, pos));
        }
    }

    /// Learns merge rules until `vocab_size` is reached or no mergeable pair remains.
    ///
    /// High-level training loop:
    ///
    /// ```text
    /// docs --split--> chains of byte tokens
    ///      --count--> most frequent adjacent pair
    ///      --merge--> rewrite every occurrence in-place
    ///      --repeat-> until vocabulary is full
    /// ```
    pub fn train(&mut self, vocab_size: TokenId, docs: impl IntoIterator<Item = impl AsRef<str>>) {
        self.reset_training_state();

        let mut chain_indexes = HashMap::<Vec<TokenId>, ChainIndex>::new();
        for doc in docs {
            for chain in self.split(doc) {
                let signature = chain.token_ids();
                if let Some(&chain_index) = chain_indexes.get(&signature) {
                    self.chains[chain_index].frequency += 1;
                } else {
                    let chain_index = self.chains.len();
                    self.chains.push(WeightedChain {
                        chain,
                        frequency: 1,
                    });
                    chain_indexes.insert(signature, chain_index);
                }
            }
        }

        // Take ownership so we can mutate chains freely while still updating `self`'s indexes.
        let mut chains = std::mem::take(&mut self.chains);

        // Seed the pair-frequency index with every currently-adjacent pair without
        // materializing an extra per-chain node snapshot.
        for (chain_index, weighted_chain) in chains.iter().enumerate() {
            let frequency = i64::from(weighted_chain.frequency);
            let mut previous = None;
            for (pos, node) in weighted_chain.chain.iter() {
                if let Some((left_pos, left_id)) = previous {
                    self.adjust((left_id, node.token_id), chain_index, left_pos, frequency);
                }
                previous = Some((pos, node.token_id));
            }
        }

        let reserved_ids: HashSet<_> = self.special_tokens.values().copied().collect();
        for merged_id in
            (BASE_VOCAB_SIZE..vocab_size).filter(|token_id| !reserved_ids.contains(token_id))
        {
            let Some(best_pair) = self
                .count_to_pairs
                .get(self.max_pair_count as usize)
                .and_then(|bucket| bucket.iter().next())
                .copied()
            else {
                break;
            };

            // Materialize the merged token bytes so decoding remains a simple table lookup.
            let new_bytes = [
                self.vocab[&best_pair.0].as_slice(),
                self.vocab[&best_pair.1].as_slice(),
            ]
            .concat();
            self.vocab.insert(merged_id, new_bytes);
            self.merge_map.insert(best_pair, merged_id);

            // Drain locations incrementally so training does not duplicate the full occurrence
            // set for the hottest pair in a temporary `Vec`.
            while let Some((chain_index, left_pos)) = self
                .pair_locs
                .get(&best_pair)
                .and_then(|locations| locations.first().copied())
            {
                let frequency = i64::from(chains[chain_index].frequency);
                let Some(left_node) = chains[chain_index].chain.nodes[left_pos] else {
                    continue;
                };
                let Some(right_pos) = left_node.next else {
                    continue;
                };
                let Some(right_node) = chains[chain_index].chain.nodes[right_pos] else {
                    continue;
                };
                if (left_node.token_id, right_node.token_id) != best_pair {
                    continue;
                }

                // Snapshot the neighborhood before splicing:
                //
                //   prev -> [left] [right] -> next
                //            \____merge____/
                //
                // After splicing we must remove the old adjacent pairs and add the new ones that
                // touch the merged token.
                let prev = left_node.prev;
                let next = right_node.next;
                let prev_id = prev.map(|pos| {
                    chains[chain_index].chain.nodes[pos]
                        .expect("previous node must exist")
                        .token_id
                });
                let next_id = next.map(|pos| {
                    chains[chain_index].chain.nodes[pos]
                        .expect("next node must exist")
                        .token_id
                });

                let new_pos = chains[chain_index]
                    .chain
                    .splice(left_pos, right_pos, merged_id);

                self.adjust(best_pair, chain_index, left_pos, -frequency);
                if let (Some(prev_id), Some(prev_pos)) = (prev_id, prev) {
                    self.adjust((prev_id, best_pair.0), chain_index, prev_pos, -frequency);
                    self.adjust((prev_id, merged_id), chain_index, prev_pos, frequency);
                }
                if let Some(next_id) = next_id {
                    self.adjust((best_pair.1, next_id), chain_index, right_pos, -frequency);
                    self.adjust((merged_id, next_id), chain_index, new_pos, frequency);
                }
            }
        }

        self.chains = chains;
    }

    /// Encodes text by repeatedly applying the learned merge with the lowest token id available.
    ///
    /// Using the lowest merge id reproduces the same merge priority as training order: earlier
    /// merges receive smaller ids and therefore win when multiple adjacent pairs are possible.
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

    /// Decodes token ids back into bytes, skipping unknown ids.
    pub fn decode(&self, token_ids: impl IntoIterator<Item = TokenId>) -> Vec<u8> {
        token_ids
            .into_iter()
            .filter_map(|id| self.vocab.get(&id))
            .flat_map(|bytes| bytes.iter().copied())
            .collect()
    }

    /// Splits input text into independently-encodable chains.
    ///
    /// Special tokens act like pre-tokenized islands. Everything between them is split by the
    /// normal regex, but the special-token literals themselves become single-node chains so they
    /// never merge with surrounding text.
    fn split(&self, doc: impl AsRef<str>) -> Vec<Chain> {
        let doc = doc.as_ref();
        let mut chains = Vec::new();
        let mut cursor = 0;

        if let Some(special_split_pattern) = &self.special_split_pattern {
            for matched in special_split_pattern.find_iter(doc).filter_map(Result::ok) {
                chains.extend(
                    self.split_pattern
                        .find_iter(&doc[cursor..matched.start()])
                        .filter_map(Result::ok)
                        .map(|matched| Chain::new(matched.as_str().as_bytes())),
                );

                if let Some(&token_id) = self.special_tokens.get(matched.as_str()) {
                    chains.push(Chain::from_token_id(token_id));
                }

                cursor = matched.end();
            }
        }

        chains.extend(
            self.split_pattern
                .find_iter(&doc[cursor..])
                .filter_map(Result::ok)
                .map(|matched| Chain::new(matched.as_str().as_bytes())),
        );
        chains
    }

    /// Clears all learned merges while preserving the immutable base vocabulary.
    fn reset_training_state(&mut self) {
        self.vocab.retain(|token_id, _| {
            *token_id < BASE_VOCAB_SIZE
                || self
                    .special_tokens
                    .values()
                    .any(|special_id| special_id == token_id)
        });
        self.merge_map.clear();
        self.chains.clear();
        self.count_to_pairs.clear();
        self.count_to_pairs.push(BTreeSet::new());
        self.max_pair_count = 0;
        self.pair_counts.clear();
        self.pair_locs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn train_aggregates_identical_chunks_and_weights_pair_counts() {
        let mut bpe = BPE::new("\\S+");
        bpe.train(258, ["the the", "the"]);

        assert_eq!(bpe.chains.len(), 1);
        assert_eq!(bpe.chains[0].frequency, 3);
        assert_eq!(bpe.chains[0].chain.token_ids(), vec![257]);
        assert_eq!(bpe.vocab.get(&256), Some(&b"he".to_vec()));
        assert_eq!(bpe.vocab.get(&257), Some(&b"the".to_vec()));
        assert_eq!(bpe.pair_counts.get(&(b't' as u32, b'h' as u32)), None);
        assert_eq!(bpe.encode("the the"), vec![257, 257]);
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
    fn training_with_empty_input_keeps_base_state() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(300, std::iter::empty::<&str>());

        assert!(bpe.encode("").is_empty());
        assert!(bpe.decode(Vec::new()).is_empty());
        assert_eq!(bpe.decode(vec![999_999]), Vec::<u8>::new());
        assert!(bpe.merge_map.is_empty());
        assert!(bpe.chains.is_empty());
        assert_eq!(bpe.count_to_pairs.len(), 1);
        assert!(bpe.count_to_pairs[0].is_empty());
        assert!(bpe.pair_counts.is_empty());
        assert!(bpe.pair_locs.is_empty());
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
        assert_eq!(bpe.vocab.len() as u32, BASE_VOCAB_SIZE + 1);
    }

    #[test]
    fn encode_prefers_lowest_merge_id_when_multiple_pairs_match() {
        let mut bpe = BPE::new("(?s).+");
        bpe.train(259, ["abba"]);

        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&256));
        assert_eq!(bpe.merge_map.get(&(b'b' as u32, b'a' as u32)), Some(&257));
        assert_eq!(bpe.merge_map.get(&(256, 257)), Some(&258));
        assert_eq!(bpe.encode("abba"), vec![258]);
    }

    #[test]
    fn try_new_with_special_tokens_rejects_ids_in_base_vocab() {
        let err = BPE::try_new_with_special_tokens("(?s).+", [("<pad>", 42)])
            .expect_err("special token ids below the byte vocabulary should be rejected");

        assert!(matches!(
            err,
            BPEError::SpecialTokenIdOverlapsBaseVocab { token_id: 42, .. }
        ));
        assert!(err.to_string().contains("must be >= 256"));
    }

    #[test]
    fn try_new_with_special_tokens_rejects_duplicate_ids() {
        let err = BPE::try_new_with_special_tokens("(?s).+", [("<pad>", 512), ("<eos>", 512)])
            .expect_err("duplicate special token ids should be rejected");

        assert!(matches!(
            err,
            BPEError::DuplicateSpecialTokenId { token_id: 512 }
        ));
        assert!(err.to_string().contains("assigned more than once"));
    }

    #[test]
    fn try_new_returns_error_for_invalid_regex() {
        assert!(BPE::try_new("(").is_err());
    }

    #[test]
    fn special_tokens_keep_custom_ids_and_roundtrip_without_splitting() {
        let bpe = BPE::new_with_special_tokens("\\S+", [("<pad>", 1024), ("<eos>", 2048)]);

        assert_eq!(
            bpe.encode("hi<pad><eos>there"),
            vec![
                b'h' as u32,
                b'i' as u32,
                1024,
                2048,
                b't' as u32,
                b'h' as u32,
                b'e' as u32,
                b'r' as u32,
                b'e' as u32,
            ]
        );
        assert_eq!(bpe.decode([1024, 2048]), b"<pad><eos>");
    }

    #[test]
    fn special_tokens_do_not_merge_with_neighbors_or_each_other() {
        let mut bpe = BPE::new_with_special_tokens("(?s).+", [("<pad>", 300), ("<eos>", 301)]);
        bpe.train(305, ["a<pad>a<pad>", "<pad><eos><pad>"]);

        assert!(
            !bpe.merge_map
                .keys()
                .any(|pair| pair.0 == 300 || pair.1 == 300 || pair.0 == 301 || pair.1 == 301)
        );
        assert_eq!(
            bpe.encode("a<pad><eos>a"),
            vec![b'a' as u32, 300, 301, b'a' as u32]
        );
    }
}
