use crate::chain::Chain;
use crate::error::BPEError;
use crate::types::{
    BASE_VOCAB_SIZE, ChainIndex, NodePos, PairLocations, SeedMap, TokenId, TokenIdPair,
};
use ahash::AHashMap;
use fancy_regex::{Regex, escape};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// One unique training chunk plus the number of corpus occurrences it represents.
#[derive(Debug)]
struct WeightedChain {
    chain: Chain,
    frequency: u32,
}

#[derive(Debug, Default)]
struct TokenIdPairInfo {
    count: u32,
    locations: PairLocations,
}

enum Segment<'a> {
    Regular(&'a str),
    Special(&'a str),
}

/// Byte Pair Encoding model with optional special-token support.
///
/// The core design treats each tokenized chunk as a mutable linked structure (`Chain`) and tracks
/// adjacent token-id-pair statistics in a frequency-indexed table. This avoids repeated full rescans of the
/// corpus during training: after each merge, only neighborhoods around changed nodes are updated.
///
/// Training data flow (coarse):
///
/// ```text
/// docs
///   |
///   v
/// split_for_training_bytes (regex chunks; special tokens are boundaries only)
///   |
///   v
/// build_training_chains (deduplicate chunks -> weighted chains)
///   |
///   v
/// build_initial_token_id_pair_stats (token-id pair -> {count, locations})
///   |
///   v
/// repeat until vocab full:
///   pick highest-frequency token-id pair
///   materialize merged token bytes
///   splice every recorded occurrence in-place
///   adjust only affected neighboring token-id pairs
/// ```
#[derive(Debug)]
pub struct BPE {
    /// Regex used to split ordinary text into independently-mergeable spans.
    split_pattern: Regex,
    /// Source pattern used to build worker-local split regexes during parallel training.
    split_pattern_source: String,
    /// Regex that spots special tokens before normal splitting is applied.
    special_split_pattern: Option<Regex>,
    /// Source pattern used to build worker-local special-token regexes during parallel training.
    special_split_pattern_source: Option<String>,
    /// Mapping from literal special-token text to the externally-visible token id.
    special_tokens: HashMap<String, TokenId>,
    /// Vocabulary entries materialized as raw bytes for decode/inspection.
    pub(crate) vocab: HashMap<TokenId, Vec<u8>>, // id -> bytes
    /// Learned merge rules keyed by `(left, right)` token pairs.
    pub(crate) merge_map: HashMap<TokenIdPair, TokenId>, // token-id pair -> merged id

    // Training-time index structures.
    //
    // `chains` stores each unique training chunk once with its corpus multiplicity.
    // `token_id_pair_info` + `count_to_token_id_pairs` form a bidirectional index:
    //
    //   token-id pair ----------------------> metadata
    //   (A,B)       token_id_pair_info[(A,B)] = {count, locations}
    //
    //   count -----------------------------> set of token-id pairs at that count
    //   17          count_to_token_id_pairs[17] = {(A,B), (C,D), ...}
    //
    // `max_token_id_pair_count` points to the highest non-empty count bucket, so selecting the current
    // best merge is an O(1)-expected bucket lookup plus tie-break in that bucket.
    chains: Vec<WeightedChain>,
    count_to_token_id_pairs: Vec<FxHashSet<TokenIdPair>>, // frequency -> token-id pairs
    max_token_id_pair_count: u32,                         // largest non-empty frequency bucket
    token_id_pair_info: FxHashMap<TokenIdPair, TokenIdPairInfo>, // token-id pair -> {frequency, (chain_idx, node_pos)}
}

impl BPE {
    pub const DEFAULT_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

    fn split_matches<'a>(&'a self, segment: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.split_pattern
            .find_iter(segment)
            .map(|matched| matched.expect("split regex evaluation should succeed"))
            .map(|matched| matched.as_str())
    }

    fn for_each_segment_between_specials(
        &self,
        doc: &str,
        mut on_segment: impl FnMut(Segment<'_>),
    ) {
        let mut cursor = 0;
        if let Some(special_split_pattern) = &self.special_split_pattern {
            for matched in special_split_pattern
                .find_iter(doc)
                .map(|matched| matched.expect("special token regex evaluation should succeed"))
            {
                on_segment(Segment::Regular(&doc[cursor..matched.start()]));
                on_segment(Segment::Special(matched.as_str()));
                cursor = matched.end();
            }
        }
        on_segment(Segment::Regular(&doc[cursor..]));
    }

    /// Splits input text into byte chunks for training, using special tokens only as boundaries.
    ///
    /// Training learns merges from raw byte spans, so special tokens are excluded rather than
    /// materialized as token ids.
    #[cfg(test)]
    fn split_for_training_bytes(&self, doc: impl AsRef<str>) -> Vec<SmallVec<[u8; 16]>> {
        Self::split_for_training_bytes_with_patterns(
            doc,
            &self.split_pattern,
            self.special_split_pattern.as_ref(),
        )
    }

    fn split_for_training_bytes_with_patterns(
        doc: impl AsRef<str>,
        split_pattern: &Regex,
        special_split_pattern: Option<&Regex>,
    ) -> Vec<SmallVec<[u8; 16]>> {
        let doc = doc.as_ref();
        let mut chunks = Vec::new();
        let mut cursor = 0;

        if let Some(special_split_pattern) = special_split_pattern {
            for matched in special_split_pattern
                .find_iter(doc)
                .map(|matched| matched.expect("special token regex evaluation should succeed"))
            {
                chunks.extend(
                    split_pattern
                        .find_iter(&doc[cursor..matched.start()])
                        .map(|matched| matched.expect("split regex evaluation should succeed"))
                        .map(|matched| SmallVec::from_slice(matched.as_str().as_bytes())),
                );
                cursor = matched.end();
            }
        }
        chunks.extend(
            split_pattern
                .find_iter(&doc[cursor..])
                .map(|matched| matched.expect("split regex evaluation should succeed"))
                .map(|matched| SmallVec::from_slice(matched.as_str().as_bytes())),
        );
        chunks
    }

    /// Parallel fold+reduce over all docs, producing byte-chunk frequencies without allocating
    /// duplicate chains for repeated chunks.
    fn build_training_chains(
        &self,
        docs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Vec<WeightedChain> {
        let docs: Vec<String> = docs
            .into_iter()
            .map(|doc| doc.as_ref().to_owned())
            .collect();
        let split_pattern_source = self.split_pattern_source.clone();
        let special_split_pattern_source = self.special_split_pattern_source.clone();

        docs.par_iter()
            .fold(
                || {
                    (
                        AHashMap::<SmallVec<[u8; 16]>, u32>::new(),
                        Regex::new(&split_pattern_source)
                            .expect("split regex source should remain valid"),
                        special_split_pattern_source.as_ref().map(|pattern| {
                            Regex::new(pattern)
                                .expect("special token regex source should remain valid")
                        }),
                    )
                },
                |(mut local_counts, split_pattern, special_split_pattern), doc| {
                    for chunk in Self::split_for_training_bytes_with_patterns(
                        doc,
                        &split_pattern,
                        special_split_pattern.as_ref(),
                    ) {
                        *local_counts.entry(chunk).or_default() += 1;
                    }
                    (local_counts, split_pattern, special_split_pattern)
                },
            )
            .reduce(
                || {
                    (
                        AHashMap::<SmallVec<[u8; 16]>, u32>::new(),
                        Regex::new(&split_pattern_source)
                            .expect("split regex source should remain valid"),
                        special_split_pattern_source.as_ref().map(|pattern| {
                            Regex::new(pattern)
                                .expect("special token regex source should remain valid")
                        }),
                    )
                },
                |(mut global_counts, global_split_pattern, global_special_split_pattern),
                 (local_counts, _, _)| {
                    for (chunk, frequency) in local_counts {
                        *global_counts.entry(chunk).or_default() += frequency;
                    }
                    (
                        global_counts,
                        global_split_pattern,
                        global_special_split_pattern,
                    )
                },
            )
            .0
            .into_iter()
            .map(|(chunk, frequency)| WeightedChain {
                chain: Chain::new(&chunk),
                frequency,
            })
            .collect()
    }

    /// Counts every adjacent token-id pair in parallel and records all locations where each pair appears.
    ///
    /// The resulting map is used to seed `token_id_pair_info` and `count_to_token_id_pairs` in one
    /// pass with one bucket insertion per unique token-id pair.
    fn build_initial_token_id_pair_stats(
        chains: &[WeightedChain],
    ) -> SeedMap<(u32, Vec<(ChainIndex, NodePos)>)> {
        chains
            .par_iter()
            .enumerate()
            .fold(
                SeedMap::<(u32, Vec<(ChainIndex, NodePos)>)>::default,
                |mut local_token_id_pairs, (chain_index, weighted_chain)| {
                    let token_id_pair_capacity = weighted_chain.chain.nodes.len().saturating_sub(1);
                    if token_id_pair_capacity > 0 {
                        local_token_id_pairs.reserve(token_id_pair_capacity);
                    }

                    let frequency = weighted_chain.frequency;
                    let mut previous = None;
                    for (pos, node) in weighted_chain.chain.iter() {
                        if let Some((left_pos, left_id)) = previous {
                            let token_id_pair = (left_id, node.token_id);
                            let (count, locations) = local_token_id_pairs
                                .entry(token_id_pair)
                                .or_insert_with(|| (0, Vec::with_capacity(1)));
                            *count += frequency;
                            locations.push((chain_index, left_pos));
                        }
                        previous = Some((pos, node.token_id));
                    }
                    local_token_id_pairs
                },
            )
            .reduce(
                SeedMap::<(u32, Vec<(ChainIndex, NodePos)>)>::default,
                |mut global_token_id_pairs, local_token_id_pairs| {
                    global_token_id_pairs.reserve(local_token_id_pairs.len());
                    for (token_id_pair, (count, mut locations)) in local_token_id_pairs {
                        let (global_count, global_locations) = global_token_id_pairs
                            .entry(token_id_pair)
                            .or_insert_with(|| (0, Vec::with_capacity(locations.len())));
                        *global_count += count;
                        global_locations.append(&mut locations);
                    }
                    global_token_id_pairs
                },
            )
    }

    /// Constructs a model with optional split regex and optional special tokens.
    ///
    /// When `split_pattern` is `None`, the default split regex is used.
    /// When `special_tokens` is `None`, no special tokens are configured.
    ///
    /// Special-token ids must live above the byte vocabulary so the base 0..=255 range remains a
    /// lossless representation of raw bytes.
    pub fn new(
        split_pattern: Option<&str>,
        special_tokens: Option<impl IntoIterator<Item = (impl Into<String>, TokenId)>>,
    ) -> Result<Self, BPEError> {
        let split_pattern_source = split_pattern
            .unwrap_or(Self::DEFAULT_SPLIT_PATTERN)
            .to_owned();
        let split_pattern = Regex::new(&split_pattern_source).map_err(BPEError::from)?;
        let mut vocab: HashMap<TokenId, Vec<u8>> = (0..BASE_VOCAB_SIZE)
            .map(|byte| (byte, vec![byte as u8]))
            .collect();
        let mut special_token_map = HashMap::new();
        let mut used_special_ids = HashSet::new();

        for (token, token_id) in special_tokens
            .map(IntoIterator::into_iter)
            .into_iter()
            .flatten()
        {
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

        let special_split_pattern_source = Self::build_special_token_pattern(&special_token_map);
        let special_split_pattern = special_split_pattern_source
            .as_deref()
            .map(Regex::new)
            .transpose()
            .map_err(BPEError::from)?;

        Ok(Self {
            split_pattern,
            split_pattern_source,
            special_split_pattern,
            special_split_pattern_source,
            special_tokens: special_token_map,
            vocab,
            merge_map: HashMap::new(),
            chains: Vec::new(),
            count_to_token_id_pairs: vec![FxHashSet::default()],
            max_token_id_pair_count: 0,
            token_id_pair_info: FxHashMap::default(),
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

    /// Updates token-id-pair frequency and location indexes after one local topology change.
    ///
    /// `delta` is weighted by the source chunk frequency, so one splice in a chain that
    /// represents N duplicate corpus chunks contributes N increments/decrements.
    ///
    /// Index-mutation flow:
    ///
    /// ```text
    /// input: (token_id_pair, location, delta)
    ///   |
    ///   v
    /// mutate token_id_pair_info[token_id_pair]:
    ///   count += delta
    ///   locations +/- (chain_index, pos)
    ///   |
    ///   +--> old_count bucket: remove token_id_pair
    ///   |
    ///   +--> if new_count == 0:
    ///   |      delete token_id_pair_info entry
    ///   |      shrink max_token_id_pair_count downward if needed
    ///   |
    ///   +--> else:
    ///          insert token_id_pair into count_to_token_id_pairs[new_count]
    ///          raise max_token_id_pair_count if new bucket is higher
    /// ```
    ///
    /// Keeping both directions synchronized prevents stale bucket membership and ensures the
    /// "next best token-id pair" query remains unambiguous after overlapping merges.
    fn adjust(
        &mut self,
        token_id_pair: TokenIdPair,
        chain_index: ChainIndex,
        pos: NodePos,
        delta: i64,
    ) {
        let (old_count, new_count) = {
            let token_id_pair_info = self.token_id_pair_info.entry(token_id_pair).or_default();
            let old_count = token_id_pair_info.count;
            let new_count_i64 = i64::from(old_count) + delta;
            debug_assert!(
                new_count_i64 >= 0,
                "token-id pair counts must stay non-negative"
            );
            let new_count = u32::try_from(new_count_i64)
                .expect("token-id pair counts must fit in u32 after adjustment");

            token_id_pair_info.count = new_count;
            if delta > 0 {
                token_id_pair_info.locations.insert((chain_index, pos));
            } else {
                token_id_pair_info.locations.remove(&(chain_index, pos));
            }
            (old_count, new_count)
        };

        if old_count > 0 {
            let bucket = self
                .count_to_token_id_pairs
                .get_mut(old_count as usize)
                .expect("existing token-id pair count bucket must exist");
            bucket.remove(&token_id_pair);
            if self.max_token_id_pair_count == old_count && bucket.is_empty() {
                while self.max_token_id_pair_count > 0
                    && self.count_to_token_id_pairs[self.max_token_id_pair_count as usize]
                        .is_empty()
                {
                    self.max_token_id_pair_count -= 1;
                }
            }
        }

        if new_count == 0 {
            self.token_id_pair_info.remove(&token_id_pair);
            return;
        }

        if self.count_to_token_id_pairs.len() <= new_count as usize {
            self.count_to_token_id_pairs
                .resize_with(new_count as usize + 1, FxHashSet::default);
        }
        self.count_to_token_id_pairs[new_count as usize].insert(token_id_pair);
        self.max_token_id_pair_count = self.max_token_id_pair_count.max(new_count);
    }

    fn best_token_id_pair(&self) -> Option<TokenIdPair> {
        self.count_to_token_id_pairs
            .get(self.max_token_id_pair_count as usize)
            .and_then(|bucket| bucket.iter().min())
            .copied()
    }

    fn register_merged_token(&mut self, merged_id: TokenId, token_id_pair: TokenIdPair) {
        let left_bytes = &self.vocab[&token_id_pair.0];
        let right_bytes = &self.vocab[&token_id_pair.1];
        let mut new_bytes = Vec::with_capacity(left_bytes.len() + right_bytes.len());
        new_bytes.extend_from_slice(left_bytes);
        new_bytes.extend_from_slice(right_bytes);
        self.vocab.insert(merged_id, new_bytes);
        self.merge_map.insert(token_id_pair, merged_id);
    }

    fn take_token_id_pair_locations(
        &mut self,
        token_id_pair: TokenIdPair,
    ) -> Vec<(ChainIndex, NodePos)> {
        let removed_token_id_pair_info = self.token_id_pair_info.remove(&token_id_pair);
        let token_id_pair_locations = removed_token_id_pair_info
            .as_ref()
            .map(|info| info.locations.iter().copied().collect())
            .unwrap_or_default();

        if let Some(token_id_pair_info) = removed_token_id_pair_info {
            debug_assert_eq!(token_id_pair_info.count, self.max_token_id_pair_count);
            self.count_to_token_id_pairs[self.max_token_id_pair_count as usize]
                .remove(&token_id_pair);
            while self.max_token_id_pair_count > 0
                && self.count_to_token_id_pairs[self.max_token_id_pair_count as usize].is_empty()
            {
                self.max_token_id_pair_count -= 1;
            }
        }

        token_id_pair_locations
    }

    fn apply_token_id_pair_merge_at(
        &mut self,
        chains: &mut [WeightedChain],
        token_id_pair: TokenIdPair,
        merged_id: TokenId,
        chain_index: ChainIndex,
        left_pos: NodePos,
    ) {
        let frequency = i64::from(chains[chain_index].frequency);
        let Some(left_node) = chains[chain_index].chain.nodes[left_pos as usize] else {
            return;
        };
        let Some(right_pos) = left_node.next else {
            return;
        };
        let Some(right_node) = chains[chain_index].chain.nodes[right_pos as usize] else {
            return;
        };
        if (left_node.token_id, right_node.token_id) != token_id_pair {
            return;
        }

        let prev = left_node.prev;
        let next = right_node.next;
        let prev_id = prev.map(|pos| {
            chains[chain_index].chain.nodes[pos as usize]
                .expect("previous node must exist")
                .token_id
        });
        let next_id = next.map(|pos| {
            chains[chain_index].chain.nodes[pos as usize]
                .expect("next node must exist")
                .token_id
        });

        let new_pos = chains[chain_index]
            .chain
            .splice(left_pos, right_pos, merged_id);

        if let (Some(prev_id), Some(prev_pos)) = (prev_id, prev) {
            let left_neighbor = (prev_id, token_id_pair.0);
            if left_neighbor != token_id_pair {
                self.adjust(left_neighbor, chain_index, prev_pos, -frequency);
            }
            self.adjust((prev_id, merged_id), chain_index, prev_pos, frequency);
        }
        if let Some(next_id) = next_id {
            let right_neighbor = (token_id_pair.1, next_id);
            if right_neighbor != token_id_pair {
                self.adjust(right_neighbor, chain_index, right_pos, -frequency);
            }
            self.adjust((merged_id, next_id), chain_index, new_pos, frequency);
        }
    }

    /// Learns merge rules until `vocab_size` is reached or no mergeable token-id pair remains.
    ///
    /// Detailed training loop:
    ///
    /// ```text
    /// [1] reset_training_state
    ///      |
    ///      v
    /// [2] build weighted chains from docs
    ///      (deduplicate identical chunks -> frequency weight)
    ///      |
    ///      v
    /// [3] build initial token-id-pair stats
    ///      token_id_pair_info[(L,R)] = {weighted_count, all left-node locations}
    ///      count_to_token_id_pairs[count] += (L,R)
    ///      |
    ///      v
    /// [4] for merged_id in vocabulary-growth order:
    ///      |
    ///      +--> select best_token_id_pair from count_to_token_id_pairs[max_token_id_pair_count]
    ///      |      (deterministic tie break by lexical token-id-pair order)
    ///      |
    ///      +--> vocab[merged_id] = vocab[left] ++ vocab[right]
    ///      |    merge_map[(left,right)] = merged_id
    ///      |
    ///      +--> for each recorded occurrence of best_token_id_pair:
    ///             verify nodes still match (skip stale locations)
    ///             splice(left,right -> merged_id)
    ///             adjust frequencies for old neighbors removed
    ///             adjust frequencies for new neighbors created
    /// ```
    ///
    /// Because only local neighborhoods are re-counted after each splice, training scales with
    /// actual edits rather than repeatedly scanning every adjacency in every chain.
    pub fn train(&mut self, vocab_size: TokenId, docs: impl IntoIterator<Item = impl AsRef<str>>) {
        self.reset_training_state();
        self.chains = self.build_training_chains(docs);

        // Take ownership so we can mutate chains freely while still updating `self`'s indexes.
        let mut chains = std::mem::take(&mut self.chains);

        // Seed all token-id pair indexes from a parallel count/location aggregation.
        let initial_token_id_pair_stats = Self::build_initial_token_id_pair_stats(&chains);
        if !initial_token_id_pair_stats.is_empty() {
            self.token_id_pair_info
                .reserve(initial_token_id_pair_stats.len());

            self.max_token_id_pair_count = initial_token_id_pair_stats
                .values()
                .map(|(count, _)| *count)
                .max()
                .unwrap_or_default();
            self.count_to_token_id_pairs.resize_with(
                self.max_token_id_pair_count as usize + 1,
                FxHashSet::default,
            );

            for (token_id_pair, (count, locations)) in initial_token_id_pair_stats {
                self.token_id_pair_info.insert(
                    token_id_pair,
                    TokenIdPairInfo {
                        count,
                        locations: locations.into_iter().collect(),
                    },
                );
                self.count_to_token_id_pairs[count as usize].insert(token_id_pair);
            }
        }

        let reserved_ids: HashSet<_> = self.special_tokens.values().copied().collect();
        for merged_id in
            (BASE_VOCAB_SIZE..vocab_size).filter(|token_id| !reserved_ids.contains(token_id))
        {
            let Some(best_token_id_pair) = self.best_token_id_pair() else {
                break;
            };
            self.register_merged_token(merged_id, best_token_id_pair);
            let best_token_id_pair_locations =
                self.take_token_id_pair_locations(best_token_id_pair);

            for (chain_index, left_pos) in best_token_id_pair_locations {
                self.apply_token_id_pair_merge_at(
                    &mut chains,
                    best_token_id_pair,
                    merged_id,
                    chain_index,
                    left_pos,
                );
            }
        }

        self.chains = chains;
    }

    /// Encodes text by repeatedly applying the learned merge with the lowest token id available.
    ///
    /// Priority resolution mirrors training chronology because merge ids are assigned in the order
    /// rules are learned (lower id => earlier, higher-priority merge).
    ///
    /// Per-chain encode flow:
    ///
    /// ```text
    /// chain nodes
    ///   |
    ///   v
    /// scan adjacent token-id pairs -> candidate merges from merge_map
    ///   |
    ///   v
    /// choose candidate with smallest merge_id
    ///   |
    ///   +--> none: emit chain token_ids
    ///   |
    ///   +--> some: splice token-id pair and rescan chain
    /// ```
    pub fn encode(&self, doc: impl AsRef<str>) -> Vec<TokenId> {
        let mut chains = self.split(doc);
        let mut encoded = Vec::new();

        for chain in &mut chains {
            loop {
                let mut best: Option<(TokenId, NodePos, NodePos)> = None;
                let mut previous: Option<(NodePos, TokenId)> = None;

                for (pos, node) in chain.iter() {
                    if let Some((prev_pos, prev_id)) = previous {
                        let token_id_pair = (prev_id, node.token_id);
                        if let Some(&merge_id) = self.merge_map.get(&token_id_pair)
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
        self.split_impl(doc, true)
    }

    fn split_impl(&self, doc: impl AsRef<str>, include_special_tokens: bool) -> Vec<Chain> {
        let doc = doc.as_ref();
        let mut chains = Vec::new();

        self.for_each_segment_between_specials(doc, |segment| match segment {
            Segment::Regular(segment) => {
                chains.extend(
                    self.split_matches(segment)
                        .map(|matched| Chain::new(matched.as_bytes())),
                );
            }
            Segment::Special(special) => {
                if include_special_tokens && let Some(&token_id) = self.special_tokens.get(special)
                {
                    chains.push(Chain::from_token_id(token_id));
                }
            }
        });
        chains
    }

    /// Clears all learned merges while preserving the immutable base vocabulary.
    fn reset_training_state(&mut self) {
        let special_token_ids: HashSet<_> = self.special_tokens.values().copied().collect();
        self.vocab.retain(|token_id, _| {
            *token_id < BASE_VOCAB_SIZE || special_token_ids.contains(token_id)
        });
        self.merge_map.clear();
        self.chains.clear();
        self.count_to_token_id_pairs.clear();
        self.count_to_token_id_pairs.push(FxHashSet::default());
        self.max_token_id_pair_count = 0;
        self.token_id_pair_info.clear();
    }
}

impl Default for BPE {
    fn default() -> Self {
        Self::new(None, None::<Vec<(String, TokenId)>>)
            .expect("default configuration should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_learns_most_frequent_pair_and_roundtrips() {
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(257, ["abababa"]);

        assert_eq!(bpe.vocab.get(&256), Some(&b"ab".to_vec()));
        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&256));

        let encoded = bpe.encode("abababa");
        assert_eq!(encoded, vec![256, 256, 256, b'a' as u32]);
        assert_eq!(bpe.decode(encoded), b"abababa");
    }

    #[test]
    fn train_aggregates_identical_chunks_and_weights_pair_counts() {
        let mut bpe = BPE::new(Some("\\S+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(258, ["the the", "the"]);

        assert_eq!(bpe.chains.len(), 1);
        assert_eq!(bpe.chains[0].frequency, 3);
        assert_eq!(bpe.vocab.get(&256), Some(&b"he".to_vec()));
        assert_eq!(bpe.vocab.get(&257), Some(&b"the".to_vec()));
        assert!(
            !bpe.token_id_pair_info
                .contains_key(&(b't' as u32, b'h' as u32))
        );
        assert_eq!(bpe.encode("the the"), vec![257, 257]);
    }

    #[test]
    fn split_for_training_bytes_uses_special_tokens_as_boundaries() {
        let bpe = BPE::new(Some("(?s).+"), Some([("<|eot|>", BASE_VOCAB_SIZE)]))
            .expect("valid config should construct");
        let chunks = bpe.split_for_training_bytes("left<|eot|>right");
        assert_eq!(
            chunks.iter().map(SmallVec::as_slice).collect::<Vec<_>>(),
            vec![b"left".as_slice(), b"right".as_slice()]
        );
    }

    #[test]
    fn training_handles_overlapping_pairs_without_corrupting_state() {
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(258, ["aaaa"]);

        assert_eq!(bpe.vocab.get(&256), Some(&b"aa".to_vec()));
        assert_eq!(bpe.vocab.get(&257), Some(&b"aaaa".to_vec()));
        assert_eq!(bpe.encode("aaaa"), vec![257]);
        assert_eq!(bpe.decode(vec![257]), b"aaaa");
    }

    #[test]
    fn split_pattern_keeps_merges_scoped_to_each_match() {
        let mut bpe = BPE::new(Some("\\S+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
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
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(300, std::iter::empty::<&str>());

        assert!(bpe.encode("").is_empty());
        assert!(bpe.decode(Vec::new()).is_empty());
        assert_eq!(bpe.decode(vec![999_999]), Vec::<u8>::new());
        assert!(bpe.merge_map.is_empty());
        assert!(bpe.chains.is_empty());
        assert_eq!(bpe.count_to_token_id_pairs.len(), 1);
        assert!(bpe.count_to_token_id_pairs[0].is_empty());
        assert!(bpe.token_id_pair_info.is_empty());
    }

    #[test]
    fn requesting_base_vocab_size_keeps_byte_level_encoding() {
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
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
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
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
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(259, ["abba"]);

        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&256));
        assert_eq!(bpe.merge_map.get(&(b'b' as u32, b'a' as u32)), Some(&257));
        assert_eq!(bpe.merge_map.get(&(256, 257)), Some(&258));
        assert_eq!(bpe.encode("abba"), vec![258]);
    }

    #[test]
    fn try_new_with_special_tokens_rejects_ids_in_base_vocab() {
        let err = BPE::new(Some("(?s).+"), Some([("<pad>", 42)]))
            .expect_err("special token ids below the byte vocabulary should be rejected");

        assert!(matches!(
            err,
            BPEError::SpecialTokenIdOverlapsBaseVocab { token_id: 42, .. }
        ));
        assert!(err.to_string().contains("must be >= 256"));
    }

    #[test]
    fn try_new_with_special_tokens_rejects_duplicate_ids() {
        let err = BPE::new(Some("(?s).+"), Some([("<pad>", 512), ("<eos>", 512)]))
            .expect_err("duplicate special token ids should be rejected");

        assert!(matches!(
            err,
            BPEError::DuplicateSpecialTokenId { token_id: 512 }
        ));
        assert!(err.to_string().contains("assigned more than once"));
    }

    #[test]
    fn try_new_returns_error_for_invalid_regex() {
        assert!(BPE::new(Some("("), None::<Vec<(String, TokenId)>>).is_err());
    }

    #[test]
    fn training_split_uses_special_tokens_as_boundaries_without_emitting_them() {
        let bpe = BPE::new(Some(r"\S+"), Some([("<pad>", 300), ("<eos>", 301)]))
            .expect("valid config should construct");
        assert_eq!(
            bpe.split_for_training_bytes("hi<pad><eos>there")
                .iter()
                .map(SmallVec::as_slice)
                .collect::<Vec<_>>(),
            vec![b"hi".as_slice(), b"there".as_slice()]
        );
    }

    #[test]
    fn special_tokens_keep_custom_ids_and_roundtrip_without_splitting() {
        let bpe = BPE::new(Some("\\S+"), Some([("<pad>", 1024), ("<eos>", 2048)]))
            .expect("valid config should construct");

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
        let mut bpe = BPE::new(Some("(?s).+"), Some([("<pad>", 300), ("<eos>", 301)]))
            .expect("valid config should construct");
        bpe.train(305, ["a<pad>a<pad>", "<pad><eos><pad>"]);

        assert!(!bpe.merge_map.keys().any(|token_id_pair| {
            token_id_pair.0 == 300
                || token_id_pair.1 == 300
                || token_id_pair.0 == 301
                || token_id_pair.1 == 301
        }));
        assert_eq!(
            bpe.encode("a<pad><eos>a"),
            vec![b'a' as u32, 300, 301, b'a' as u32]
        );
    }

    #[test]
    fn split_includes_special_tokens_as_single_node_chains() {
        let bpe = BPE::new(Some(r"\S+"), Some([("<pad>", 300), ("<eos>", 301)]))
            .expect("valid config should construct");

        let chains = bpe.split("hi<pad>there<eos>");
        let tokens_per_chain: Vec<Vec<TokenId>> = chains
            .iter()
            .map(|chain| chain.iter().map(|(_, node)| node.token_id).collect())
            .collect();

        assert_eq!(
            tokens_per_chain,
            vec![
                vec![b'h' as u32, b'i' as u32],
                vec![300],
                vec![
                    b't' as u32,
                    b'h' as u32,
                    b'e' as u32,
                    b'r' as u32,
                    b'e' as u32
                ],
                vec![301]
            ]
        );
    }

    #[test]
    fn build_special_token_pattern_prefers_longer_tokens_before_prefixes() {
        let bpe = BPE::new(Some(r"\S+"), Some([("<e>", 400), ("<eos>", 401)]))
            .expect("valid config should construct");

        assert_eq!(bpe.encode("<eos><e>"), vec![401, 400]);
    }

    #[test]
    fn split_preserves_document_order_across_regular_and_special_segments() {
        let bpe = BPE::new(Some(r"\S+"), Some([("<pad>", 300), ("<eos>", 301)]))
            .expect("valid config should construct");

        let chains = bpe.split("x<eos>y<pad>z");
        let tokens_per_chain: Vec<Vec<TokenId>> = chains
            .iter()
            .map(|chain| chain.iter().map(|(_, node)| node.token_id).collect())
            .collect();

        assert_eq!(
            tokens_per_chain,
            vec![
                vec![b'x' as u32],
                vec![301],
                vec![b'y' as u32],
                vec![300],
                vec![b'z' as u32]
            ]
        );
    }

    #[test]
    fn train_skips_reserved_special_token_ids_when_assigning_merges() {
        let mut bpe = BPE::new(Some("(?s).+"), Some([("<pad>", 256), ("<eos>", 257)]))
            .expect("valid config should construct");
        bpe.train(260, ["abab"]);

        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&258));
        assert_eq!(bpe.vocab.get(&256), Some(&b"<pad>".to_vec()));
        assert_eq!(bpe.vocab.get(&257), Some(&b"<eos>".to_vec()));
        assert_eq!(bpe.encode("<pad>ab<eos>"), vec![256, 258, 257]);
    }

    #[test]
    fn train_stops_when_no_mergeable_pairs_exist() {
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(300, ["a", "b", "c"]);

        assert!(bpe.merge_map.is_empty());
        assert_eq!(bpe.vocab.len() as u32, BASE_VOCAB_SIZE);
        assert_eq!(
            bpe.encode("abc"),
            vec![b'a' as u32, b'b' as u32, b'c' as u32]
        );
    }

    #[test]
    fn encode_returns_only_special_tokens_when_input_is_only_specials() {
        let bpe = BPE::new(Some(r"\S+"), Some([("<pad>", 600), ("<eos>", 601)]))
            .expect("valid config should construct");

        assert_eq!(bpe.encode("<pad><eos><pad>"), vec![600, 601, 600]);
    }

    #[test]
    fn train_deduplicates_chunks_across_split_batches() {
        let mut bpe = BPE::new(Some("(?s).+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        let doc_count = 1_017;
        let docs = std::iter::repeat_n("xy", doc_count);

        bpe.train(257, docs);

        assert_eq!(bpe.chains.len(), 1);
        assert_eq!(bpe.chains[0].frequency as usize, doc_count);
        assert_eq!(bpe.vocab.get(&256), Some(&b"xy".to_vec()));
    }

    #[test]
    fn reset_training_state_preserves_special_tokens_between_retrains() {
        let mut bpe = BPE::new(Some("(?s).+"), Some([("<pad>", 700)]))
            .expect("valid config should construct");
        bpe.train(258, ["abab"]);
        assert!(bpe.vocab.contains_key(&700));

        bpe.train(258, ["cdcd"]);

        assert_eq!(bpe.vocab.get(&700), Some(&b"<pad>".to_vec()));
        assert_eq!(bpe.merge_map.get(&(b'c' as u32, b'd' as u32)), Some(&256));
        assert_eq!(bpe.encode("<pad>cd"), vec![700, 256]);
    }

    #[test]
    fn train_uses_stable_pair_order_when_frequencies_tie() {
        let mut bpe = BPE::new(Some("\\S+"), None::<Vec<(String, TokenId)>>)
            .expect("valid regex should construct");
        bpe.train(258, ["ab ac"]);

        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&256));
        assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'c' as u32)), Some(&257));
        assert_eq!(bpe.encode("ab ac"), vec![256, 257]);
    }

    #[test]
    fn encode_of_empty_string_is_empty_even_with_special_tokens_configured() {
        let bpe =
            BPE::new(Some(r"\S+"), Some([("<pad>", 900)])).expect("valid config should construct");

        assert!(bpe.encode("").is_empty());
        assert!(bpe.split("").is_empty());
    }
}
