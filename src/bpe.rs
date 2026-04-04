use crate::error::BPEError;
use crate::merge_sequence::MergeSequence;
use crate::types::{
    BASE_VOCAB_SIZE, MergeNodeSlot, MergeSequenceIndex, SeedMap, TokenId, TokenIdPair,
    TokenIdPairOccurrences, TrainingChunk,
};
use ahash::RandomState as AHashRandomState;
use fancy_regex::{Regex, escape};
use hashbrown::{HashMap as HbHashMap, hash_map::EntryRef};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};

/// One unique training chunk plus the number of corpus occurrences it represents.
#[derive(Debug)]
struct WeightedMergeSequence {
    merge_sequence: MergeSequence,
    frequency: u32,
}

#[derive(Debug, Default)]
struct TokenIdPairInfo {
    count: u32,
    occurrences: TokenIdPairOccurrences,
}

enum Segment<'a> {
    Regular(&'a str),
    Special(&'a str),
}

/// Byte Pair Encoding model with optional special-token support.
///
/// The core design treats each tokenized chunk as a mutable linked structure (`MergeSequence`) and tracks
/// adjacent token-id-pair statistics in a frequency-indexed table. This avoids repeated full rescans of the
/// corpus during training: after each merge, only neighborhoods around changed nodes are updated.
///
/// Training data flow (coarse):
///
/// ```text
/// docs
///   |
///   v
/// split into regular segments + special-token boundaries
///   regular segments -> regex chunk matches (bytes)
///   |
///   v
/// build_training_merge_sequences (deduplicate chunks -> weighted merge_sequences)
///   |
///   v
/// build_initial_token_id_pair_stats (token-id pair -> {count, locations})
///   |
///   v
/// repeat until vocab full:
///   pick highest-frequency token-id pair (stable tie-break: lexical pair order)
///   materialize merged token bytes
///   splice every recorded occurrence in-place
///   adjust only affected neighboring token-id pairs
/// ```
#[derive(Debug)]
pub struct BPE {
    /// Source pattern used to build worker-local split regexes.
    split_pattern_source: String,
    /// Source pattern used to build worker-local special-token regexes.
    special_split_pattern_source: Option<String>,
    /// Mapping from literal special-token text to the externally-visible token id.
    special_tokens: HashMap<String, TokenId>,
    /// Vocabulary entries materialized as raw bytes for decode/inspection.
    pub(crate) vocab: HashMap<TokenId, Vec<u8>>, // id -> bytes
    /// Learned merge rules keyed by `(left, right)` token pairs.
    pub(crate) merge_map: HashMap<TokenIdPair, TokenId>, // token-id pair -> merged id

    // Training-time index structures.
    //
    // `merge_sequences` stores each unique training chunk once with its corpus multiplicity.
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
    merge_sequences: Vec<WeightedMergeSequence>,
    count_to_token_id_pairs: FxHashMap<u32, FxHashSet<TokenIdPair>>, // frequency -> token-id pairs
    max_token_id_pair_count: u32, // largest non-empty frequency bucket
    token_id_pair_info: FxHashMap<TokenIdPair, TokenIdPairInfo>, // token-id pair -> {frequency, (merge_sequence_idx, node_pos)}
}

impl BPE {
    pub const DEFAULT_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

    fn split_matches<'a>(
        split_pattern: &'a Regex,
        segment: &'a str,
    ) -> impl Iterator<Item = &'a str> + 'a {
        split_pattern
            .find_iter(segment)
            .map(|matched| matched.expect("split regex evaluation should succeed"))
            .map(|matched| matched.as_str())
    }

    fn for_each_segment_between_specials(
        doc: &str,
        special_split_pattern: Option<&Regex>,
        mut on_segment: impl FnMut(Segment<'_>),
    ) {
        let mut cursor = 0;
        if let Some(special_split_pattern) = special_split_pattern {
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

    /// Parallel fold+reduce over all docs, producing byte-chunk frequencies.
    ///
    /// Think of this stage as a distributed "word histogram", but at regex-chunk byte granularity:
    ///
    /// ```text
    /// worker 1: {"the": 2, "cat": 1}
    /// worker 2: {"the": 1, "sat": 3}
    /// --------------------------------
    /// reduce  : {"the": 3, "cat": 1, "sat": 3}
    /// ```
    ///
    /// Later, each unique chunk becomes exactly one merge_sequence with a `frequency` weight, so repeated
    /// text does not allocate duplicate merge_sequence structures.
    fn build_training_merge_sequences(
        &self,
        docs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Vec<WeightedMergeSequence> {
        #[inline]
        fn count_chunk(
            counts: &mut HbHashMap<TrainingChunk, u32, AHashRandomState>,
            chunk_bytes: &[u8],
        ) {
            match counts.entry_ref(chunk_bytes) {
                EntryRef::Occupied(mut entry) => {
                    *entry.get_mut() += 1;
                }
                EntryRef::Vacant(entry) => {
                    entry.insert(1);
                }
            }
        }

        let docs: Vec<String> = docs
            .into_iter()
            .map(|doc| doc.as_ref().to_owned())
            .collect();
        docs.par_iter()
            .fold(
                || {
                    (
                        HbHashMap::<TrainingChunk, u32, AHashRandomState>::default(),
                        Regex::new(&self.split_pattern_source)
                            .expect("split regex source should remain valid"),
                        self.special_split_pattern_source.as_ref().map(|pattern| {
                            Regex::new(pattern)
                                .expect("special token regex source should remain valid")
                        }),
                    )
                },
                |(mut local_counts, split_pattern, special_split_pattern), doc| {
                    let mut cursor = 0usize;
                    if let Some(special_split_pattern) = special_split_pattern.as_ref() {
                        for matched in special_split_pattern.find_iter(doc).map(|matched| {
                            matched.expect("special token regex evaluation should succeed")
                        }) {
                            for chunk_match in split_pattern
                                .find_iter(&doc[cursor..matched.start()])
                                .map(|chunk_match| {
                                    chunk_match.expect("split regex evaluation should succeed")
                                })
                            {
                                count_chunk(&mut local_counts, chunk_match.as_str().as_bytes());
                            }
                            cursor = matched.end();
                        }
                    }
                    for chunk_match in split_pattern.find_iter(&doc[cursor..]).map(|chunk_match| {
                        chunk_match.expect("split regex evaluation should succeed")
                    }) {
                        count_chunk(&mut local_counts, chunk_match.as_str().as_bytes());
                    }
                    (local_counts, split_pattern, special_split_pattern)
                },
            )
            .reduce(
                || {
                    (
                        HbHashMap::<TrainingChunk, u32, AHashRandomState>::default(),
                        Regex::new(&self.split_pattern_source)
                            .expect("split regex source should remain valid"),
                        self.special_split_pattern_source.as_ref().map(|pattern| {
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
            .map(|(chunk, frequency)| WeightedMergeSequence {
                merge_sequence: MergeSequence::new(&chunk),
                frequency,
            })
            .collect()
    }

    /// Counts every adjacent token-id pair in parallel and records all left-node locations.
    ///
    /// The resulting map is used to seed `token_id_pair_info` and `count_to_token_id_pairs` in one
    /// pass with one bucket insertion per unique token-id pair.
    fn build_initial_token_id_pair_stats(
        merge_sequences: &[WeightedMergeSequence],
    ) -> SeedMap<(u32, Vec<(MergeSequenceIndex, MergeNodeSlot)>)> {
        merge_sequences
            .par_iter()
            .enumerate()
            .fold(
                SeedMap::<(u32, Vec<(MergeSequenceIndex, MergeNodeSlot)>)>::default,
                |mut local_token_id_pairs, (merge_sequence_index, weighted_merge_sequence)| {
                    let token_id_pair_capacity = weighted_merge_sequence
                        .merge_sequence
                        .slots
                        .len()
                        .saturating_sub(1);
                    if token_id_pair_capacity > 0 {
                        local_token_id_pairs.reserve(token_id_pair_capacity);
                    }

                    let frequency = weighted_merge_sequence.frequency;
                    let mut previous = None;
                    for (pos, node) in weighted_merge_sequence.merge_sequence.iter() {
                        if let Some((left_slot, left_id)) = previous {
                            let token_id_pair = (left_id, node.token_id);
                            let (count, occurrences) = local_token_id_pairs
                                .entry(token_id_pair)
                                .or_insert_with(|| (0, Vec::with_capacity(1)));
                            *count += frequency;
                            occurrences.push((merge_sequence_index, left_slot));
                        }
                        previous = Some((pos, node.token_id));
                    }
                    local_token_id_pairs
                },
            )
            .reduce(
                SeedMap::<(u32, Vec<(MergeSequenceIndex, MergeNodeSlot)>)>::default,
                |mut global_token_id_pairs, local_token_id_pairs| {
                    global_token_id_pairs.reserve(local_token_id_pairs.len());
                    for (token_id_pair, (count, mut occurrences)) in local_token_id_pairs {
                        let (global_count, global_occurrences) = global_token_id_pairs
                            .entry(token_id_pair)
                            .or_insert_with(|| (0, Vec::with_capacity(occurrences.len())));
                        *global_count += count;
                        global_occurrences.append(&mut occurrences);
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
        Regex::new(&split_pattern_source).map_err(BPEError::from)?;
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
        special_split_pattern_source
            .as_deref()
            .map(Regex::new)
            .transpose()
            .map_err(BPEError::from)?;

        Ok(Self {
            split_pattern_source,
            special_split_pattern_source,
            special_tokens: special_token_map,
            vocab,
            merge_map: HashMap::new(),
            merge_sequences: Vec::new(),
            count_to_token_id_pairs: FxHashMap::default(),
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
    /// `delta` is weighted by the source chunk frequency, so one splice in a merge_sequence that
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
    ///   locations +/- (merge_sequence_index, pos)
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
        merge_sequence_index: MergeSequenceIndex,
        pos: MergeNodeSlot,
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
                token_id_pair_info
                    .occurrences
                    .insert((merge_sequence_index, pos));
            } else {
                token_id_pair_info
                    .occurrences
                    .remove(&(merge_sequence_index, pos));
            }
            (old_count, new_count)
        };

        if old_count > 0 {
            let bucket = self
                .count_to_token_id_pairs
                .get_mut(&old_count)
                .expect("existing token-id pair count bucket must exist");
            bucket.remove(&token_id_pair);
            if self.max_token_id_pair_count == old_count && bucket.is_empty() {
                while self.max_token_id_pair_count > 0
                    && self
                        .count_to_token_id_pairs
                        .get(&self.max_token_id_pair_count)
                        .is_none_or(FxHashSet::is_empty)
                {
                    self.max_token_id_pair_count -= 1;
                }
            }
        }

        if new_count == 0 {
            self.token_id_pair_info.remove(&token_id_pair);
            return;
        }

        self.count_to_token_id_pairs
            .entry(new_count)
            .or_default()
            .insert(token_id_pair);
        self.max_token_id_pair_count = self.max_token_id_pair_count.max(new_count);
    }

    /// Returns the current best merge candidate:
    /// - highest observed frequency bucket
    /// - deterministic tie-break by `(left_id, right_id)` order.
    fn best_token_id_pair(&self) -> Option<TokenIdPair> {
        self.count_to_token_id_pairs
            .get(&self.max_token_id_pair_count)
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

    fn take_token_id_pair_occurrences(
        &mut self,
        token_id_pair: TokenIdPair,
    ) -> Vec<(MergeSequenceIndex, MergeNodeSlot)> {
        let removed_token_id_pair_info = self.token_id_pair_info.remove(&token_id_pair);
        let token_id_pair_occurrences = removed_token_id_pair_info
            .as_ref()
            .map(|info| info.occurrences.iter().copied().collect())
            .unwrap_or_default();

        if let Some(token_id_pair_info) = removed_token_id_pair_info {
            debug_assert_eq!(token_id_pair_info.count, self.max_token_id_pair_count);
            self.count_to_token_id_pairs
                .entry(self.max_token_id_pair_count)
                .or_default()
                .remove(&token_id_pair);
            while self.max_token_id_pair_count > 0
                && self
                    .count_to_token_id_pairs
                    .get(&self.max_token_id_pair_count)
                    .is_none_or(FxHashSet::is_empty)
            {
                self.max_token_id_pair_count -= 1;
            }
        }

        token_id_pair_occurrences
    }

    fn apply_token_id_pair_merge_at(
        &mut self,
        merge_sequences: &mut [WeightedMergeSequence],
        token_id_pair: TokenIdPair,
        merged_id: TokenId,
        merge_sequence_index: MergeSequenceIndex,
        left_slot: MergeNodeSlot,
    ) {
        let frequency = i64::from(merge_sequences[merge_sequence_index].frequency);
        let Some(left_node) =
            merge_sequences[merge_sequence_index].merge_sequence.slots[left_slot as usize]
        else {
            return;
        };
        let Some(right_slot) = left_node.next else {
            return;
        };
        let Some(right_node) =
            merge_sequences[merge_sequence_index].merge_sequence.slots[right_slot as usize]
        else {
            return;
        };
        if (left_node.token_id, right_node.token_id) != token_id_pair {
            return;
        }

        let prev = left_node.prev;
        let next = right_node.next;
        let prev_id = prev.map(|pos| {
            merge_sequences[merge_sequence_index].merge_sequence.slots[pos as usize]
                .expect("previous node must exist")
                .token_id
        });
        let next_id = next.map(|pos| {
            merge_sequences[merge_sequence_index].merge_sequence.slots[pos as usize]
                .expect("next node must exist")
                .token_id
        });

        let new_slot = merge_sequences[merge_sequence_index]
            .merge_sequence
            .splice(left_slot, right_slot, merged_id);

        if let (Some(prev_id), Some(prev_slot)) = (prev_id, prev) {
            let left_neighbor = (prev_id, token_id_pair.0);
            if left_neighbor != token_id_pair {
                self.adjust(left_neighbor, merge_sequence_index, prev_slot, -frequency);
            }
            self.adjust(
                (prev_id, merged_id),
                merge_sequence_index,
                prev_slot,
                frequency,
            );
        }
        if let Some(next_id) = next_id {
            let right_neighbor = (token_id_pair.1, next_id);
            if right_neighbor != token_id_pair {
                self.adjust(right_neighbor, merge_sequence_index, right_slot, -frequency);
            }
            self.adjust(
                (merged_id, next_id),
                merge_sequence_index,
                new_slot,
                frequency,
            );
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
    /// [2] build weighted merge_sequences from docs
    ///      (deduplicate identical chunks -> frequency weight)
    ///      |
    ///      v
    /// [3] build initial token-id-pair stats
    ///      token_id_pair_info[(L,R)] = {weighted_count, all left-node locations}
    ///      count_to_token_id_pairs[count] += (L,R)
    ///      |
    ///      v
    /// [4] for merged_id in vocabulary-growth order (skipping reserved special-token ids):
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
    /// Local splice/update picture for one occurrence:
    ///
    /// ```text
    /// before:   P <-> L <-> R <-> N
    /// pair:            (L,R)              chosen best pair
    ///
    /// after :   P <->  M  <-> N
    ///                merged_id
    ///
    /// counters:
    ///   remove (P,L) and (R,N)   [if those neighbors exist]
    ///   add    (P,M) and (M,N)
    /// ```
    ///
    /// Because only local neighborhoods are re-counted after each splice, training scales with
    /// actual edits rather than repeatedly scanning every adjacency in every merge_sequence.
    pub fn train(&mut self, vocab_size: TokenId, docs: impl IntoIterator<Item = impl AsRef<str>>) {
        self.reset_training_state();
        self.merge_sequences = self.build_training_merge_sequences(docs);

        // Take ownership so we can mutate merge_sequences freely while still updating `self`'s indexes.
        let mut merge_sequences = std::mem::take(&mut self.merge_sequences);

        // Seed all token-id pair indexes from a parallel count/location aggregation.
        let initial_token_id_pair_stats = Self::build_initial_token_id_pair_stats(&merge_sequences);
        if !initial_token_id_pair_stats.is_empty() {
            self.token_id_pair_info
                .reserve(initial_token_id_pair_stats.len());

            self.max_token_id_pair_count = initial_token_id_pair_stats
                .values()
                .map(|(count, _)| *count)
                .max()
                .unwrap_or_default();
            for (token_id_pair, (count, occurrences)) in initial_token_id_pair_stats {
                self.token_id_pair_info.insert(
                    token_id_pair,
                    TokenIdPairInfo {
                        count,
                        occurrences: occurrences.into_iter().collect(),
                    },
                );
                self.count_to_token_id_pairs
                    .entry(count)
                    .or_default()
                    .insert(token_id_pair);
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
            let best_token_id_pair_occurrences =
                self.take_token_id_pair_occurrences(best_token_id_pair);

            for (merge_sequence_index, left_slot) in best_token_id_pair_occurrences {
                self.apply_token_id_pair_merge_at(
                    &mut merge_sequences,
                    best_token_id_pair,
                    merged_id,
                    merge_sequence_index,
                    left_slot,
                );
            }
        }

        self.merge_sequences = merge_sequences;
    }

    /// Encodes text by repeatedly applying the currently available merge with the lowest token id.
    ///
    /// Priority resolution mirrors training chronology because merge ids are assigned in learning
    /// order (lower id => earlier, higher-priority merge).
    ///
    /// Per-merge_sequence encode flow:
    ///
    /// ```text
    /// merge_sequence nodes
    ///   |
    ///   v
    /// scan adjacent token-id pairs -> candidate merges from merge_map
    ///   |
    ///   v
    /// choose candidate with smallest merge_id across the whole merge_sequence
    ///   |
    ///   +--> none: emit merge_sequence token_ids
    ///   |
    ///   +--> some: splice token-id pair and rescan merge_sequence
    /// ```
    pub fn encode(&self, doc: impl AsRef<str>) -> Vec<TokenId> {
        let mut merge_sequences = self.split(doc);
        let mut encoded = Vec::new();

        for merge_sequence in &mut merge_sequences {
            loop {
                let mut best: Option<(TokenId, MergeNodeSlot, MergeNodeSlot)> = None;
                let mut previous: Option<(MergeNodeSlot, TokenId)> = None;

                for (pos, node) in merge_sequence.iter() {
                    if let Some((prev_slot, prev_id)) = previous {
                        let token_id_pair = (prev_id, node.token_id);
                        if let Some(&merge_id) = self.merge_map.get(&token_id_pair)
                            && best.is_none_or(|(best_id, _, _)| merge_id < best_id)
                        {
                            best = Some((merge_id, prev_slot, pos));
                        }
                    }
                    previous = Some((pos, node.token_id));
                }

                let Some((merge_id, left_slot, right_slot)) = best else {
                    break;
                };
                merge_sequence.splice(left_slot, right_slot, merge_id);
            }

            encoded.extend(merge_sequence.iter().map(|(_, node)| node.token_id));
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

    /// Splits input text into independently-encodable merge_sequences.
    ///
    /// Special tokens act like pre-tokenized islands. Everything between them is split by the
    /// normal regex, but the special-token literals themselves become single-node merge_sequences so they
    /// never merge with surrounding text.
    fn split(&self, doc: impl AsRef<str>) -> Vec<MergeSequence> {
        self.split_impl(doc, true)
    }

    fn split_impl(&self, doc: impl AsRef<str>, include_special_tokens: bool) -> Vec<MergeSequence> {
        let doc = doc.as_ref();
        let mut merge_sequences = Vec::new();
        let split_pattern =
            Regex::new(&self.split_pattern_source).expect("split regex source should remain valid");
        let special_split_pattern = self.special_split_pattern_source.as_ref().map(|pattern| {
            Regex::new(pattern).expect("special token regex source should remain valid")
        });

        Self::for_each_segment_between_specials(doc, special_split_pattern.as_ref(), |segment| {
            match segment {
                Segment::Regular(segment) => {
                    merge_sequences.extend(
                        Self::split_matches(&split_pattern, segment)
                            .map(|matched| MergeSequence::new(matched.as_bytes())),
                    );
                }
                Segment::Special(special) => {
                    if include_special_tokens
                        && let Some(&token_id) = self.special_tokens.get(special)
                    {
                        merge_sequences.push(MergeSequence::from_token_id(token_id));
                    }
                }
            }
        });
        merge_sequences
    }

    /// Clears all learned merges while preserving the immutable base vocabulary.
    fn reset_training_state(&mut self) {
        let special_token_ids: HashSet<_> = self.special_tokens.values().copied().collect();
        self.vocab.retain(|token_id, _| {
            *token_id < BASE_VOCAB_SIZE || special_token_ids.contains(token_id)
        });
        self.merge_map.clear();
        self.merge_sequences.clear();
        self.count_to_token_id_pairs.clear();
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

    fn training_chunks(bpe: &BPE, doc: &str) -> Vec<TrainingChunk> {
        let split_pattern =
            Regex::new(&bpe.split_pattern_source).expect("split regex source should remain valid");
        let special_split_pattern = bpe.special_split_pattern_source.as_ref().map(|pattern| {
            Regex::new(pattern).expect("special token regex source should remain valid")
        });
        let mut chunks = Vec::new();
        BPE::for_each_segment_between_specials(doc, special_split_pattern.as_ref(), |segment| {
            if let Segment::Regular(segment) = segment {
                chunks.extend(
                    BPE::split_matches(&split_pattern, segment)
                        .map(|matched| TrainingChunk::from_slice(matched.as_bytes())),
                );
            }
        });
        chunks
    }

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

        assert_eq!(bpe.merge_sequences.len(), 1);
        assert_eq!(bpe.merge_sequences[0].frequency, 3);
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
        let chunks = training_chunks(&bpe, "left<|eot|>right");
        assert_eq!(
            chunks
                .iter()
                .map(TrainingChunk::as_slice)
                .collect::<Vec<_>>(),
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
        assert!(bpe.merge_sequences.is_empty());
        assert!(bpe.count_to_token_id_pairs.is_empty());
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
            training_chunks(&bpe, "hi<pad><eos>there")
                .iter()
                .map(TrainingChunk::as_slice)
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
    fn split_includes_special_tokens_as_single_node_merge_sequences() {
        let bpe = BPE::new(Some(r"\S+"), Some([("<pad>", 300), ("<eos>", 301)]))
            .expect("valid config should construct");

        let merge_sequences = bpe.split("hi<pad>there<eos>");
        let tokens_per_merge_sequence: Vec<Vec<TokenId>> = merge_sequences
            .iter()
            .map(|merge_sequence| {
                merge_sequence
                    .iter()
                    .map(|(_, node)| node.token_id)
                    .collect()
            })
            .collect();

        assert_eq!(
            tokens_per_merge_sequence,
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

        let merge_sequences = bpe.split("x<eos>y<pad>z");
        let tokens_per_merge_sequence: Vec<Vec<TokenId>> = merge_sequences
            .iter()
            .map(|merge_sequence| {
                merge_sequence
                    .iter()
                    .map(|(_, node)| node.token_id)
                    .collect()
            })
            .collect();

        assert_eq!(
            tokens_per_merge_sequence,
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

        assert_eq!(bpe.merge_sequences.len(), 1);
        assert_eq!(bpe.merge_sequences[0].frequency as usize, doc_count);
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
