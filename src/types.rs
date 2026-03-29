/// Size of the immutable byte-level vocabulary (`0..=255`).
pub(crate) const BASE_VOCAB_SIZE: u32 = 256;

/// Public/external token identifier type used throughout the crate.
pub(crate) type TokenId = u32;
/// Index into a [`crate::chain::Chain`] node buffer.
pub(crate) type NodePos = u32;
/// Adjacent token pair used as the key for merge rules.
pub(crate) type Pair = (TokenId, TokenId);
/// Index into [`crate::bpe::BPE`] training chains.
pub(crate) type ChainIndex = usize;
/// All observed locations for a pair, ordered deterministically for stable iteration.
pub(crate) type PairLocations = std::collections::BTreeSet<(ChainIndex, NodePos)>;
/// Fast hash map used for pair-keyed training indexes.
pub(crate) type SeedMap<V> = rustc_hash::FxHashMap<Pair, V>;
