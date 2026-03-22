/// Size of the immutable byte-level vocabulary (`0..=255`).
pub(crate) const BASE_VOCAB_SIZE: u32 = 256;

/// Sentinel value representing "no link" in [`crate::chain::Node`] prev/next fields,
/// and a tombstone marker for removed nodes.
pub(crate) const NONE: u32 = u32::MAX;

/// Public/external token identifier type used throughout the crate.
pub(crate) type TokenId = u32;
/// Index into a [`crate::chain::Chain`] node buffer.
pub(crate) type NodePos = u32;
/// Adjacent token pair used as the key for merge rules.
pub(crate) type Pair = (TokenId, TokenId);
/// Index into [`crate::bpe::BPE`] training chains.
pub(crate) type ChainIndex = u32;
/// All observed locations for a pair.
pub(crate) type PairLocations = Vec<(ChainIndex, NodePos)>;
