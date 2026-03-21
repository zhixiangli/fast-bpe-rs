pub(crate) const BASE_VOCAB_SIZE: u32 = 256;

pub(crate) type TokenId = u32;
pub(crate) type NodePos = usize;
pub(crate) type Pair = (TokenId, TokenId);
pub(crate) type ChainIndex = usize;
pub(crate) type PairLocations = std::collections::BTreeSet<(ChainIndex, NodePos)>;
