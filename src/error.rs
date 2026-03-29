use crate::types::{BASE_VOCAB_SIZE, TokenId};
use std::fmt;

/// Errors that can occur while constructing a [`crate::bpe::BPE`] instance.
#[derive(Debug)]
pub enum BPEError {
    /// The caller supplied an invalid regex for splitting documents into chunks.
    InvalidSplitRegex(regex::Error),
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

impl From<regex::Error> for BPEError {
    fn from(value: regex::Error) -> Self {
        Self::InvalidSplitRegex(value)
    }
}
