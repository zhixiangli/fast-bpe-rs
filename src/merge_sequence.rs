use crate::types::{MergeNodeSlot, TokenId};

/// One token inside a [`MergeSequence`].
///
/// The `prev`/`next` links let us remove a merged pair in `O(1)` without shifting the backing
/// vector.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MergeNode {
    /// Token currently stored at this position.
    pub(crate) token_id: TokenId,
    /// Index of the previous live node, if any.
    pub(crate) prev: Option<MergeNodeSlot>,
    /// Index of the next live node, if any.
    pub(crate) next: Option<MergeNodeSlot>,
}

/// Sparse linked list stored inside a fixed-width `Vec`.
///
/// The module is designed for BPE's "merge adjacent pair" primitive:
/// - `Vec<Option<MergeNode>>` provides stable integer positions (`MergeNodeSlot`).
/// - Live traversal follows `prev/next` links from `head`.
/// - Removing a node creates a tombstone (`None`) instead of shifting indices.
///
/// Structural flow:
///
/// ```text
/// Allocation axis (stable slots):
///   slots[0]  slots[1]  slots[2]  slots[3] ...
///      |         |         |         |
///      v         v         v         v
///     Some      Some      Some      Some
///
/// Logical merge_sequence axis (links):
///   head -> [0] <-> [1] <-> [2] <-> [3]
///
/// After splice(1,2):
///   head -> [0] <-> [1*] <-> [3]
///                  |
///                  +-- token_id replaced by merged token
///   slots[2] = None   (tombstone, slot retained for index stability)
/// ```
///
/// This layout guarantees O(1) local rewrites and avoids index invalidation in pair-location
/// bookkeeping maintained by the trainer.
#[derive(Debug)]
pub(crate) struct MergeSequence {
    /// Slots for live nodes and tombstones from earlier merges.
    pub(crate) slots: Vec<Option<MergeNode>>,
    /// Index of the first live node in the linked structure.
    pub(crate) head: Option<MergeNodeSlot>,
}

impl MergeSequence {
    /// Creates a merge_sequence containing exactly one pre-tokenized token.
    pub(crate) fn from_token_id(token_id: TokenId) -> Self {
        Self {
            slots: vec![Some(MergeNode {
                token_id,
                prev: None,
                next: None,
            })],
            head: Some(0),
        }
    }

    /// Creates a byte-level merge_sequence where each input byte becomes one node.
    pub(crate) fn new(bytes: &[u8]) -> Self {
        let len = bytes.len();
        let slots = bytes
            .iter()
            .enumerate()
            .map(|(index, &byte)| {
                let pos = MergeNodeSlot::try_from(index)
                    .expect("merge_sequence length exceeds MergeNodeSlot capacity");
                Some(MergeNode {
                    token_id: TokenId::from(byte),
                    prev: pos.checked_sub(1),
                    next: (index + 1 < len).then_some(pos + 1),
                })
            })
            .collect();

        Self {
            slots,
            head: (!bytes.is_empty()).then_some(0),
        }
    }

    /// Iterates over live nodes in linked-list order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (MergeNodeSlot, MergeNode)> + '_ {
        let mut current = self.head;
        std::iter::from_fn(move || {
            let pos = current?;
            let node =
                self.slots[pos as usize].expect("merge_sequence iterator visited a removed node");
            current = node.next;
            Some((pos, node))
        })
    }

    /// Replaces an adjacent `[left, right]` pair with an in-place merged node.
    ///
    /// Pointer-rewrite sequence:
    ///
    /// ```text
    /// Input topology:
    ///   prev <-> left <-> right <-> next
    ///
    /// Step A: create merged node payload
    ///   merged.prev = left.prev
    ///   merged.next = right.next
    ///
    /// Step B: reconnect neighbors to left slot
    ///   if prev exists: prev.next = left
    ///   else:           head = left
    ///   if next exists: next.prev = left
    ///
    /// Step C: finalize slots
    ///   slots[left]  = Some(merged)
    ///   slots[right] = None
    ///
    /// Output topology:
    ///   prev <-> left(merged) <-> next
    /// ```
    ///
    /// Returning `left` lets callers reuse the surviving position when updating pair-location
    /// indexes, which prevents ambiguity about where the merged token now lives.
    pub(crate) fn splice(
        &mut self,
        left: MergeNodeSlot,
        right: MergeNodeSlot,
        new_token_id: TokenId,
    ) -> MergeNodeSlot {
        let left_node = self.slots[left as usize].expect("left splice node must exist");
        let right_node = self.slots[right as usize].expect("right splice node must exist");
        debug_assert_eq!(
            left_node.next,
            Some(right),
            "splice requires adjacent nodes"
        );

        let merged = MergeNode {
            token_id: new_token_id,
            prev: left_node.prev,
            next: right_node.next,
        };

        if let Some(prev_pos) = merged.prev {
            self.slots[prev_pos as usize]
                .as_mut()
                .expect("previous splice node must exist")
                .next = Some(left);
        } else {
            self.head = Some(left);
        }

        if let Some(next_pos) = merged.next {
            self.slots[next_pos as usize]
                .as_mut()
                .expect("next splice node must exist")
                .prev = Some(left);
        }

        self.slots[left as usize] = Some(merged);
        self.slots[right as usize] = None;
        left
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_merge_sequence_preserves_byte_order_and_links() {
        let merge_sequence = MergeSequence::new(b"abc");
        let slots: Vec<_> = merge_sequence.iter().collect();

        assert_eq!(merge_sequence.head, Some(0));
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].0, 0);
        assert_eq!(slots[0].1.token_id, b'a' as u32);
        assert_eq!(slots[0].1.prev, None);
        assert_eq!(slots[0].1.next, Some(1));
        assert_eq!(slots[1].0, 1);
        assert_eq!(slots[1].1.token_id, b'b' as u32);
        assert_eq!(slots[1].1.prev, Some(0));
        assert_eq!(slots[1].1.next, Some(2));
        assert_eq!(slots[2].0, 2);
        assert_eq!(slots[2].1.token_id, b'c' as u32);
        assert_eq!(slots[2].1.prev, Some(1));
        assert_eq!(slots[2].1.next, None);
    }

    #[test]
    fn empty_merge_sequence_has_no_head_or_nodes() {
        let merge_sequence = MergeSequence::new(b"");

        assert_eq!(merge_sequence.head, None);
        assert!(merge_sequence.slots.is_empty());
        assert_eq!(merge_sequence.iter().count(), 0);
    }

    #[test]
    fn splice_updates_links_for_middle_pair() {
        let mut merge_sequence = MergeSequence::new(b"abcd");

        let merged_slot = merge_sequence.splice(1, 2, 999);
        let slots: Vec<(MergeNodeSlot, u32)> = merge_sequence
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_slot, 1);
        assert_eq!(slots, vec![(0, b'a' as u32), (1, 999), (3, b'd' as u32)]);
        assert_eq!(merge_sequence.head, Some(0));
        assert_eq!(merge_sequence.slots.len(), 4);
        assert_eq!(
            merge_sequence.slots[0].expect("node 0 should exist").next,
            Some(1)
        );
        assert_eq!(
            merge_sequence.slots[3].expect("node 3 should exist").prev,
            Some(1)
        );
        assert!(merge_sequence.slots[2].is_none());
    }

    #[test]
    fn splice_updates_head_when_merging_first_pair() {
        let mut merge_sequence = MergeSequence::new(b"abc");

        let merged_slot = merge_sequence.splice(0, 1, 777);
        let slots: Vec<(MergeNodeSlot, u32)> = merge_sequence
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_slot, 0);
        assert_eq!(merge_sequence.head, Some(0));
        assert_eq!(slots, vec![(0, 777), (2, b'c' as u32)]);
        assert_eq!(
            merge_sequence.slots[2]
                .expect("tail node should exist")
                .prev,
            Some(0)
        );
        assert!(merge_sequence.slots[1].is_none());
    }

    #[test]
    fn repeated_splices_reuse_existing_capacity() {
        let mut merge_sequence = MergeSequence::new(b"aaaa");

        let first = merge_sequence.splice(0, 1, 256);
        let second = merge_sequence.splice(first, 2, 257);

        let slots: Vec<(MergeNodeSlot, u32)> = merge_sequence
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(second, 0);
        assert_eq!(merge_sequence.slots.len(), 4);
        assert_eq!(slots, vec![(0, 257), (3, b'a' as u32)]);
        assert!(merge_sequence.slots[1].is_none());
        assert!(merge_sequence.slots[2].is_none());
    }

    #[test]
    fn from_token_id_creates_single_node_merge_sequence() {
        let merge_sequence = MergeSequence::from_token_id(42);
        let slots: Vec<_> = merge_sequence.iter().collect();

        assert_eq!(merge_sequence.head, Some(0));
        assert_eq!(merge_sequence.slots.len(), 1);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].0, 0);
        assert_eq!(slots[0].1.token_id, 42);
        assert_eq!(slots[0].1.prev, None);
        assert_eq!(slots[0].1.next, None);
    }

    #[test]
    fn splice_on_two_node_merge_sequence_produces_single_live_node() {
        let mut merge_sequence = MergeSequence::new(b"ab");

        let merged_slot = merge_sequence.splice(0, 1, 300);
        let slots: Vec<_> = merge_sequence.iter().collect();

        assert_eq!(merged_slot, 0);
        assert_eq!(merge_sequence.head, Some(0));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].0, 0);
        assert_eq!(slots[0].1.token_id, 300);
        assert_eq!(slots[0].1.prev, None);
        assert_eq!(slots[0].1.next, None);
        assert!(merge_sequence.slots[1].is_none());
    }

    #[test]
    fn splice_updates_tail_when_merging_last_pair() {
        let mut merge_sequence = MergeSequence::new(b"abc");

        let merged_slot = merge_sequence.splice(1, 2, 500);
        let slots: Vec<(MergeNodeSlot, u32)> = merge_sequence
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_slot, 1);
        assert_eq!(slots, vec![(0, b'a' as u32), (1, 500)]);
        assert_eq!(
            merge_sequence.slots[0].expect("head should exist").next,
            Some(1)
        );
        let tail = merge_sequence.slots[1].expect("merged tail should exist");
        assert_eq!(tail.prev, Some(0));
        assert_eq!(tail.next, None);
        assert!(merge_sequence.slots[2].is_none());
    }

    #[test]
    fn iter_follows_updated_links_after_multiple_splices() {
        let mut merge_sequence = MergeSequence::new(b"abcde");

        let first = merge_sequence.splice(1, 2, 600);
        let second = merge_sequence.splice(first, 3, 601);

        let slots: Vec<(MergeNodeSlot, u32)> = merge_sequence
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(second, 1);
        assert_eq!(slots, vec![(0, b'a' as u32), (1, 601), (4, b'e' as u32)]);
        assert_eq!(
            merge_sequence.slots[0].expect("head should exist").next,
            Some(1)
        );
        assert_eq!(
            merge_sequence.slots[4].expect("tail should exist").prev,
            Some(1)
        );
        assert!(merge_sequence.slots[2].is_none());
        assert!(merge_sequence.slots[3].is_none());
    }

    #[test]
    fn splice_keeps_head_when_merging_non_head_pair() {
        let mut merge_sequence = MergeSequence::new(b"abcd");

        merge_sequence.splice(2, 3, 700);
        let slots: Vec<(MergeNodeSlot, u32)> = merge_sequence
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merge_sequence.head, Some(0));
        assert_eq!(slots, vec![(0, b'a' as u32), (1, b'b' as u32), (2, 700)]);
        assert_eq!(
            merge_sequence.slots[1].expect("node 1 should exist").next,
            Some(2)
        );
        assert_eq!(
            merge_sequence.slots[2].expect("node 2 should exist").prev,
            Some(1)
        );
        assert_eq!(
            merge_sequence.slots[2].expect("node 2 should exist").next,
            None
        );
        assert!(merge_sequence.slots[3].is_none());
    }
}
