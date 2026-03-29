use crate::types::{NodePos, TokenId};

/// One token inside a [`Chain`].
///
/// The `prev`/`next` links let us remove a merged pair in `O(1)` without shifting the backing
/// vector.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    /// Token currently stored at this position.
    pub(crate) token_id: TokenId,
    /// Index of the previous live node, if any.
    pub(crate) prev: Option<NodePos>,
    /// Index of the next live node, if any.
    pub(crate) next: Option<NodePos>,
}

/// Sparse linked list stored inside a fixed-width `Vec`.
///
/// The module is designed for BPE's "merge adjacent pair" primitive:
/// - `Vec<Option<Node>>` provides stable integer positions (`NodePos`).
/// - Live traversal follows `prev/next` links from `head`.
/// - Removing a node creates a tombstone (`None`) instead of shifting indices.
///
/// Structural flow:
///
/// ```text
/// Allocation axis (stable slots):
///   nodes[0]  nodes[1]  nodes[2]  nodes[3] ...
///      |         |         |         |
///      v         v         v         v
///     Some      Some      Some      Some
///
/// Logical chain axis (links):
///   head -> [0] <-> [1] <-> [2] <-> [3]
///
/// After splice(1,2):
///   head -> [0] <-> [1*] <-> [3]
///                  |
///                  +-- token_id replaced by merged token
///   nodes[2] = None   (tombstone, slot retained for index stability)
/// ```
///
/// This layout guarantees O(1) local rewrites and avoids index invalidation in pair-location
/// bookkeeping maintained by the trainer.
#[derive(Debug)]
pub(crate) struct Chain {
    /// Slots for live nodes and tombstones from earlier merges.
    pub(crate) nodes: Vec<Option<Node>>,
    /// Index of the first live node in the linked structure.
    pub(crate) head: Option<NodePos>,
}

impl Chain {
    /// Creates a chain containing exactly one pre-tokenized token.
    pub(crate) fn from_token_id(token_id: TokenId) -> Self {
        Self {
            nodes: vec![Some(Node {
                token_id,
                prev: None,
                next: None,
            })],
            head: Some(0),
        }
    }

    /// Creates a byte-level chain where each input byte becomes one node.
    pub(crate) fn new(bytes: &[u8]) -> Self {
        let len = bytes.len();
        let nodes = bytes
            .iter()
            .enumerate()
            .map(|(index, &byte)| {
                let pos = NodePos::try_from(index).expect("chain length exceeds NodePos capacity");
                Some(Node {
                    token_id: TokenId::from(byte),
                    prev: pos.checked_sub(1),
                    next: (index + 1 < len).then_some(pos + 1),
                })
            })
            .collect();

        Self {
            nodes,
            head: (!bytes.is_empty()).then_some(0),
        }
    }

    /// Iterates over live nodes in linked-list order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (NodePos, Node)> + '_ {
        let mut current = self.head;
        std::iter::from_fn(move || {
            let pos = current?;
            let node = self.nodes[pos as usize].expect("chain iterator visited a removed node");
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
    ///   nodes[left]  = Some(merged)
    ///   nodes[right] = None
    ///
    /// Output topology:
    ///   prev <-> left(merged) <-> next
    /// ```
    ///
    /// Returning `left` lets callers reuse the surviving position when updating pair-location
    /// indexes, which prevents ambiguity about where the merged token now lives.
    pub(crate) fn splice(
        &mut self,
        left: NodePos,
        right: NodePos,
        new_token_id: TokenId,
    ) -> NodePos {
        let left_node = self.nodes[left as usize].expect("left splice node must exist");
        let right_node = self.nodes[right as usize].expect("right splice node must exist");
        debug_assert_eq!(
            left_node.next,
            Some(right),
            "splice requires adjacent nodes"
        );

        let merged = Node {
            token_id: new_token_id,
            prev: left_node.prev,
            next: right_node.next,
        };

        if let Some(prev_pos) = merged.prev {
            self.nodes[prev_pos as usize]
                .as_mut()
                .expect("previous splice node must exist")
                .next = Some(left);
        } else {
            self.head = Some(left);
        }

        if let Some(next_pos) = merged.next {
            self.nodes[next_pos as usize]
                .as_mut()
                .expect("next splice node must exist")
                .prev = Some(left);
        }

        self.nodes[left as usize] = Some(merged);
        self.nodes[right as usize] = None;
        left
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_chain_preserves_byte_order_and_links() {
        let chain = Chain::new(b"abc");
        let nodes: Vec<_> = chain.iter().collect();

        assert_eq!(chain.head, Some(0));
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].0, 0);
        assert_eq!(nodes[0].1.token_id, b'a' as u32);
        assert_eq!(nodes[0].1.prev, None);
        assert_eq!(nodes[0].1.next, Some(1));
        assert_eq!(nodes[1].0, 1);
        assert_eq!(nodes[1].1.token_id, b'b' as u32);
        assert_eq!(nodes[1].1.prev, Some(0));
        assert_eq!(nodes[1].1.next, Some(2));
        assert_eq!(nodes[2].0, 2);
        assert_eq!(nodes[2].1.token_id, b'c' as u32);
        assert_eq!(nodes[2].1.prev, Some(1));
        assert_eq!(nodes[2].1.next, None);
    }

    #[test]
    fn empty_chain_has_no_head_or_nodes() {
        let chain = Chain::new(b"");

        assert_eq!(chain.head, None);
        assert!(chain.nodes.is_empty());
        assert_eq!(chain.iter().count(), 0);
    }

    #[test]
    fn splice_updates_links_for_middle_pair() {
        let mut chain = Chain::new(b"abcd");

        let merged_pos = chain.splice(1, 2, 999);
        let nodes: Vec<(NodePos, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_pos, 1);
        assert_eq!(nodes, vec![(0, b'a' as u32), (1, 999), (3, b'd' as u32)]);
        assert_eq!(chain.head, Some(0));
        assert_eq!(chain.nodes.len(), 4);
        assert_eq!(chain.nodes[0].expect("node 0 should exist").next, Some(1));
        assert_eq!(chain.nodes[3].expect("node 3 should exist").prev, Some(1));
        assert!(chain.nodes[2].is_none());
    }

    #[test]
    fn splice_updates_head_when_merging_first_pair() {
        let mut chain = Chain::new(b"abc");

        let merged_pos = chain.splice(0, 1, 777);
        let nodes: Vec<(NodePos, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_pos, 0);
        assert_eq!(chain.head, Some(0));
        assert_eq!(nodes, vec![(0, 777), (2, b'c' as u32)]);
        assert_eq!(
            chain.nodes[2].expect("tail node should exist").prev,
            Some(0)
        );
        assert!(chain.nodes[1].is_none());
    }

    #[test]
    fn repeated_splices_reuse_existing_capacity() {
        let mut chain = Chain::new(b"aaaa");

        let first = chain.splice(0, 1, 256);
        let second = chain.splice(first, 2, 257);

        let nodes: Vec<(NodePos, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(second, 0);
        assert_eq!(chain.nodes.len(), 4);
        assert_eq!(nodes, vec![(0, 257), (3, b'a' as u32)]);
        assert!(chain.nodes[1].is_none());
        assert!(chain.nodes[2].is_none());
    }

    #[test]
    fn from_token_id_creates_single_node_chain() {
        let chain = Chain::from_token_id(42);
        let nodes: Vec<_> = chain.iter().collect();

        assert_eq!(chain.head, Some(0));
        assert_eq!(chain.nodes.len(), 1);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, 0);
        assert_eq!(nodes[0].1.token_id, 42);
        assert_eq!(nodes[0].1.prev, None);
        assert_eq!(nodes[0].1.next, None);
    }

    #[test]
    fn splice_on_two_node_chain_produces_single_live_node() {
        let mut chain = Chain::new(b"ab");

        let merged_pos = chain.splice(0, 1, 300);
        let nodes: Vec<_> = chain.iter().collect();

        assert_eq!(merged_pos, 0);
        assert_eq!(chain.head, Some(0));
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, 0);
        assert_eq!(nodes[0].1.token_id, 300);
        assert_eq!(nodes[0].1.prev, None);
        assert_eq!(nodes[0].1.next, None);
        assert!(chain.nodes[1].is_none());
    }

    #[test]
    fn splice_updates_tail_when_merging_last_pair() {
        let mut chain = Chain::new(b"abc");

        let merged_pos = chain.splice(1, 2, 500);
        let nodes: Vec<(NodePos, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_pos, 1);
        assert_eq!(nodes, vec![(0, b'a' as u32), (1, 500)]);
        assert_eq!(chain.nodes[0].expect("head should exist").next, Some(1));
        let tail = chain.nodes[1].expect("merged tail should exist");
        assert_eq!(tail.prev, Some(0));
        assert_eq!(tail.next, None);
        assert!(chain.nodes[2].is_none());
    }

    #[test]
    fn iter_follows_updated_links_after_multiple_splices() {
        let mut chain = Chain::new(b"abcde");

        let first = chain.splice(1, 2, 600);
        let second = chain.splice(first, 3, 601);

        let nodes: Vec<(NodePos, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(second, 1);
        assert_eq!(nodes, vec![(0, b'a' as u32), (1, 601), (4, b'e' as u32)]);
        assert_eq!(chain.nodes[0].expect("head should exist").next, Some(1));
        assert_eq!(chain.nodes[4].expect("tail should exist").prev, Some(1));
        assert!(chain.nodes[2].is_none());
        assert!(chain.nodes[3].is_none());
    }

    #[test]
    fn splice_keeps_head_when_merging_non_head_pair() {
        let mut chain = Chain::new(b"abcd");

        chain.splice(2, 3, 700);
        let nodes: Vec<(NodePos, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(chain.head, Some(0));
        assert_eq!(nodes, vec![(0, b'a' as u32), (1, b'b' as u32), (2, 700)]);
        assert_eq!(chain.nodes[1].expect("node 1 should exist").next, Some(2));
        assert_eq!(chain.nodes[2].expect("node 2 should exist").prev, Some(1));
        assert_eq!(chain.nodes[2].expect("node 2 should exist").next, None);
        assert!(chain.nodes[3].is_none());
    }
}
