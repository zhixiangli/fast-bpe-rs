use crate::types::{NONE, NodePos, TokenId};

/// One token inside a [`Chain`].
///
/// The `prev`/`next` links let us remove a merged pair in `O(1)` without shifting the backing
/// vector. A `NONE` sentinel replaces `Option<NodePos>` to halve the per-node footprint.
/// A node with `token_id == NONE` is a tombstone (removed by a previous merge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Node {
    /// Token currently stored at this position (`NONE` = tombstone).
    pub(crate) token_id: TokenId,
    /// Index of the previous live node (`NONE` = no predecessor).
    pub(crate) prev: NodePos,
    /// Index of the next live node (`NONE` = no successor).
    pub(crate) next: NodePos,
}

impl Node {
    /// Returns `true` if this slot has been removed by an earlier merge.
    #[inline]
    pub(crate) fn is_tombstone(self) -> bool {
        self.token_id == NONE
    }
}

/// Sparse linked list stored inside a fixed-width `Vec`.
///
/// Visual model:
///
/// ```text
/// nodes vec:  [0] <-> [1] <-> [2] <-> [3]
/// merge 1,2:  [0] <-> [1]     x    [3]
///                    ^ merged token now reuses the left slot
/// ```
///
/// Removed right-hand nodes become tombstones (`token_id == NONE`), while the left-hand node is
/// updated in place so repeated merges do not grow the backing allocation.
#[derive(Debug)]
pub(crate) struct Chain {
    /// Slots for live nodes and tombstones from earlier merges.
    pub(crate) nodes: Vec<Node>,
    /// Index of the first live node in the linked structure (`NONE` = empty).
    pub(crate) head: NodePos,
}

impl Chain {
    /// Creates a chain containing exactly one pre-tokenized token.
    pub(crate) fn from_token_id(token_id: TokenId) -> Self {
        Self {
            nodes: vec![Node {
                token_id,
                prev: NONE,
                next: NONE,
            }],
            head: 0,
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
                Node {
                    token_id: TokenId::from(byte),
                    prev: if pos > 0 { pos - 1 } else { NONE },
                    next: if index + 1 < len { pos + 1 } else { NONE },
                }
            })
            .collect();

        Self {
            nodes,
            head: if bytes.is_empty() { NONE } else { 0 },
        }
    }

    /// Iterates over live nodes in linked-list order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (NodePos, Node)> + '_ {
        let mut current = self.head;
        std::iter::from_fn(move || {
            if current == NONE {
                return None;
            }
            let pos = current;
            let node = self.nodes[pos as usize];
            debug_assert!(
                !node.is_tombstone(),
                "chain iterator visited a removed node"
            );
            current = node.next;
            Some((pos, node))
        })
    }

    /// Replaces an adjacent `[left, right]` pair with an in-place merged node.
    ///
    /// Before:
    ///
    /// ```text
    /// prev <-> left <-> right <-> next
    /// ```
    ///
    /// After:
    ///
    /// ```text
    /// prev <-> left(merged) <-> next
    /// ```
    pub(crate) fn splice(
        &mut self,
        left: NodePos,
        right: NodePos,
        new_token_id: TokenId,
    ) -> NodePos {
        let left_node = self.nodes[left as usize];
        let right_node = self.nodes[right as usize];
        debug_assert!(!left_node.is_tombstone(), "left splice node must exist");
        debug_assert!(!right_node.is_tombstone(), "right splice node must exist");
        debug_assert_eq!(left_node.next, right, "splice requires adjacent nodes");

        let merged = Node {
            token_id: new_token_id,
            prev: left_node.prev,
            next: right_node.next,
        };

        if merged.prev != NONE {
            self.nodes[merged.prev as usize].next = left;
        } else {
            self.head = left;
        }

        if merged.next != NONE {
            self.nodes[merged.next as usize].prev = left;
        }

        self.nodes[left as usize] = merged;
        self.nodes[right as usize].token_id = NONE; // tombstone
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

        assert_eq!(chain.head, 0);
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].0, 0);
        assert_eq!(nodes[0].1.token_id, b'a' as u32);
        assert_eq!(nodes[0].1.prev, NONE);
        assert_eq!(nodes[0].1.next, 1);
        assert_eq!(nodes[1].0, 1);
        assert_eq!(nodes[1].1.token_id, b'b' as u32);
        assert_eq!(nodes[1].1.prev, 0);
        assert_eq!(nodes[1].1.next, 2);
        assert_eq!(nodes[2].0, 2);
        assert_eq!(nodes[2].1.token_id, b'c' as u32);
        assert_eq!(nodes[2].1.prev, 1);
        assert_eq!(nodes[2].1.next, NONE);
    }

    #[test]
    fn empty_chain_has_no_head_or_nodes() {
        let chain = Chain::new(b"");

        assert_eq!(chain.head, NONE);
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
        assert_eq!(chain.head, 0);
        assert_eq!(chain.nodes.len(), 4);
        assert_eq!(chain.nodes[0].next, 1);
        assert_eq!(chain.nodes[3].prev, 1);
        assert!(chain.nodes[2].is_tombstone());
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
        assert_eq!(chain.head, 0);
        assert_eq!(nodes, vec![(0, 777), (2, b'c' as u32)]);
        assert_eq!(chain.nodes[2].prev, 0);
        assert!(chain.nodes[1].is_tombstone());
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
        assert!(chain.nodes[1].is_tombstone());
        assert!(chain.nodes[2].is_tombstone());
    }
}
