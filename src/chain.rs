use crate::types::{NodePos, TokenId};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub(crate) token_id: TokenId,
    pub(crate) prev: Option<NodePos>,
    pub(crate) next: Option<NodePos>,
}

#[derive(Debug)]
pub(crate) struct Chain {
    pub(crate) nodes: Vec<Option<Node>>,
    pub(crate) head: Option<NodePos>,
}

impl Chain {
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

    pub(crate) fn new(bytes: &[u8]) -> Self {
        let len = bytes.len();
        let nodes = bytes
            .iter()
            .enumerate()
            .map(|(index, &byte)| {
                let pos = index;
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

    pub(crate) fn iter(&self) -> impl Iterator<Item = (NodePos, Node)> + '_ {
        let mut current = self.head;
        std::iter::from_fn(move || {
            let pos = current?;
            let node = self.nodes[pos].expect("chain iterator visited a removed node");
            current = node.next;
            Some((pos, node))
        })
    }

    /// Replaces the `[left, right]` pair with a new merged node, returning the new node's position.
    pub(crate) fn splice(
        &mut self,
        left: NodePos,
        right: NodePos,
        new_token_id: TokenId,
    ) -> NodePos {
        let left_node = self.nodes[left].expect("left splice node must exist");
        let right_node = self.nodes[right].expect("right splice node must exist");
        debug_assert_eq!(
            left_node.next,
            Some(right),
            "splice requires adjacent nodes"
        );

        let prev = left_node.prev;
        let next = right_node.next;
        let pos = self.nodes.len();

        if let Some(prev_pos) = prev {
            self.nodes[prev_pos]
                .as_mut()
                .expect("previous splice node must exist")
                .next = Some(pos);
        } else {
            self.head = Some(pos);
        }

        if let Some(next_pos) = next {
            self.nodes[next_pos]
                .as_mut()
                .expect("next splice node must exist")
                .prev = Some(pos);
        }

        self.nodes.push(Some(Node {
            token_id: new_token_id,
            prev,
            next,
        }));
        self.nodes[left] = None;
        self.nodes[right] = None;
        pos
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
        let nodes: Vec<(usize, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_pos, 4);
        assert_eq!(nodes, vec![(0, b'a' as u32), (4, 999), (3, b'd' as u32)]);
        assert_eq!(chain.head, Some(0));
        assert_eq!(chain.nodes[0].expect("node 0 should exist").next, Some(4));
        assert_eq!(chain.nodes[3].expect("node 3 should exist").prev, Some(4));
    }

    #[test]
    fn splice_updates_head_when_merging_first_pair() {
        let mut chain = Chain::new(b"abc");

        let merged_pos = chain.splice(0, 1, 777);
        let nodes: Vec<(usize, u32)> = chain
            .iter()
            .map(|(pos, node)| (pos, node.token_id))
            .collect();

        assert_eq!(merged_pos, 3);
        assert_eq!(chain.head, Some(3));
        assert_eq!(nodes, vec![(3, 777), (2, b'c' as u32)]);
        assert_eq!(
            chain.nodes[2].expect("tail node should exist").prev,
            Some(3)
        );
    }
}
