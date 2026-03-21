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
