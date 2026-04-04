use crate::types::TokenIdPair;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone)]
struct MergeCandidateBucketNode {
    count: u32,
    token_id_pairs: FxHashSet<TokenIdPair>,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Frequency-indexed merge-candidate buckets linked in descending frequency order.
///
/// The highest-frequency non-empty bucket is tracked explicitly for O(1) access.
#[derive(Debug, Default, Clone)]
pub(crate) struct MergeCandidateBuckets {
    nodes: Vec<Option<MergeCandidateBucketNode>>,
    free_nodes: Vec<usize>,
    count_to_node: FxHashMap<u32, usize>,
    head: Option<usize>,
}

impl MergeCandidateBuckets {
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.free_nodes.clear();
        self.count_to_node.clear();
        self.head = None;
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    pub(crate) fn best_token_id_pair(&self) -> Option<TokenIdPair> {
        let head = self.head?;
        self.node(head).token_id_pairs.iter().min().copied()
    }

    pub(crate) fn insert(&mut self, count: u32, token_id_pair: TokenIdPair) {
        debug_assert!(count > 0, "count buckets must be positive");
        let node_index = self.ensure_bucket(count);
        self.node_mut(node_index)
            .token_id_pairs
            .insert(token_id_pair);
    }

    pub(crate) fn update_token_id_pair(
        &mut self,
        token_id_pair: TokenIdPair,
        old_count: u32,
        new_count: u32,
    ) {
        if old_count > 0 {
            self.remove_from_bucket(old_count, token_id_pair);
        }
        if new_count > 0 {
            self.insert(new_count, token_id_pair);
        }
    }

    fn node(&self, index: usize) -> &MergeCandidateBucketNode {
        self.nodes[index]
            .as_ref()
            .expect("merge-candidate bucket node must exist")
    }

    fn node_mut(&mut self, index: usize) -> &mut MergeCandidateBucketNode {
        self.nodes[index]
            .as_mut()
            .expect("merge-candidate bucket node must exist")
    }

    fn allocate_node(&mut self, count: u32) -> usize {
        let node = MergeCandidateBucketNode {
            count,
            token_id_pairs: FxHashSet::default(),
            prev: None,
            next: None,
        };

        if let Some(index) = self.free_nodes.pop() {
            self.nodes[index] = Some(node);
            index
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        }
    }

    fn ensure_bucket(&mut self, count: u32) -> usize {
        if let Some(&existing) = self.count_to_node.get(&count) {
            return existing;
        }

        let new_index = self.allocate_node(count);
        self.count_to_node.insert(count, new_index);

        let Some(mut current) = self.head else {
            self.head = Some(new_index);
            return new_index;
        };

        if self.node(current).count < count {
            self.node_mut(current).prev = Some(new_index);
            self.node_mut(new_index).next = Some(current);
            self.head = Some(new_index);
            return new_index;
        }

        while let Some(next) = self.node(current).next {
            if self.node(next).count < count {
                break;
            }
            current = next;
        }

        let next = self.node(current).next;
        self.node_mut(new_index).prev = Some(current);
        self.node_mut(new_index).next = next;
        self.node_mut(current).next = Some(new_index);
        if let Some(next) = next {
            self.node_mut(next).prev = Some(new_index);
        }

        new_index
    }

    fn remove_from_bucket(&mut self, count: u32, token_id_pair: TokenIdPair) {
        let node_index = *self
            .count_to_node
            .get(&count)
            .expect("existing merge-candidate bucket must exist");
        let became_empty = {
            let node = self.node_mut(node_index);
            node.token_id_pairs.remove(&token_id_pair);
            node.token_id_pairs.is_empty()
        };

        if !became_empty {
            return;
        }

        let (prev, next) = {
            let node = self.node(node_index);
            (node.prev, node.next)
        };

        if let Some(prev) = prev {
            self.node_mut(prev).next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.node_mut(next).prev = prev;
        }

        self.count_to_node.remove(&count);
        self.nodes[node_index] = None;
        self.free_nodes.push(node_index);
    }
}

#[cfg(test)]
mod tests {
    use super::MergeCandidateBuckets;

    #[test]
    fn returns_none_when_empty() {
        let buckets = MergeCandidateBuckets::default();
        assert!(buckets.best_token_id_pair().is_none());
        assert!(buckets.is_empty());
    }

    #[test]
    fn tracks_highest_count_bucket_after_inserts() {
        let mut buckets = MergeCandidateBuckets::default();
        buckets.insert(2, (1, 2));
        buckets.insert(5, (4, 5));
        buckets.insert(5, (3, 8));

        assert_eq!(buckets.best_token_id_pair(), Some((3, 8)));
    }

    #[test]
    fn updates_bucket_membership_when_count_changes() {
        let mut buckets = MergeCandidateBuckets::default();
        buckets.insert(3, (1, 1));
        buckets.insert(2, (2, 2));

        buckets.update_token_id_pair((2, 2), 2, 6);
        assert_eq!(buckets.best_token_id_pair(), Some((2, 2)));

        buckets.update_token_id_pair((2, 2), 6, 0);
        assert_eq!(buckets.best_token_id_pair(), Some((1, 1)));
    }

    #[test]
    fn clears_all_state() {
        let mut buckets = MergeCandidateBuckets::default();
        buckets.insert(3, (9, 9));
        buckets.clear();

        assert!(buckets.is_empty());
        assert!(buckets.best_token_id_pair().is_none());
    }
}
