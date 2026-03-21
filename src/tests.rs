use crate::bpe::BPE;
use crate::chain::Chain;
use crate::types::{BASE_VOCAB_SIZE, TokenId};

#[test]
fn chain_splice_updates_links_for_middle_pair() {
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
fn train_learns_most_frequent_pair_and_roundtrips() {
    let mut bpe = BPE::new("(?s).+");
    bpe.train(257, ["abababa"]);

    assert_eq!(bpe.vocab.get(&256), Some(&b"ab".to_vec()));
    assert_eq!(bpe.merge_map.get(&(b'a' as u32, b'b' as u32)), Some(&256));

    let encoded = bpe.encode("abababa");
    assert_eq!(encoded, vec![256, 256, 256, b'a' as u32]);
    assert_eq!(bpe.decode(encoded), b"abababa");
}

#[test]
fn training_handles_overlapping_pairs_without_corrupting_state() {
    let mut bpe = BPE::new("(?s).+");
    bpe.train(258, ["aaaa"]);

    assert_eq!(bpe.vocab.get(&256), Some(&b"aa".to_vec()));
    assert_eq!(bpe.vocab.get(&257), Some(&b"aaaa".to_vec()));
    assert_eq!(bpe.encode("aaaa"), vec![257]);
    assert_eq!(bpe.decode(vec![257]), b"aaaa");
}

#[test]
fn split_pattern_keeps_merges_scoped_to_each_match() {
    let mut bpe = BPE::new("\\S+");
    bpe.train(257, ["go go", "go stop"]);

    assert_eq!(bpe.vocab.get(&256), Some(&b"go".to_vec()));
    assert_eq!(
        bpe.encode("go stop go"),
        vec![256, b's' as u32, b't' as u32, b'o' as u32, b'p' as u32, 256]
    );
    assert_eq!(
        bpe.decode(vec![
            256,
            b's' as u32,
            b't' as u32,
            b'o' as u32,
            b'p' as u32,
            256
        ]),
        b"gostopgo"
    );
}

#[test]
fn empty_inputs_and_unknown_token_ids_are_handled() {
    let mut bpe = BPE::new("(?s).+");
    bpe.train(300, std::iter::empty::<&str>());

    assert!(bpe.encode("").is_empty());
    assert!(bpe.decode(Vec::new()).is_empty());
    assert_eq!(bpe.decode(vec![999_999]), Vec::<u8>::new());
    assert!(bpe.merge_map.is_empty());
}

#[test]
fn requesting_base_vocab_size_keeps_byte_level_encoding() {
    let mut bpe = BPE::new("(?s).+");
    bpe.train(BASE_VOCAB_SIZE, ["banana"]);

    let encoded = bpe.encode("banana");
    assert_eq!(
        encoded,
        b"banana"
            .iter()
            .map(|&byte| TokenId::from(byte))
            .collect::<Vec<_>>()
    );
    assert_eq!(bpe.decode(encoded), b"banana");
    assert!(bpe.merge_map.is_empty());
}

#[test]
fn retraining_replaces_previous_merge_state() {
    let mut bpe = BPE::new("(?s).+");
    bpe.train(257, ["aaaa"]);
    let first_encoding = bpe.encode("aaaa");
    assert_eq!(first_encoding, vec![256, 256]);

    bpe.train(257, ["bbbb"]);

    assert_eq!(bpe.vocab.get(&256), Some(&b"bb".to_vec()));
    assert_eq!(bpe.encode("bbbb"), vec![256, 256]);
    assert_eq!(bpe.encode("aaaa"), vec![b'a' as u32; 4]);
}

#[test]
fn try_new_returns_error_for_invalid_regex() {
    assert!(BPE::try_new("(").is_err());
}
