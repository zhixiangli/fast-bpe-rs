use fast_bpe_rs::BPE;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::env;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

const DATASET_REPO: &str = "Salesforce/wikitext";
const TRAIN_FILE: &str = "wikitext-2-raw-v1/train-00000-of-00001.parquet";
const TARGET_VOCAB_SIZE: u32 = 5_000;
const DEFAULT_MAX_TRAIN_DOCS: usize = 512;
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn limit_training_docs(docs: &mut Vec<String>, max_docs: usize) {
    if docs.len() > max_docs {
        docs.truncate(max_docs);
    }
}

fn max_train_docs() -> usize {
    env::var("FAST_BPE_RS_MAX_TRAIN_DOCS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MAX_TRAIN_DOCS)
}

fn load_wikitext_train_docs(max_docs: usize) -> Result<Vec<String>, Box<dyn Error>> {
    let api = Api::new()?;
    let train_path = api.dataset(DATASET_REPO.to_owned()).get(TRAIN_FILE)?;

    let file = File::open(train_path)?;
    let parquet = SerializedFileReader::new(file)?;
    let rows = parquet.get_row_iter(None)?;

    let mut docs = rows
        .filter_map(|row| row.ok())
        .filter_map(|row| row.get_string(0).ok().map(|text| text.trim().to_owned()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    limit_training_docs(&mut docs, max_docs);

    Ok(docs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_docs = max_train_docs();
    let docs = load_wikitext_train_docs(max_docs)?;
    let mut bpe = BPE::try_new(SPLIT_PATTERN)?;

    let train_started = Instant::now();
    bpe.train(TARGET_VOCAB_SIZE, docs.iter());
    let train_elapsed = train_started.elapsed();

    println!(
        "Loaded {} documents from {DATASET_REPO}/{TRAIN_FILE}",
        docs.len()
    );
    if docs.len() == max_docs {
        println!(
            "Training demo limited to {max_docs} documents (override with FAST_BPE_RS_MAX_TRAIN_DOCS)"
        );
    }
    println!("Finished BPE training up to vocab size {TARGET_VOCAB_SIZE}");
    println!("Training elapsed: {:.3?}", train_elapsed);

    let paragraph = "Fast-BPE-RS now supports both library usage and `cargo run` demos. \
After training on WikiText, this paragraph is encoded into token ids and decoded back into text.";
    println!("\nOriginal paragraph:\n{paragraph}");

    let encoded = bpe.encode(paragraph);
    println!("\nEncoded token ids:\n{encoded:?}");

    let decoded_bytes = bpe.decode(encoded.iter().copied());
    let decoded_text = String::from_utf8(decoded_bytes)?;
    println!("\nDecoded paragraph:\n{decoded_text}");
    println!("\nRoundtrip exact match: {}", paragraph == decoded_text);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::limit_training_docs;

    #[test]
    fn limit_training_docs_truncates_only_when_needed() {
        let mut docs = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        limit_training_docs(&mut docs, 2);
        assert_eq!(docs, vec!["a", "b"]);

        limit_training_docs(&mut docs, 4);
        assert_eq!(docs, vec!["a", "b"]);
    }
}
