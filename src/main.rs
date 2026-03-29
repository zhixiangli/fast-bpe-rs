use fast_bpe_rs::BPE;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

const DATASET_REPO: &str = "roneneldan/TinyStories";
const TRAIN_FILES: &[&str] = &[
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
];
const TARGET_VOCAB_SIZE: u32 = 65_536;
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn load_tinystories_train_docs() -> Result<Vec<String>, Box<dyn Error>> {
    let api = Api::new()?;
    let dataset = api.dataset(DATASET_REPO.to_owned());

    let mut docs = Vec::new();
    for train_file in TRAIN_FILES {
        let train_path = dataset.get(train_file)?;

        let file = File::open(train_path)?;
        let parquet = SerializedFileReader::new(file)?;
        let rows = parquet.get_row_iter(None)?;

        docs.extend(
            rows.filter_map(|row| row.ok())
                .filter_map(|row| row.get_string(0).ok().map(|text| text.trim().to_owned()))
                .filter(|text| !text.is_empty()),
        );
    }

    Ok(docs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let docs = load_tinystories_train_docs()?;
    let mut bpe = BPE::try_new(SPLIT_PATTERN)?;

    let train_started = Instant::now();
    bpe.train(TARGET_VOCAB_SIZE, docs.iter());
    let train_elapsed = train_started.elapsed();

    println!(
        "Loaded {} documents from {} train shards in {DATASET_REPO}",
        docs.len(),
        TRAIN_FILES.len()
    );
    println!("Finished BPE training up to vocab size {TARGET_VOCAB_SIZE}");
    println!("Training elapsed: {:.3?}", train_elapsed);

    let paragraph = "Fast-BPE-RS now supports both library usage and `cargo run` demos. \
After training on TinyStories, this paragraph is encoded into token ids and decoded back into text.";
    println!("\nOriginal paragraph:\n{paragraph}");

    let encoded = bpe.encode(paragraph);
    println!("\nEncoded token ids:\n{encoded:?}");

    let decoded_bytes = bpe.decode(encoded.iter().copied());
    let decoded_text = String::from_utf8(decoded_bytes)?;
    println!("\nDecoded paragraph:\n{decoded_text}");
    println!("\nRoundtrip exact match: {}", paragraph == decoded_text);

    Ok(())
}
