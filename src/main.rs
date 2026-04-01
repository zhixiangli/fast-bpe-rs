use fast_bpe_rs::BPE;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use stats_alloc::{Region, StatsAlloc};
use std::alloc::System;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

const DATASET_REPO: &str = "Salesforce/wikitext";
const DATASET_CONFIG: &str = "wikitext-103-raw-v1";
const TRAIN_SPLIT_PREFIX: &str = "train-";
const TARGET_VOCAB_SIZE: u32 = 1 << 15;

#[global_allocator]
static GLOBAL: StatsAlloc<System> = StatsAlloc::system();

fn load_wikitext_train_docs() -> Result<Vec<String>, Box<dyn Error>> {
    let api = Api::new()?;
    let dataset = api.dataset(DATASET_REPO.to_owned());
    let repo_info = dataset.info()?;

    let mut train_files: Vec<String> = repo_info
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|path| {
            path.starts_with(DATASET_CONFIG)
                && path.ends_with(".parquet")
                && path
                    .rsplit('/')
                    .next()
                    .is_some_and(|name| name.starts_with(TRAIN_SPLIT_PREFIX))
        })
        .collect();
    train_files.sort_unstable();

    let mut docs = Vec::new();
    for train_file in &train_files {
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
    let docs = load_wikitext_train_docs()?;
    let mut bpe_speed = BPE::try_new()?;
    let mut bpe_memory = BPE::try_new()?;

    let speed_started = Instant::now();
    bpe_speed.train(TARGET_VOCAB_SIZE, docs.iter());
    let speed_elapsed = speed_started.elapsed();

    let memory_region = Region::new(&GLOBAL);
    let memory_started = Instant::now();
    bpe_memory.train(TARGET_VOCAB_SIZE, docs.iter());
    let memory_elapsed = memory_started.elapsed();
    let memory_stats = memory_region.change();

    println!(
        "Loaded {} documents from {DATASET_REPO}/{DATASET_CONFIG} ({TRAIN_SPLIT_PREFIX}*.parquet)",
        docs.len(),
    );
    println!("Finished BPE training up to vocab size {TARGET_VOCAB_SIZE} (speed run)");
    println!("Speed run elapsed: {:.3?}", speed_elapsed);
    println!("Finished BPE training up to vocab size {TARGET_VOCAB_SIZE} (memory profile run)");
    println!("Memory profile elapsed: {:.3?}", memory_elapsed);
    println!(
        "Memory profile bytes allocated: {}",
        memory_stats.bytes_allocated
    );
    println!(
        "Memory profile bytes deallocated: {}",
        memory_stats.bytes_deallocated
    );
    println!(
        "Memory profile reallocations: {}",
        memory_stats.reallocations
    );
    println!("Memory profile allocations: {}", memory_stats.allocations);
    println!(
        "Memory profile deallocations: {}",
        memory_stats.deallocations
    );

    let paragraph = "Fast-BPE-RS now supports both library usage and `cargo run` demos. \
After training on WikiText, this paragraph is encoded into token ids and decoded back into text.";
    println!("\nOriginal paragraph:\n{paragraph}");

    let encoded = bpe_speed.encode(paragraph);
    println!("\nEncoded token ids:\n{encoded:?}");

    let decoded_bytes = bpe_speed.decode(encoded.iter().copied());
    let decoded_text = String::from_utf8(decoded_bytes)?;
    println!("\nDecoded paragraph:\n{decoded_text}");
    println!("\nRoundtrip exact match: {}", paragraph == decoded_text);

    Ok(())
}
