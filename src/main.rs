use fast_bpe_rs::BPE;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use stats_alloc::{Region, StatsAlloc};
use std::alloc::System;
use std::env;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

const DATASET_REPO: &str = "Salesforce/wikitext";
const DATASET_CONFIG_DEFAULT: &str = "wikitext-103-raw-v1";
const DATASET_CONFIG_SMALL: &str = "wikitext-2-raw-v1";
const TRAIN_SPLIT_PREFIX: &str = "train-";
const TARGET_VOCAB_SIZE: u32 = 1 << 15;

#[global_allocator]
static GLOBAL: StatsAlloc<System> = StatsAlloc::system();

fn parse_dataset_config() -> &'static str {
    let use_small_dataset = env::args().skip(1).any(|arg| arg == "--small-dataset");
    if use_small_dataset {
        DATASET_CONFIG_SMALL
    } else {
        DATASET_CONFIG_DEFAULT
    }
}

fn load_wikitext_train_docs(dataset_config: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let api = Api::new()?;
    let dataset = api.dataset(DATASET_REPO.to_owned());
    let repo_info = dataset.info()?;

    let mut train_files: Vec<String> = repo_info
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|path| {
            path.starts_with(dataset_config)
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
    let dataset_config = parse_dataset_config();
    let docs = load_wikitext_train_docs(dataset_config)?;
    let mut bpe_speed = BPE::new(None, None::<Vec<(String, u32)>>)?;
    let mut bpe_memory = BPE::new(None, None::<Vec<(String, u32)>>)?;

    let speed_started = Instant::now();
    bpe_speed.train(TARGET_VOCAB_SIZE, docs.iter());
    let speed_elapsed = speed_started.elapsed();

    let memory_region = Region::new(&GLOBAL);
    let memory_started = Instant::now();
    bpe_memory.train(TARGET_VOCAB_SIZE, docs.iter());
    let memory_elapsed = memory_started.elapsed();
    let memory_stats = memory_region.change();

    println!(
        "RUN_CONTEXT dataset_repo={DATASET_REPO} dataset_config={dataset_config} split_prefix={TRAIN_SPLIT_PREFIX} docs_loaded={} target_vocab_size={TARGET_VOCAB_SIZE}",
        docs.len(),
    );
    println!(
        "TRAIN_RUN mode=speed elapsed_ms={}",
        speed_elapsed.as_millis()
    );
    println!(
        "TRAIN_RUN mode=memory elapsed_ms={} bytes_allocated={} bytes_deallocated={} allocations={} deallocations={} reallocations={}",
        memory_elapsed.as_millis(),
        memory_stats.bytes_allocated,
        memory_stats.bytes_deallocated,
        memory_stats.allocations,
        memory_stats.deallocations,
        memory_stats.reallocations,
    );
    println!("TRAINING_COMPLETE speed_and_memory_runs_finished=true");

    Ok(())
}
