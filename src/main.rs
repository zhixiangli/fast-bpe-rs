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

struct CliOptions {
    dataset_config: &'static str,
    run_speed: bool,
    run_memory: bool,
}

fn parse_cli_options() -> Result<CliOptions, Box<dyn Error>> {
    let mut dataset_config = DATASET_CONFIG_DEFAULT;
    let mut speed_flag = false;
    let mut memory_flag = false;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--small-dataset" => dataset_config = DATASET_CONFIG_SMALL,
            "--speed" => speed_flag = true,
            "--memory" => memory_flag = true,
            "--help" | "-h" => {
                println!("Usage: fast-bpe-rs [--small-dataset] [--speed] [--memory]");
                println!(
                    "  --small-dataset   Use wikitext-2-raw-v1 instead of wikitext-103-raw-v1"
                );
                println!("  --speed           Run speed benchmark");
                println!("  --memory          Run memory benchmark");
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unrecognized argument: {arg}. Use --help to see available options"
                )
                .into());
            }
        }
    }

    let run_speed = speed_flag || !memory_flag;
    let run_memory = memory_flag || !speed_flag;

    Ok(CliOptions {
        dataset_config,
        run_speed,
        run_memory,
    })
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
    let options = parse_cli_options()?;
    let docs = load_wikitext_train_docs(options.dataset_config)?;

    println!(
        "RUN_CONTEXT dataset_repo={DATASET_REPO} dataset_config={} split_prefix={TRAIN_SPLIT_PREFIX} docs_loaded={} target_vocab_size={TARGET_VOCAB_SIZE} run_speed={} run_memory={}",
        options.dataset_config,
        docs.len(),
        options.run_speed,
        options.run_memory,
    );

    if options.run_speed {
        let mut bpe_speed = BPE::new(None, None::<Vec<(String, u32)>>)?;
        let speed_started = Instant::now();
        bpe_speed.train(TARGET_VOCAB_SIZE, docs.iter());
        let speed_elapsed = speed_started.elapsed();
        println!(
            "TRAIN_RUN mode=speed elapsed_ms={}",
            speed_elapsed.as_millis()
        );
    }

    if options.run_memory {
        let mut bpe_memory = BPE::new(None, None::<Vec<(String, u32)>>)?;
        let memory_region = Region::new(&GLOBAL);
        let memory_started = Instant::now();
        bpe_memory.train(TARGET_VOCAB_SIZE, docs.iter());
        let memory_elapsed = memory_started.elapsed();
        let memory_stats = memory_region.change();
        println!(
            "TRAIN_RUN mode=memory elapsed_ms={} bytes_allocated={} bytes_deallocated={} allocations={} deallocations={} reallocations={}",
            memory_elapsed.as_millis(),
            memory_stats.bytes_allocated,
            memory_stats.bytes_deallocated,
            memory_stats.allocations,
            memory_stats.deallocations,
            memory_stats.reallocations,
        );
    }

    println!(
        "TRAINING_COMPLETE run_speed={} run_memory={} finished=true",
        options.run_speed, options.run_memory
    );

    Ok(())
}
