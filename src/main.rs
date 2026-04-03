use fast_bpe_rs::BPE;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::env;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

const DATASET_REPO: &str = "Salesforce/wikitext";
const DATASET_CONFIG_DEFAULT: &str = "wikitext-103-raw-v1";
const DATASET_CONFIG_SMALL: &str = "wikitext-2-raw-v1";
const TRAIN_SPLIT_PREFIX: &str = "train-";
const TARGET_VOCAB_SIZE: u32 = 1 << 15;

struct CliOptions {
    dataset_config: &'static str,
}

fn parse_cli_options() -> Result<CliOptions, Box<dyn Error>> {
    let mut dataset_config = DATASET_CONFIG_DEFAULT;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--small-dataset" => dataset_config = DATASET_CONFIG_SMALL,
            "--help" | "-h" => {
                println!("Usage: fast-bpe-rs [--small-dataset]");
                println!(
                    "  --small-dataset   Use wikitext-2-raw-v1 instead of wikitext-103-raw-v1"
                );
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

    Ok(CliOptions { dataset_config })
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
        "RUN_CONTEXT dataset_repo={DATASET_REPO} dataset_config={} split_prefix={TRAIN_SPLIT_PREFIX} docs_loaded={} target_vocab_size={TARGET_VOCAB_SIZE}",
        options.dataset_config,
        docs.len(),
    );

    let mut bpe = BPE::new(None, None::<Vec<(String, u32)>>)?;
    let started = Instant::now();
    bpe.train(TARGET_VOCAB_SIZE, docs.iter());
    let elapsed = started.elapsed();
    println!("TRAIN_RUN mode=speed elapsed_ms={}", elapsed.as_millis());

    println!("TRAINING_COMPLETE finished=true");

    Ok(())
}
