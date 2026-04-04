use fast_bpe_rs::BPE;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

const DATASET_REPO: &str = "Salesforce/wikitext";
const DATASET_CONFIG: &str = "wikitext-103-raw-v1";
const TRAIN_SPLIT_PREFIX: &str = "train-";
const TARGET_VOCAB_SIZE: u32 = 1 << 15;

fn load_wikitext_train_docs() -> Result<Vec<String>, Box<dyn Error>> {
    let api = Api::new()?;
    let dataset = api.dataset(DATASET_REPO.to_owned());
    let repo_info = dataset.info()?;

    let mut train_files: Vec<_> = repo_info
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
    for train_file in train_files {
        let train_path = dataset.get(&train_file)?;
        let file = File::open(train_path)?;
        let parquet = SerializedFileReader::new(file)?;

        docs.extend(
            parquet
                .get_row_iter(None)?
                .filter_map(Result::ok)
                .filter_map(|row| row.get_string(0).ok().map(|text| text.trim().to_owned()))
                .filter(|text| !text.is_empty()),
        );
    }

    Ok(docs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let docs = load_wikitext_train_docs()?;

    println!(
        "RUN_CONTEXT dataset_repo={DATASET_REPO} dataset_config={DATASET_CONFIG} split_prefix={TRAIN_SPLIT_PREFIX} docs_loaded={} target_vocab_size={TARGET_VOCAB_SIZE}",
        docs.len(),
    );

    let mut bpe = BPE::new(None, None::<Vec<(String, u32)>>)?;
    let started = Instant::now();
    bpe.train(TARGET_VOCAB_SIZE, docs.iter());
    println!(
        "TRAIN_RUN mode=speed elapsed_ms={}",
        started.elapsed().as_millis()
    );
    println!("TRAINING_COMPLETE finished=true");

    Ok(())
}
