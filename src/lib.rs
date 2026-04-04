//! Crate entry point for the Rust library and Python extension module.

mod bpe;
mod error;
mod merge_sequence;
mod python;
mod types;

pub use crate::bpe::BPE;
pub use crate::error::BPEError;
pub use crate::python::PyBPE;

use pyo3::prelude::*;

/// Registers the `fast_bpe_rs` Python module and exposes the `BPE` class.
#[pymodule]
fn fast_bpe_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = pyo3_log::try_init();
    log::debug!("initialized python logging bridge");
    module.add_class::<PyBPE>()?;
    log::info!("registered fast_bpe_rs.BPE python class");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensures the embedded Python interpreter is initialized before using PyO3 APIs.
    fn prepare_python() {
        Python::initialize();
    }

    #[test]
    fn module_registration_exports_bpe_class() {
        prepare_python();
        Python::attach(|py| -> PyResult<()> {
            let module = PyModule::new(py, "fast_bpe_rs")?;
            fast_bpe_rs(py, &module)?;

            let exported = module.getattr("BPE")?;
            assert_eq!(exported.getattr("__name__")?.extract::<&str>()?, "BPE");
            Ok(())
        })
        .expect("module should register the Python BPE class");
    }
}
