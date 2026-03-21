mod bpe;
mod chain;
mod python;
mod types;

pub use crate::bpe::BPE;
pub use crate::python::PyBPE;

use pyo3::prelude::*;

#[pymodule]
fn fast_bpe_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBPE>()?;
    Ok(())
}

#[cfg(test)]
mod tests;
