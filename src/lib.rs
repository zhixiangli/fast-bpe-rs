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
mod tests {
    use super::*;

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
