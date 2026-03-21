use crate::bpe::BPE;
use crate::types::TokenId;
use pyo3::exceptions::{PyUnicodeDecodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(name = "BPE")]
pub struct PyBPE {
    inner: BPE,
}

#[pymethods]
impl PyBPE {
    #[new]
    #[pyo3(signature = (split_pattern))]
    fn py_new(split_pattern: &str) -> PyResult<Self> {
        let inner = BPE::try_new(split_pattern)
            .map_err(|err| PyValueError::new_err(format!("invalid split regex: {err}")))?;
        Ok(Self { inner })
    }

    fn train(&mut self, vocab_size: TokenId, docs: Vec<String>) {
        self.inner
            .train(vocab_size, docs.iter().map(String::as_str));
    }

    fn encode(&self, doc: &str) -> Vec<TokenId> {
        self.inner.encode(doc)
    }

    fn decode<'py>(&self, py: Python<'py>, token_ids: Vec<TokenId>) -> Bound<'py, PyBytes> {
        let bytes = self.inner.decode(token_ids);
        PyBytes::new(py, &bytes)
    }

    fn decode_to_string<'py>(
        &self,
        py: Python<'py>,
        token_ids: Vec<TokenId>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bytes = self.inner.decode(token_ids);
        match String::from_utf8(bytes) {
            Ok(text) => Ok(text.into_pyobject(py)?.into_any()),
            Err(err) => {
                let utf8_err = err.utf8_error();
                Err(PyUnicodeDecodeError::new_utf8(py, err.as_bytes(), utf8_err)?.into())
            }
        }
    }
}
