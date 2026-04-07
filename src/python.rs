use crate::bpe::BPE;
use crate::types::TokenId;
use pyo3::exceptions::{PyUnicodeDecodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyBytes;
use std::collections::HashMap;

#[cfg(feature = "arrow")]
use arrow::array::{Array, ArrayRef, AsArray, make_array};
#[cfg(feature = "arrow")]
use arrow::datatypes::DataType;
#[cfg(feature = "arrow")]
use arrow::pyarrow::FromPyArrow;

/// Thin PyO3 wrapper that exposes [`BPE`] to Python callers.
#[pyclass(name = "BPE")]
pub struct PyBPE {
    /// Pure-Rust implementation that does the actual work.
    inner: BPE,
}

impl PyBPE {
    fn train_from_docs<'a>(&mut self, vocab_size: TokenId, docs: impl Iterator<Item = &'a str>) {
        self.inner.train(vocab_size, docs);
    }
}

#[pymethods]
impl PyBPE {
    /// Creates a Python-facing BPE model and translates construction errors into `ValueError`.
    #[new]
    #[pyo3(signature = (split_pattern, special_tokens=None))]
    fn py_new(
        split_pattern: &str,
        special_tokens: Option<HashMap<String, TokenId>>,
    ) -> PyResult<Self> {
        let inner = BPE::new(Some(split_pattern), special_tokens)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Trains the model in place from Python strings without copying each document into Rust-owned `String`s.
    ///
    /// `PyBackedStr` keeps references to Python-owned immutable `str` objects so the Rust trainer can
    /// read the corpus directly, then `detach` releases the GIL while heavy work runs in Rust.
    fn train(&mut self, py: Python<'_>, vocab_size: TokenId, docs: Vec<PyBackedStr>) {
        py.detach(|| self.train_from_docs(vocab_size, docs.iter().map(PyBackedStr::as_str)));
    }

    /// Trains the model from a PyArrow `Array` or `ChunkedArray` of strings.
    ///
    /// Arrow buffers are accessed with zero copy — gigabytes of text pass from
    /// Python to Rust without serialisation or duplication.
    #[cfg(feature = "arrow")]
    fn train_arrow(
        &mut self,
        py: Python<'_>,
        vocab_size: TokenId,
        arrow_data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Extract ArrayRef(s) from PyArrow Array or ChunkedArray.
        let arrays: Vec<ArrayRef> = if arrow_data.hasattr("chunks")? {
            let py_chunks = arrow_data.getattr("chunks")?;
            let n: usize = py_chunks.len()?;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let data = arrow::array::ArrayData::from_pyarrow_bound(&py_chunks.get_item(i)?)?;
                v.push(make_array(data));
            }
            v
        } else {
            let data = arrow::array::ArrayData::from_pyarrow_bound(arrow_data)?;
            vec![make_array(data)]
        };

        // Validate data types BEFORE dropping the GIL, so we can return a Python exception.
        for array in &arrays {
            match array.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {}
                other => {
                    return Err(PyValueError::new_err(format!(
                        "expected a string array, got {other}"
                    )));
                }
            }
        }

        // Move the `ArrayRef`s into the detached thread and build a lazy iterator.
        py.detach(move || {
            let docs = arrays
                .iter()
                .flat_map(|array| -> Box<dyn Iterator<Item = &str>> {
                    // .iter() yields Option<&str>, and .flatten() seamlessly skips the nulls.
                    let non_null: Box<dyn Iterator<Item = &str>> = match array.data_type() {
                        DataType::Utf8 => Box::new(array.as_string::<i32>().iter().flatten()),
                        DataType::LargeUtf8 => Box::new(array.as_string::<i64>().iter().flatten()),
                        DataType::Utf8View => Box::new(array.as_string_view().iter().flatten()),
                        _ => unreachable!("Array types were pre-validated"),
                    };
                    Box::new(non_null.filter(|s| !s.trim().is_empty()))
                });

            self.inner.train(vocab_size, docs);
        });

        Ok(())
    }

    /// Encodes a string into token ids.
    fn encode(&self, doc: &str) -> Vec<TokenId> {
        self.inner.encode(doc)
    }

    /// Decodes token ids into raw Python `bytes`.
    fn decode<'py>(&self, py: Python<'py>, token_ids: Vec<TokenId>) -> Bound<'py, PyBytes> {
        let bytes = self.inner.decode(token_ids);
        PyBytes::new(py, &bytes)
    }

    /// Decodes token ids into a Python `str`, surfacing UTF-8 errors when bytes are invalid text.
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyString;

    /// Ensures the embedded Python interpreter is initialized before using PyO3 APIs.
    fn prepare_python() {
        Python::initialize();
    }

    #[test]
    fn py_new_returns_value_error_for_invalid_regex() {
        let err = match PyBPE::py_new("(", None) {
            Ok(_) => panic!("invalid regex should fail"),
            Err(err) => err,
        };
        prepare_python();
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(
                err.to_string().contains("invalid split regex"),
                "unexpected error message: {err}"
            );
        });
    }

    #[test]
    fn py_new_returns_value_error_for_invalid_special_token_ids() {
        let err = match PyBPE::py_new("(?s).+", Some(HashMap::from([("<pad>".to_owned(), 42)]))) {
            Ok(_) => panic!("special token ids in the base vocabulary should fail"),
            Err(err) => err,
        };
        prepare_python();
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("must be >= 256"));
        });
    }

    #[test]
    fn python_wrapper_trains_encodes_and_decodes() {
        let mut bpe = PyBPE::py_new("(?s).+", None).expect("valid regex should construct");
        bpe.train_from_docs(257, ["abab"].into_iter());

        assert_eq!(bpe.encode("abab"), vec![256, 256]);

        prepare_python();
        Python::attach(|py| {
            let decoded = bpe.decode(py, vec![256, 256]);
            assert_eq!(decoded.as_bytes(), b"abab");

            let text = bpe
                .decode_to_string(py, vec![256, 256])
                .expect("valid utf-8 should decode to string");
            let py_str = text
                .cast::<PyString>()
                .expect("decoded text should be a string");
            assert_eq!(
                py_str.to_str().expect("python string should be utf-8"),
                "abab"
            );
        });
    }

    #[test]
    fn python_wrapper_supports_special_tokens() {
        let mut bpe = PyBPE::py_new(
            "(?s).+",
            Some(HashMap::from([
                ("<pad>".to_owned(), 600),
                ("<eos>".to_owned(), 601),
            ])),
        )
        .expect("valid regex should construct");
        bpe.train_from_docs(605, ["a<pad>a"].into_iter());

        assert_eq!(
            bpe.encode("a<pad><eos>a"),
            vec![b'a' as u32, 600, 601, b'a' as u32]
        );

        prepare_python();
        Python::attach(|py| {
            let decoded = bpe.decode(py, vec![600, 601]);
            assert_eq!(decoded.as_bytes(), b"<pad><eos>");
        });
    }

    #[test]
    fn decode_to_string_surfaces_unicode_decode_errors() {
        let mut bpe = PyBPE::py_new("(?s).+", None).expect("valid regex should construct");
        bpe.train_from_docs(257, ["éa"].into_iter());

        prepare_python();
        Python::attach(|py| {
            let err = bpe
                .decode_to_string(py, vec![256, b'a' as u32])
                .expect_err("invalid utf-8 should raise");
            assert!(err.is_instance_of::<PyUnicodeDecodeError>(py));
        });
    }
}
