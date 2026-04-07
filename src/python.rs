use crate::bpe::BPE;
use crate::types::TokenId;
use arrow::array::{Array, LargeStringArray, StringArray, make_array};
use arrow::datatypes::DataType;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use pyo3::exceptions::{PyUnicodeDecodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyBytes;
use std::collections::HashMap;

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

    fn train_from_arrow_array(
        &mut self,
        py: Python<'_>,
        vocab_size: TokenId,
        docs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut ffi_array = FFI_ArrowArray::empty();
        let mut ffi_schema = FFI_ArrowSchema::empty();
        docs.call_method1(
            "_export_to_c",
            (
                &mut ffi_array as *mut FFI_ArrowArray as usize,
                &mut ffi_schema as *mut FFI_ArrowSchema as usize,
            ),
        )?;
        let field = unsafe { from_ffi(ffi_array, &ffi_schema) }.map_err(|err| {
            PyValueError::new_err(format!("failed to import pyarrow string array: {err}"))
        })?;
        let array = make_array(field);

        match array.data_type() {
            DataType::Utf8 => {
                let Some(string_array) = array.as_any().downcast_ref::<StringArray>() else {
                    return Err(PyValueError::new_err(
                        "expected a pyarrow utf8 array for training docs",
                    ));
                };
                py.detach(|| self.train_from_docs(vocab_size, string_array.iter().flatten()));
                Ok(())
            }
            DataType::LargeUtf8 => {
                let Some(string_array) = array.as_any().downcast_ref::<LargeStringArray>() else {
                    return Err(PyValueError::new_err(
                        "expected a pyarrow large_utf8 array for training docs",
                    ));
                };
                py.detach(|| self.train_from_docs(vocab_size, string_array.iter().flatten()));
                Ok(())
            }
            other => Err(PyValueError::new_err(format!(
                "expected a pyarrow string array for training docs, got {other:?}",
            ))),
        }
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
    ///
    /// This API also accepts `pyarrow.Array` values (utf8 / large_utf8) and imports Arrow buffers
    /// through the Arrow C Data Interface.
    fn train(
        &mut self,
        py: Python<'_>,
        vocab_size: TokenId,
        docs: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if docs.hasattr("_export_to_c")? {
            return self.train_from_arrow_array(py, vocab_size, &docs);
        }

        let docs: Vec<PyBackedStr> = docs.extract()?;
        py.detach(|| self.train_from_docs(vocab_size, docs.iter().map(PyBackedStr::as_str)));
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
