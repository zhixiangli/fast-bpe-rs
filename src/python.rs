use crate::bpe::BPE;
use crate::types::TokenId;
use pyo3::exceptions::{PyUnicodeDecodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

#[pyclass(name = "BPE")]
pub struct PyBPE {
    inner: BPE,
}

#[pymethods]
impl PyBPE {
    #[new]
    #[pyo3(signature = (split_pattern, special_tokens=None))]
    fn py_new(
        split_pattern: &str,
        special_tokens: Option<HashMap<String, TokenId>>,
    ) -> PyResult<Self> {
        let inner =
            BPE::try_new_with_special_tokens(split_pattern, special_tokens.unwrap_or_default())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyString;

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
        bpe.train(257, vec!["abab".to_owned()]);

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
        bpe.train(605, vec!["a<pad>a".to_owned()]);

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
        bpe.train(257, vec!["éa".to_owned()]);

        prepare_python();
        Python::attach(|py| {
            let err = bpe
                .decode_to_string(py, vec![256, b'a' as u32])
                .expect_err("invalid utf-8 should raise");
            assert!(err.is_instance_of::<PyUnicodeDecodeError>(py));
        });
    }
}
