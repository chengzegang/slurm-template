mod _os;
use pyo3::prelude::*;
use pyo3::types::PyList;
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn listdir(
    py: Python,
    root: &str,
    pattern: Option<&str>,
    recursive: Option<bool>,
) -> PyResult<PyObject> {
    let result = _os::listdir_rs(root, pattern, recursive);
    let py_list = PyList::new(py, &result);
    Ok(py_list.to_object(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(listdir, m)?)?;
    Ok(())
}
