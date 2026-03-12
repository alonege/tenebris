use thiserror::Error;

/// The main error type for the `libmlrs` library.
///
/// This enum represents all possible errors that can occur during
/// tensor and layer operations within the library.
#[derive(Error, Debug)]
pub enum LibError {
    /// Error representing a mismatch in tensor shapes for an operation.
    #[error("Shape mismatch: expected {expected:?}, but got {actual:?}. Operation: {operation}.")]
    ShapeMismatch {
        operation: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Error when the data length does not match the expected length for a given shape.
    #[error(
        "Invalid data for new tensor: shape {shape:?} requires {expected_len} elements, but provided data has {actual_len} elements."
    )]
    InvalidShapeForData {
        shape: Vec<usize>,
        expected_len: usize,
        actual_len: usize,
    },

    /// Error for when an index is out of the tensor's bounds.
    #[error("Index out of bounds: index {index:?} is invalid for shape {shape:?}.")]
    OutOfBounds {
        shape: Vec<usize>,
        index: Vec<usize>,
    },

    /// Error when a mutable operation is attempted on a shared tensor.
    #[error(
        "A mutable operation was attempted on a tensor with multiple owners (strong_count > 1)."
    )]
    SharedTensorMutability,

    /// Error for operations that require a specific number of dimensions for two tensors.
    #[error(
        "Invalid dimensionality for two tensors: operation '{operation}' requires 
        ({expected_a}, {expected_b}) dimensions, but tensors have ({actual_a}, {actual_b})."
    )]
    InvalidDimensionalityTwoTensors {
        operation: String,
        expected_a: usize,
        actual_a: usize,
        expected_b: usize,
        actual_b: usize,
    },

    /// Error for operations that require a specific number of dimensions.
    #[error(
        "Invalid dimensionality: operation '{operation}' requires {expected} dimensions, but tensor has {actual}."
    )]
    InvalidDimensionality {
        operation: String,
        expected: usize,
        actual: usize,
    },

    #[error("Unimplemented feature: {0}")]
    Unimplemented(String),

    #[error("Shape Error: {0}")]
    ShapeError(String),

    #[error("Other error: {0}")]
    OtherError(String),

    #[error("Layer Error: No saved input info found during backward pass")]
    LayerErrorBackwardNoGradient,

    #[error("ModuleData conversion failed: {0}")]
    ModuleDataConversionFailed(String),

    /// A generic error for wrapping issues from external libraries or for internal invariants.
    /// Using `anyhow::Error` allows for flexible error chaining with context.
    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, LibError>;
