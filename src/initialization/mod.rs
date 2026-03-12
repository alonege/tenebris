use crate::{
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub trait Initialization<T: TensorFloat> {
    /// Primary method: initialize raw data
    /// Most efficient, no shape needed for simple initializers
    fn initialize_data(
        &self,
        fan_in: usize,
        fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, LibError>;

    /// Optional override for shape-aware initialization
    /// Default implementation uses initialize_data
    fn initialize_tensor(
        &self,
        fan_in: usize,
        fan_out: usize,
        shape: Vec<usize>,
    ) -> Result<Tensor<T>, LibError> {
        let n_elements = shape.iter().product();
        let data = self.initialize_data(fan_in, fan_out, n_elements)?;
        Tensor::new(shape, data)
    }
}

pub mod glorot;
pub mod he;
pub mod lecun;
pub mod normal;
pub mod uniform;
