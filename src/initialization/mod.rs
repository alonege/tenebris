use crate::{
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub trait Initialization<T: TensorFloat>: Sized {
    fn initialize_data(
        &self,
        fan_in: usize,
        fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, LibError>;
    fn initialize_tensor<const D: usize>(
        &self,
        fan_in: usize,
        fan_out: usize,
        shape: [usize; D],
    ) -> Result<Tensor<T, D>, LibError> {
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
