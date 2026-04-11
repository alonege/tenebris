use crate::tensor::tensor::{Tensor, TensorFloat};

pub mod sgd;

pub trait Optimizer {
    type HyperParams;
    fn update_tensor<T: TensorFloat, const D: usize>(
        &mut self,
        tensor: &mut Tensor<T, D>,
    ) -> Result<(), crate::error::LibError>;
    fn get_hyper_params(&self) -> &Self::HyperParams;
    fn set_hyper_params(&mut self, hyper_params: Self::HyperParams);
}
