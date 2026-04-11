use crate::{
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub trait ErrorFn<T: TensorFloat, const D: usize> {
    fn compute(
        &self,
        y_pred: &Tensor<T, D>,
        y_true: &Tensor<T, D>,
    ) -> Result<(T, Tensor<T, D>), LibError>;
    fn loss(&self, y_pred: &Tensor<T, D>, y_true: &Tensor<T, D>) -> Result<T, LibError> {
        Ok(self.compute(y_pred, y_true)?.0)
    }
    fn grad(&self, y_pred: &Tensor<T, D>, y_true: &Tensor<T, D>) -> Result<Tensor<T, D>, LibError> {
        Ok(self.compute(y_pred, y_true)?.1)
    }
}

pub mod mse;
pub mod softmax_with_crossentropy;
