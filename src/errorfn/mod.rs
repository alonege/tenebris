use crate::{
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub trait ErrorFn<T: TensorFloat> {
    fn compute(&self, y_pred: &Tensor<T>, y_true: &Tensor<T>) -> Result<(T, Tensor<T>), LibError>;
    fn loss(&self, y_pred: &Tensor<T>, y_true: &Tensor<T>) -> Result<T, LibError> {
        Ok(self.compute(y_pred, y_true)?.0)
    }
    fn grad(&self, y_pred: &Tensor<T>, y_true: &Tensor<T>) -> Result<Tensor<T>, LibError> {
        Ok(self.compute(y_pred, y_true)?.1)
    }
}

pub mod mse;
pub mod softmax_with_crossentropy;
