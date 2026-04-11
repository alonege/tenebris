use crate::tensor::{
    tensor::{Tensor, TensorFloat},
    tensorops::TensorAdd,
};

pub struct Mse;

impl Mse {
    pub fn new() -> Self {
        Mse
    }
}

impl<T: TensorFloat + std::iter::Sum, const D: usize> crate::errorfn::ErrorFn<T, D> for Mse {
    fn compute(
        &self,
        y_pred: &Tensor<T, D>,
        y_true: &Tensor<T, D>,
    ) -> Result<(T, Tensor<T, D>), crate::error::LibError> {
        let diff = y_pred.tensoradd(&y_true.map(|y| -*y))?;
        let n = T::from(y_pred.shape()[0]).unwrap(); // Liczba próbek

        // Loss = (1/N) * sum((y_pred - y_true)^2)
        let loss = diff.iter().map(|x| x.powi(2)).sum::<T>() / n;

        // Grad = (2/N) * (y_pred - y_true)
        let grad = diff.map(|x| T::from(2.0).unwrap() * *x / n);

        Ok((loss, grad))
    }
}
