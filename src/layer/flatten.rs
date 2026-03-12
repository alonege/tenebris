use crate::{
    error::LibError,
    layer::{Module, Tensor, TensorFloat},
};
use std::marker::PhantomData;

/// A layer that flattens the input.
/// It remembers original shape for the backward pass (if save_grads is true).
pub struct Flatten<T: TensorFloat> {
    input_shape_cache: Option<Vec<usize>>,
    _marker: PhantomData<T>,
}

impl<T: TensorFloat> Flatten<T> {
    pub fn new() -> Self {
        Self {
            input_shape_cache: None,
            _marker: PhantomData,
        }
    }
}

impl<T: TensorFloat> Module<T> for Flatten<T> {
    type Input<A> = Tensor<T>;
    type Output<B> = Tensor<T>;

    fn forward(
        &mut self,
        input: Self::Input<T>,
        save_grads: bool,
    ) -> Result<Self::Output<T>, LibError> {
        if save_grads {
            self.input_shape_cache = Some(input.shape().to_vec());
        }
        let n = *input.shape().last().unwrap_or(&1);
        let features = input.shape().iter().product::<usize>() / n;
        input.reshape(vec![features, n])
    }

    fn backward(&mut self, grad_output: Self::Input<T>) -> Result<Self::Output<T>, LibError> {
        let original_shape = self
            .input_shape_cache
            .as_ref()
            .ok_or_else(|| LibError::LayerErrorBackwardNoGradient)?;

        grad_output.reshape(original_shape.clone())
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn clear_grad(&mut self) {
        self.input_shape_cache = None;
    }
}
