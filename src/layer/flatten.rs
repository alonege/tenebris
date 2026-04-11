use crate::{
    error::LibError,
    layer::{Layer, Tensor, TensorFloat},
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

impl<T: TensorFloat, const D: usize> Layer<Tensor<T, D>> for Flatten<T> {
    type Output = Tensor<T, 2>;

    fn forward(&mut self, input: Tensor<T, D>, save_grads: bool) -> Result<Self::Output, LibError> {
        if save_grads {
            self.input_shape_cache = Some(input.shape().to_vec());
        }
        let n = *input.shape().last().unwrap_or(&1);
        let features = input.shape().iter().product::<usize>() / n;
        input.reshape([features, n])
    }

    fn backward(&mut self, grad_output: Self::Output) -> Result<Tensor<T, D>, LibError> {
        let original_shape_vec = self
            .input_shape_cache
            .as_ref()
            .ok_or_else(|| LibError::LayerErrorBackwardNoGradient)?;

        let mut original_shape = [0usize; D];
        original_shape.copy_from_slice(original_shape_vec);

        grad_output.reshape(original_shape)
    }

    fn visit_params<O: crate::optimizer::Optimizer>(&mut self, _optimizer: &mut O) {
        // No parameters to update
    }

    fn clear_grad(&mut self) {
        self.input_shape_cache = None;
    }
}
