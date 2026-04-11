use crate::{
    error::LibError,
    optimizer::Optimizer,
    tensor::tensor::{Tensor, TensorFloat},
};

pub mod activation;
pub mod chain;
pub mod conv2d;
pub mod dropout;
pub mod flatten;
pub mod linear;

/// Primary, user facing trait for Layer, using Generic Associated Types for comptime type safety
/// Not `dyn``-safe!
pub trait Layer<Input> {
    type Output;
    fn forward(&mut self, input: Input, save_grads: bool) -> Result<Self::Output, LibError>;
    fn backward(&mut self, y_out: Self::Output) -> Result<Input, LibError>;
    fn visit_params<O: Optimizer>(&mut self, optimizer: &mut O);
    fn clear_grad(&mut self);
    fn training(&mut self) {}
    fn inference(&mut self) {}
}
