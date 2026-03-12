use crate::{error::LibError, layer::Module, tensor::tensor::TensorFloat};

pub mod sgd;

pub trait Optimizer<T: TensorFloat> {
    fn step(&mut self, module: &mut impl Module<T>) -> Result<(), LibError>;
    fn get_learning_rate(&self) -> T;
    fn set_learning_rate(&mut self, lr: T);
}
