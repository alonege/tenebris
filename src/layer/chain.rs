use crate::{error::LibError, layer::Layer};

// Abandon hope, all ye who enter here

pub struct Chain<L1, L2> {
    layer1: L1,
    layer2: L2,
}

// Trait which adds the `.add()` method to anything that implements `Layer`
pub trait ChainBuilder<Input>: Layer<Input> + Sized {
    /// Connects the current layer (or chain) with a new layer, creating a new, static type.
    fn add<L2>(self, next_layer: L2) -> Chain<Self, L2>
    where
        L2: Layer<Self::Output>,
    {
        Chain {
            layer1: self,
            layer2: next_layer,
        }
    }
}

impl<L1, L2, Input> Layer<Input> for Chain<L1, L2>
where
    L1: Layer<Input>,      // L1 przyjmuje początkowy Input
    L2: Layer<L1::Output>, // L2 MUSI przyjmować Output z L1!
{
    type Output = L2::Output;

    #[inline(always)]
    fn forward(&mut self, input: Input, save_grads: bool) -> Result<Self::Output, LibError> {
        let out1 = self.layer1.forward(input, save_grads)?;
        let out = self.layer2.forward(out1, save_grads);
        out
    }

    #[inline(always)]
    fn backward(&mut self, grad_output: Self::Output) -> Result<Input, LibError> {
        let grad1 = self.layer2.backward(grad_output)?;
        self.layer1.backward(grad1)
    }

    #[inline(always)]
    fn visit_params<O: crate::optimizer::Optimizer>(&mut self, optimizer: &mut O) {
        self.layer1.visit_params(optimizer);
        self.layer2.visit_params(optimizer);
    }

    #[inline(always)]
    fn clear_grad(&mut self) {
        self.layer1.clear_grad();
        self.layer2.clear_grad();
    }
}

// auto implementation for any Layer, so we can chain as many as we want without worrying about the
// types
impl<L, Input> ChainBuilder<Input> for L where L: Layer<Input> {}

#[allow(unused_imports)]
pub mod tests {

    use std::vec;

    use crate::{
        layer::{self, chain::Chain, linear::Linear},
        tensor::tensor::Tensor,
    };
}
