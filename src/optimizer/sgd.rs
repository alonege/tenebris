use std::collections::HashMap;

use crate::{
    error::{self, LibError},
    layer::Module,
    optimizer::Optimizer,
    tensor::{
        tensor::{Tensor, TensorFloat},
        tensorops::TensorAdd,
    },
};

pub struct SGD<T> {
    pub learning_rate: T,
}

impl<T> SGD<T> {
    pub fn new(lr: T) -> Self {
        Self { learning_rate: lr }
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: TensorFloat,
{
    fn step(&mut self, module: &mut impl Module<T>) -> Result<(), LibError> {
        for param in module.parameters_mut() {
            let grad = param.grad.take();
            let mut grad = match grad {
                Some(g) => *g,
                None => return Err(error::LibError::LayerErrorBackwardNoGradient),
            };
            grad.map_inplace(|v| -v * self.learning_rate);
            let updated_value = param.tensoradd(&grad);
            let updated_value = match updated_value {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            param.change_tensor(updated_value);
        }

        module.clear_grad();
        Ok(())
    }

    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }
}

pub struct SGDWithMomentum<T: TensorFloat> {
    lr: T,
    momentum: T,
    decay: T,
    velocity: HashMap<usize, Tensor<T>>,
}

impl<T: TensorFloat> SGDWithMomentum<T> {
    pub fn new(lr: T, momentum: T, decay: T) -> Self {
        Self {
            lr,
            momentum,
            decay,
            velocity: HashMap::new(),
        }
    }
}

impl<T: TensorFloat> Optimizer<T> for SGDWithMomentum<T> {
    #[inline(always)]
    fn step(&mut self, model: &mut impl Module<T>) -> Result<(), LibError> {
        let params = model.parameters_mut();

        for (i, param) in params.into_iter().enumerate() {
            if let Some(grad) = &param.grad {
                // get/create velocity
                let velocity = self
                    .velocity
                    .entry(i)
                    .or_insert_with(|| Tensor::zeros(param.shape().to_vec()).unwrap());

                // v = momentum * v + grad
                *velocity = velocity
                    .map(|x| *x * self.momentum)
                    .tensoradd(grad)
                    .unwrap();

                // param -= lr * v
                *param = param
                    .map(|x| *x * (T::one() - self.decay * self.lr))
                    .sub(&velocity.map(|x| *x * self.lr))
                    .unwrap();
            }
        }

        model.clear_grad();
        Ok(())
    }

    #[inline(always)]
    fn get_learning_rate(&self) -> T {
        self.lr
    }
    #[inline(always)]
    fn set_learning_rate(&mut self, lr: T) {
        self.lr = lr;
    }
}
