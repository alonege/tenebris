use std::{any::Any, collections::HashMap};

use crate::{
    error::{self, LibError},
    optimizer::Optimizer,
    tensor::{
        tensor::{Tensor, TensorFloat},
        tensorops::TensorAdd,
    },
};

pub struct SGDHyperParams {
    pub learning_rate: f64,
}

pub struct SGD {
    hyper_params: SGDHyperParams,
}

impl SGD {
    pub fn new(hyper_params: SGDHyperParams) -> Self {
        Self { hyper_params }
    }
}

impl Optimizer for SGD {
    type HyperParams = SGDHyperParams;

    fn update_tensor<T: TensorFloat, const D: usize>(
        &mut self,
        tensor: &mut Tensor<T, D>,
    ) -> Result<(), LibError> {
        let grad = tensor.grad.take();
        let mut grad = match grad {
            Some(g) => *g,
            None => return Err(error::LibError::LayerErrorBackwardNoGradient),
        };
        grad.map_inplace(|v| -v * T::from(self.hyper_params.learning_rate).unwrap());
        let updated_value = tensor.tensoradd(&grad);
        let updated_value = match updated_value {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        tensor.change_tensor(updated_value);
        Ok(())
    }

    #[inline(always)]
    fn get_hyper_params(&self) -> &Self::HyperParams {
        &self.hyper_params
    }

    #[inline(always)]
    fn set_hyper_params(&mut self, hyper_params: Self::HyperParams) {
        self.hyper_params = hyper_params;
    }
}

pub struct SGDWithMomentumHyperParams {
    pub learning_rate: f64,
    pub momentum: f64,
    pub decay: f64,
}

pub struct SGDWithMomentum {
    hyper_params: SGDWithMomentumHyperParams,
    velocity: HashMap<usize, Box<dyn Any>>,
}

impl SGDWithMomentum {
    pub fn new(hyper_params: SGDWithMomentumHyperParams) -> Self {
        Self {
            hyper_params,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGDWithMomentum {
    type HyperParams = SGDWithMomentumHyperParams;

    #[inline(always)]
    fn update_tensor<T: TensorFloat, const D: usize>(
        &mut self,
        tensor: &mut Tensor<T, D>,
    ) -> Result<(), LibError> {
        if let Some(grad) = &tensor.grad {
            let i = tensor.get_id(); // Zgodnie z nowym nazewnictwem: tensor.id

            // 1. Zapisujemy/Pobieramy "czarną skrzynkę" z HashMapy.
            // Zauważ rzutowanie `as Box<dyn Any>` przy tworzeniu!
            let velocity_any = self.velocity.entry(i).or_insert_with(|| {
                Box::new(Tensor::<T, D>::zeros(*tensor.shape()).unwrap()) as Box<dyn Any>
            });

            // 2. MAGIA RUSTA: Odzyskujemy typ Tensor<T, D> z czarnej skrzynki (Downcasting)!
            let velocity = velocity_any
                .downcast_mut::<Tensor<T, D>>()
                .expect("Krytyczny błąd optymalizatora: Niezgodność typu w cache Momentum!");

            // 3. Konwersje hyperparametrów z f64 do generycznego T
            let momentum = T::from(self.hyper_params.momentum).unwrap();
            let lr = T::from(self.hyper_params.learning_rate).unwrap();
            let decay = T::from(self.hyper_params.decay).unwrap();
            let one = T::one();

            // v = momentum * v + grad
            *velocity = velocity.map(|x| *x * momentum).tensoradd(grad).unwrap();

            // tensor -= lr * v
            // Najpierw wyliczamy współczynnik wygaszania wag (weight decay factor)
            let decay_factor = one - (decay * lr);

            // Wyliczamy przesunięcie z momentum
            let velocity_step = velocity.map(|x| *x * lr);

            // Aplikujemy to do tensora!
            *tensor = tensor
                .map(|x| *x * decay_factor)
                .sub(&velocity_step)
                .unwrap();
        }

        Ok(())
    }

    #[inline(always)]
    fn get_hyper_params(&self) -> &Self::HyperParams {
        &self.hyper_params
    }
    #[inline(always)]
    fn set_hyper_params(&mut self, hyper_params: Self::HyperParams) {
        self.hyper_params = hyper_params;
    }
}
