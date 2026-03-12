use crate::{
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub mod activation;
pub mod chain;
pub mod conv2d;
pub mod dropout;
pub mod flatten;
pub mod linear;
pub mod softmax;

/// Primary, user facing trait for modules, using Generic Associated Types for comptime type safety
/// Not `dyn``-safe!
pub trait Module<T: TensorFloat> {
    type Input<A>;
    type Output<B>;
    fn forward(
        &mut self,
        input: Self::Input<T>,
        save_grads: bool,
    ) -> Result<Self::Output<T>, LibError>;
    fn backward(&mut self, y_out: Self::Input<T>) -> Result<Self::Output<T>, LibError>;
    fn parameters(&self) -> Vec<crate::tensor::tensor::Tensor<T>>;
    fn parameters_mut(&mut self) -> Vec<&mut crate::tensor::tensor::Tensor<T>>;
    fn clear_grad(&mut self);
    fn training(&mut self) {}
    fn inference(&mut self) {}
}

/// Struct for Tensor flow between `dyn` Modules in `Vec` or `dyn` using layers, e.g. `Chain`.
pub enum ModuleData<T: TensorFloat> {
    Single(Tensor<T>),
    Multi(Vec<Tensor<T>>),
}

/// `dyn`-safe trait for dynamic dispatch in container Modules, eg. `Chain`.
pub trait ModuleDyn<T: TensorFloat> {
    fn forward_dyn(
        &mut self,
        input: ModuleData<T>,
        save_grads: bool,
    ) -> Result<ModuleData<T>, LibError>;
    fn backward_dyn(&mut self, y_out: ModuleData<T>) -> Result<ModuleData<T>, LibError>;
    fn parameters_dyn(&self) -> Vec<crate::tensor::tensor::Tensor<T>>;
    fn parameters_mut_dyn(&mut self) -> Vec<&mut crate::tensor::tensor::Tensor<T>>;
    fn clear_grad_dyn(&mut self);
    fn training_dyn(&mut self) {}
    fn inference_dyn(&mut self) {}
}

pub trait IntoModuleData<T: TensorFloat> {
    fn into_module_data(self) -> ModuleData<T>;
}

pub trait FromModuleData<T: TensorFloat>: Sized {
    fn from_module_data(data: ModuleData<T>) -> Result<Self, LibError>;
}

impl<T: TensorFloat> IntoModuleData<T> for Tensor<T> {
    fn into_module_data(self) -> ModuleData<T> {
        ModuleData::Single(self)
    }
}

impl<T: TensorFloat> FromModuleData<T> for Tensor<T> {
    fn from_module_data(data: ModuleData<T>) -> Result<Self, LibError> {
        match data {
            ModuleData::Single(t) => Ok(t),
            _ => Err(LibError::ModuleDataConversionFailed(
                "ModuleDyn: ModuleData::Multi(Vec<Tensor<T>>) can't be changed into Tensor<T>"
                    .to_string(),
            )),
        }
    }
}

impl<T: TensorFloat> IntoModuleData<T> for Vec<Tensor<T>> {
    fn into_module_data(self) -> ModuleData<T> {
        ModuleData::Multi(self)
    }
}
impl<T: TensorFloat> FromModuleData<T> for Vec<Tensor<T>> {
    fn from_module_data(data: ModuleData<T>) -> Result<Self, LibError> {
        match data {
            ModuleData::Multi(v) => Ok(v),
            _ => Err(LibError::ModuleDataConversionFailed(
                "ModuleDyn: ModuleData::Single(Tensor<T>) can't be changed into Vec<Tensor<T>>"
                    .to_string(),
            )),
        }
    }
}

impl<T: TensorFloat> IntoModuleData<T> for ModuleData<T> {
    fn into_module_data(self) -> ModuleData<T> {
        self
    }
}

impl<T: TensorFloat> FromModuleData<T> for ModuleData<T> {
    fn from_module_data(data: ModuleData<T>) -> Result<Self, LibError> {
        Ok(data)
    }
}

impl<T, M> ModuleDyn<T> for M
where
    T: TensorFloat,
    M: Module<T>,
    M::Input<T>: FromModuleData<T>,
    M::Output<T>: IntoModuleData<T>,
{
    fn forward_dyn(
        &mut self,
        input: ModuleData<T>,
        save_grad: bool,
    ) -> Result<ModuleData<T>, LibError> {
        let concrete_input = M::Input::from_module_data(input)?;

        let concrete_output = self.forward(concrete_input, save_grad)?;

        Ok(concrete_output.into_module_data())
    }

    fn backward_dyn(&mut self, y_out: ModuleData<T>) -> Result<ModuleData<T>, LibError> {
        let concrete_grad_output = M::Input::from_module_data(y_out)?;

        let concrete_downstream_grad = self.backward(concrete_grad_output)?;

        Ok(concrete_downstream_grad.into_module_data())
    }

    fn parameters_dyn(&self) -> Vec<crate::tensor::tensor::Tensor<T>> {
        self.parameters()
    }

    fn parameters_mut_dyn(&mut self) -> Vec<&mut crate::tensor::tensor::Tensor<T>> {
        self.parameters_mut()
    }

    fn clear_grad_dyn(&mut self) {
        self.clear_grad();
    }

    fn training_dyn(&mut self) {
        self.training();
    }

    fn inference_dyn(&mut self) {
        self.inference();
    }
}
