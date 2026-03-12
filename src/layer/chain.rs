use crate::{
    error::LibError,
    layer::{FromModuleData, IntoModuleData, Module, ModuleData, ModuleDyn},
    tensor::tensor::TensorFloat,
};

// Abandon hope, all ye who enter here

pub struct Chain<T> {
    modules: Vec<Box<dyn ModuleDyn<T>>>,
}

impl<T: TensorFloat> Chain<T> {
    pub fn new(modules: Vec<Box<dyn ModuleDyn<T>>>) -> Self {
        Self { modules }
    }

    pub fn forward<In, Out>(&mut self, input: In, save_grads: bool) -> Result<Out, LibError>
    where
        In: IntoModuleData<T>,
        Out: FromModuleData<T>,
    {
        let initial_data: ModuleData<T> = input.into_module_data();

        //println!("CHAIN: ");
        let final_data = self
            .modules
            .iter_mut()
            .try_fold(initial_data, |data, module| {
                //let dur = Instant::now();
                let out = module.forward_dyn(data, save_grads);
                //println!("TIME: {:?}", dur.elapsed());
                out
            });

        Out::from_module_data(final_data?)
    }
    pub fn backward_internal<In, Out>(&mut self, grad_output: In) -> Result<Out, LibError>
    where
        In: IntoModuleData<T>,
        Out: FromModuleData<T>,
    {
        let initial_grad: ModuleData<T> = grad_output.into_module_data();

        let final_grad = self
            .modules
            .iter_mut()
            .rev()
            .try_fold(initial_grad, |data, module| module.backward_dyn(data));

        Out::from_module_data(final_grad?)
    }
}

impl<T: TensorFloat> Module<T> for Chain<T> {
    type Input<A> = ModuleData<T>;
    type Output<B> = ModuleData<T>;

    #[inline(always)]
    fn forward(
        &mut self,
        input: Self::Input<T>,
        save_grads: bool,
    ) -> Result<Self::Output<T>, crate::error::LibError> {
        self.forward(input, save_grads)
    }

    #[inline(always)]
    fn backward(&mut self, grad_output: Self::Input<T>) -> Result<Self::Output<T>, LibError> {
        self.backward_internal(grad_output)
    }

    #[inline(always)]
    fn parameters(&self) -> Vec<crate::tensor::tensor::Tensor<T>> {
        self.modules
            .iter()
            .flat_map(|module| module.parameters_dyn())
            .collect()
    }

    #[inline(always)]
    fn parameters_mut(&mut self) -> Vec<&mut crate::tensor::tensor::Tensor<T>> {
        self.modules
            .iter_mut()
            .flat_map(|module| module.parameters_mut_dyn())
            .collect()
    }

    #[inline(always)]
    fn clear_grad(&mut self) {
        self.modules
            .iter_mut()
            .for_each(|module| module.clear_grad_dyn());
    }

    #[inline(always)]
    fn training(&mut self) {
        self.modules
            .iter_mut()
            .for_each(|module| module.training_dyn());
    }

    #[inline(always)]
    fn inference(&mut self) {
        self.modules
            .iter_mut()
            .for_each(|module| module.inference_dyn());
    }
}

#[allow(unused_imports)]
pub mod tests {

    use std::vec;

    use crate::layer::FromModuleData;
    use crate::{
        layer::{self, IntoModuleData, Module, ModuleDyn, chain::Chain, linear::Linear},
        tensor::tensor::Tensor,
    };

    // TODO: FIX IT
    #[allow(unused)]
    #[test]
    fn two_linear() {
        let a: Box<dyn ModuleDyn<f32>> = Box::new(Linear::new_rand(4, 3));
        let b = Box::new(Linear::new_rand(3, 5));
        let modules = vec![a, b];
        let mut sequence = Chain::new(modules);
        let input = Tensor::<f32>::random(&[4, 1]);
        let out: Tensor<f32> = sequence.forward(input, false).unwrap();
        /*
        let weights = Tensor::new_row(
            vec![4, 3],
            vec![
                1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0,
            ],
        )
        .unwrap();
        let bias = Tensor::new(vec![3, 1], vec![0.4, 1.5, 15.0]).unwrap();
        let cache_input = None;
        let a = Linear::new(weights, bias, cache_input);
        */
    }

    #[test]
    fn chain_linear_val_check() {
        let layer1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a: Box<dyn ModuleDyn<f32>> =
            Box::new(Linear::new(layer1, Tensor::zeros([2, 1]).unwrap(), None).unwrap());
        let layer2 = Tensor::new(vec![1, 2], vec![100.0, 1.0]).unwrap();
        let b: Box<dyn ModuleDyn<f32>> =
            Box::new(Linear::new(layer2, Tensor::zeros([1, 1]).unwrap(), None).unwrap());

        let modules = vec![a, b];
        let mut sequence = Chain::new(modules);

        let input = Tensor::new(vec![2, 1], vec![1.0, 2.0]).unwrap();
        let out: Tensor<f32> = sequence.forward(input, false).unwrap();
        println!("Tensor output: {out}");
    }
}
