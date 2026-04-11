use crate::{
    error::LibError,
    layer::Module,
    profile_layer,
    tensor::{
        tensor::{Tensor, TensorFloat},
        tensorops::MatMul,
    },
};

pub struct Softmax<T: TensorFloat> {
    output: Option<Tensor<T>>,
}

impl<T: TensorFloat> Softmax<T> {
    pub fn new() -> Self {
        Softmax { output: None }
    }
}

impl<T> Module<T> for Softmax<T>
where
    T: TensorFloat,
{
    type Input<A> = Tensor<T>;
    type Output<B> = Tensor<T>;
    fn forward(
        &mut self,
        input: Self::Input<T>,
        save_grads: bool,
    ) -> Result<Self::Output<T>, LibError> {
        profile_layer!(Self, "Forward", {
            //y[i] = exp(x[i]) / Σⱼ exp(x[j])
            //println!("Softmax input: {:?}", input);
            if input.shape().len() != 2 {
                return Err(LibError::InvalidDimensionality {
                    operation: "Softmax needs 2 dim tensor".to_string(),
                    expected: 2,
                    actual: input.shape().len(),
                });
            }
            if input.shape()[1] != 1 {
                return Err(LibError::InvalidDimensionality {
                    operation: "Softmax needs shape (N, 1)".to_string(),
                    expected: 1,
                    actual: input.shape()[1],
                });
            }
            let max_val = input.iter().fold(T::neg_infinity(), T::max);
            let mut input = input.make_unique();
            input.map_inplace(|x| T::exp(x - max_val));
            let mut input_exp = input;
            let exp_sum: T = input_exp.iter().fold(T::zero(), |acc, x| acc + x);
            input_exp.map_inplace(|x| x / exp_sum);
            //println!("Softmax output: {:?}", input_exp);
            if save_grads {
                self.output = Some(input_exp.clone());
            }
            Ok(input_exp)
        })
    }

    fn backward(&mut self, grad_output: Self::Input<T>) -> Result<Self::Output<T>, LibError> {
        profile_layer!(Self, "Backward", {
            let y = self.output.take();
            let y = match y {
                Some(y) => y,
                None => {
                    return Err(LibError::LayerErrorBackwardNoGradient);
                }
            };
            let dot = y.t().matmul(&grad_output)?;
            let dot = dot.get(&[0, 0]).unwrap();
            //println!("Softmax backward dot: {:?}", dot);

            let mut grad_output = grad_output.make_unique();
            grad_output.map_inplace(|x| x - dot);
            let grad_minus_dot = grad_output;
            //println!("Softmax backward grad_minus_dot: {:?}", grad_minus_dot);

            let downstream_grad = y.mul_elem(&grad_minus_dot);
            //println!("Softmax backward downstream_grad: {:?}", downstream_grad);

            Ok(downstream_grad)
        })
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn clear_grad(&mut self) {
        self.output = None;
    }
}

#[allow(unused)]
pub mod tests {
    use crate::{layer::Module, tensor::tensor::Tensor};

    use super::Softmax;

    #[test]
    fn test_softmax_forward() {
        let mut softmax: Softmax<f32> = Softmax::new();
        let input = Tensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let output = softmax.forward(input, false).unwrap();
        let expected_output =
            Tensor::new(vec![3, 1], vec![0.09003057, 0.24472848, 0.66524094]).unwrap();
        for (o, e) in output.iter().zip(expected_output.iter()) {
            assert!((o - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_backward() {
        let mut softmax: Softmax<f32> = Softmax::new();
        let input = Tensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let _ = softmax.forward(input, true).unwrap();
        let grad_output = Tensor::new(vec![3, 1], vec![0.1, 0.2, 0.7]).unwrap();
        let grad_input = softmax.backward(grad_output).unwrap();
        let expected_grad_input =
            Tensor::new(vec![3, 1], vec![-0.038138516, -0.079198375, 0.117336891]).unwrap();
        for (g, e) in grad_input.iter().zip(expected_grad_input.iter()) {
            assert!((g - e).abs() < 1e-5);
        }
    }
}
