use crate::layer::Module;
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::TensorFloat;

/// Activation Module struct.
///
/// It contains both activation function and its derivative.
pub struct Activation<T: TensorFloat> {
    fun: Box<dyn Fn(T) -> T + Send + Sync>,
    derivative: Box<dyn Fn(T) -> T + Send + Sync>,
    input_cache: Option<Tensor<T>>,
}

impl<T: TensorFloat + 'static> Activation<T> {
    /// function to create new activation Module with user-defined
    /// function and its derivative.
    ///
    /// Example for creating Activation Module with square function:
    /// ```
    /// use libmlrs::layer::activation::Activation;
    /// use libmlrs::tensor::tensor::Tensor;
    /// use crate::libmlrs::layer::Module;
    ///
    /// let square = |x| x * x;
    /// let square_deriv = |x| 2.0 * x;
    ///
    /// let mut square = Activation::new(square, square_deriv);
    ///
    /// let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    ///
    /// let output = square.forward(input.clone(), false).unwrap();
    /// assert_eq!(output.get_data(), &[1.0, 4.0, 9.0, 16.0]);
    /// ```
    pub fn new<F, D>(fun: F, derivative: D) -> Self
    where
        F: Fn(T) -> T + 'static + Send + Sync,
        D: Fn(T) -> T + 'static + Send + Sync,
    {
        Self {
            fun: Box::new(fun),
            derivative: Box::new(derivative),
            input_cache: None,
        }
    }

    pub fn relu() -> Self {
        let fun = Box::new(move |x: T| if x >= T::zero() { x } else { T::zero() });
        let derivative = Box::new(move |x: T| if x >= T::zero() { T::one() } else { T::zero() });
        Self {
            fun,
            derivative,
            input_cache: None,
        }
    }

    pub fn lrelu(alpha: T) -> Self {
        let fun = Box::new(move |x: T| if x >= T::zero() { x } else { alpha * x });
        let derivative = Box::new(move |x: T| if x >= T::zero() { T::one() } else { alpha });
        Self {
            fun,
            derivative,
            input_cache: None,
        }
    }

    pub fn elu(alpha: T) -> Self {
        let fun = Box::new(move |x: T| {
            if x >= T::zero() {
                x
            } else {
                alpha * (x.exp() - T::one())
            }
        });
        let derivative = Box::new(move |x: T| {
            if x >= T::zero() {
                T::one()
            } else {
                alpha * x.exp()
            }
        });

        Self {
            fun,
            derivative,
            input_cache: None,
        }
    }

    pub fn sigmoid() -> Self {
        let fun = Box::new(|x: T| T::one() / (T::one() + (-x).exp()));
        let derivative = Box::new(|x: T| {
            let s = T::one() / (T::one() + (-x).exp());
            s * (T::one() - s)
        });
        Self {
            fun,
            derivative,
            input_cache: None,
        }
    }

    pub fn tanh() -> Self {
        let fun = Box::new(|x: T| x.tanh());
        let derivative = Box::new(|x: T| {
            let t = x.tanh();
            T::one() - t * t
        });
        Self {
            fun,
            derivative,
            input_cache: None,
        }
    }
}

impl<T: TensorFloat> Module<T> for Activation<T> {
    type Input<A> = Tensor<T>;
    type Output<B> = Tensor<T>;

    fn forward(
        &mut self,
        input: Self::Input<T>,
        save_grads: bool,
    ) -> Result<Self::Output<T>, crate::error::LibError> {
        if save_grads == true {
            self.input_cache = Some(input.clone());
        }
        match input.is_unique() {
            true => {
                let mut input = input.make_unique();
                input.map_inplace(|x| (self.fun)(x));
                return Ok(input);
            }
            false => {
                return Ok(input.map(|x| (self.fun)(*x)));
            }
        }
    }

    fn backward(
        &mut self,
        grad_output: Self::Input<T>,
    ) -> Result<Self::Output<T>, crate::error::LibError> {
        let input = self.input_cache.take();
        let input = match input {
            Some(a) => a,
            None => return Err(crate::error::LibError::LayerErrorBackwardNoGradient),
        };
        let input = match input.is_unique() {
            true => {
                let mut input = input.make_unique();
                input.map_inplace(|x| (self.derivative)(x));
                input
            }
            false => input.map(|x| (self.derivative)(*x)),
        };
        Ok(grad_output.mul_elem(&input))
    }

    fn parameters(&self) -> Vec<crate::tensor::tensor::Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut crate::tensor::tensor::Tensor<T>> {
        vec![]
    }

    fn clear_grad(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Module;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_activation_layer() {
        // Define a simple square function and its derivative
        let square = |x| x * x;
        let square_deriv = |x| 2.0 * x;

        // Create the activation layer
        let mut activation_layer = Activation::new(square, square_deriv);

        // Create input tensor
        let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test forward pass
        let output = activation_layer.forward(input.clone(), false).unwrap();
        assert_eq!(output.get_data(), &[1.0, 4.0, 9.0, 16.0]);

        // Test backward pass
        let _ = activation_layer.forward(input.clone(), true).unwrap();
        let grad = Tensor::new(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let downstream_grad = activation_layer.backward(grad).unwrap();

        // Expected downstream grad is grad * derivative(input)
        // derivative(input) = [2*1, 2*2, 2*3, 2*4] = [2, 4, 6, 8]
        // downstream_grad = [0.1*2, 0.2*4, 0.3*6, 0.4*8] = [0.2, 0.8, 1.8, 3.2]
        let expected_grad_data = vec![0.2, 0.8, 1.8, 3.2];
        let downstream_grad_data = downstream_grad.get_data();

        downstream_grad_data
            .iter()
            .zip(expected_grad_data.iter())
            .for_each(|(val, expected)| {
                assert!(((val - expected) as f64).abs() < 1e-6);
            });
    }
}
