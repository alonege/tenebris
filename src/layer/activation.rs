//! Activation Layer Module.
//!
//! Applies highly-optimized, statically dispatched non-linear activation functions.

use crate::error::LibError;
use crate::layer::Layer;
use crate::profile_layer;
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::TensorFloat;

/// Mathematical definition of an activation function.
///
/// This trait allows for fully static dispatch and aggressive LLVM inlining,
/// enabling auto-vectorization (SIMD) across tensor elements.
pub trait ActivationFn<T: TensorFloat>: Clone + Send + Sync {
    /// Applies the activation function to a single element.
    fn activate(&self, x: T) -> T;

    /// Computes the derivative of the activation function for a single element.
    fn derivative(&self, x: T) -> T;
}

/// Rectified Linear Unit (ReLU) activation function.
#[derive(Debug, Clone, Default)]
pub struct Relu;

impl<T: TensorFloat> ActivationFn<T> for Relu {
    #[inline(always)]
    fn activate(&self, x: T) -> T {
        if x >= T::zero() { x } else { T::zero() }
    }

    #[inline(always)]
    fn derivative(&self, x: T) -> T {
        if x >= T::zero() { T::one() } else { T::zero() }
    }
}

/// Leaky ReLU activation function with a configurable alpha slope.
#[derive(Debug, Clone)]
pub struct LeakyRelu<T: TensorFloat> {
    pub alpha: T,
}

impl<T: TensorFloat> ActivationFn<T> for LeakyRelu<T> {
    #[inline(always)]
    fn activate(&self, x: T) -> T {
        if x >= T::zero() { x } else { self.alpha * x }
    }

    #[inline(always)]
    fn derivative(&self, x: T) -> T {
        if x >= T::zero() { T::one() } else { self.alpha }
    }
}

/// Leaky ELU activation function with a configurable alpha slope.
#[derive(Debug, Clone)]
pub struct Elu<T: TensorFloat> {
    pub alpha: T,
}

impl<T: TensorFloat> ActivationFn<T> for Elu<T> {
    #[inline(always)]
    fn activate(&self, x: T) -> T {
        if x >= T::zero() {
            x
        } else {
            self.alpha * (x.exp() - T::one())
        }
    }

    #[inline(always)]
    fn derivative(&self, x: T) -> T {
        if x >= T::zero() {
            T::one()
        } else {
            self.alpha * x.exp()
        }
    }
}

/// Sigmoid activation function.
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl<T: TensorFloat> ActivationFn<T> for Sigmoid {
    #[inline(always)]
    fn activate(&self, x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }

    #[inline(always)]
    fn derivative(&self, x: T) -> T {
        let s = self.activate(x); // Reuse activate logic!
        s * (T::one() - s)
    }
}

/// Tanh activation function.
#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl<T: TensorFloat> ActivationFn<T> for Tanh {
    #[inline(always)]
    fn activate(&self, x: T) -> T {
        x.tanh()
    }

    #[inline(always)]
    fn derivative(&self, x: T) -> T {
        let t = x.tanh();
        T::one() - t * t
    }
}

/// The Universal Activation Layer.
///
/// It wraps any static `ActivationFn` and handles dimensionality-agnostic caching
/// and Copy-On-Write logic for the forward and backward passes.
#[derive(Debug, Clone)]
pub struct Activation<T: TensorFloat, F: ActivationFn<T>> {
    /// The statically dispatched mathematical function (e.g., Relu, Sigmoid)
    func: F,
    /// Flat 1D cache to support inputs of any dimensionality `D`
    input_cache: Option<Tensor<T, 1>>,
}

impl<T: TensorFloat, F: ActivationFn<T>> Activation<T, F> {
    /// Creates a new activation layer with the specified function logic.
    pub fn new(func: F) -> Self {
        Self {
            func,
            input_cache: None,
        }
    }
}

impl<T: TensorFloat, const D: usize, F: ActivationFn<T>> Layer<Tensor<T, D>> for Activation<T, F> {
    type Output = Tensor<T, D>;

    #[inline(always)]
    fn forward(&mut self, input: Tensor<T, D>, save_grads: bool) -> Result<Self::Output, LibError> {
        profile_layer!(Self, "Forward", {
            if save_grads {
                let numel = input.shape().iter().product();
                let input_r = input.reshape([numel])?;
                self.input_cache = Some(input_r);
            }

            let mut output = input.make_unique();

            // PERFORMANCE: THINK about paralell
            output.map_inplace(|x| self.func.activate(x));

            Ok(output)
        })
    }

    #[inline(always)]
    fn backward(&mut self, grad_output: Self::Output) -> Result<Tensor<T, D>, LibError> {
        profile_layer!(Self, "backward", {
            let cached_1d = self
                .input_cache
                .take()
                .ok_or(LibError::LayerErrorBackwardNoGradient)?;

            let cached_input: Tensor<T, D> = cached_1d.reshape(grad_output.shape().clone())?;

            let mut derivative_tensor = cached_input.make_unique();

            // Same here: fully inlined SIMD loop
            derivative_tensor.map_inplace(|x| self.func.derivative(x));

            Ok(grad_output.mul_elem(&derivative_tensor))
        })
    }

    #[inline(always)]
    fn visit_params<O: crate::optimizer::Optimizer>(&mut self, _optimizer: &mut O) {}

    #[inline(always)]
    fn clear_grad(&mut self) {
        self.input_cache = None;
    }
}
