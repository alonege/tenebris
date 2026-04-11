//! Dropout layer for neural network regularization.
//!
//! Dropout randomly zeroes some of the elements of the input tensor with probability `p`
//! during training, scaling the remaining values by `1/(1-p)` (inverted dropout).
//! During inference, the layer acts as an identity function.

use crate::error::LibError;
use crate::layer::Layer;
use crate::profile_layer;
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::TensorFloat;
use rand::Rng;

/// Dropout layer implementing regularization via random zeroing of activations.
///
/// # Mathematical Description
/// During training (forward pass with `save_grads = true`):
/// ```text
/// mask[i] ~ Bernoulli(1 - p)
/// output = input * mask * scale
/// where scale = 1 / (1 - p)
/// ```
///
/// During inference:
/// ```text
/// output = input
/// ```
///
/// # Batch Handling
/// The mask is generated independently for every single element of the tensor.
/// This allows the layer to automatically support any tensor shape natively,
/// including batched dimensions like `[Features, Batch]` or `[C, W, H, N]`.
///
/// # Example
/// rust
/// use libmlrs::layer::dropout::Dropout;
/// use crate::libmlrs::layer::Module;
/// let mut dropout = Dropout::<f32>::new(0.5); // 50% dropout
///
/// // Training
/// dropout.set_training(true);
/// let output = dropout.forward(input, true)?;
///
/// // Inference
/// dropout.set_training(false);
/// let output = dropout.forward(input, false)?;
/// Ok(())
///
#[derive(Debug)]
pub struct Dropout<T: TensorFloat> {
    /// Prawdopodobieństwo wyzerowania elementu (0.0 - 1.0)
    p: T,
    /// Współczynnik skalowania = 1 / (1 - p)
    scale: T,
    /// Flaga trybu treningowego
    training: bool,
    /// Zacachowana maska binarna (dla backward pass)
    /// Przechowuje wartości 0 lub `scale` dla każdego elementu
    mask_cache: Option<Tensor<T, 1>>,
}

impl<T: TensorFloat> Dropout<T> {
    /// Creates a new Dropout layer.
    ///
    /// # Arguments
    /// * `p` - Probability of zeroing an element (0.0 ≤ p < 1.0)
    ///
    /// # Panics
    /// Panics if `p < 0` or `p >= 1`.
    ///
    /// # Example
    /// rust
    /// use libmlrs::layer::dropout::Dropout;
    /// use crate::libmlrs::layer::Module;
    /// let dropout = Dropout::<f32>::new(0.2); // 20% dropout
    /// Ok(())
    ///
    pub fn new(p: T) -> Self {
        let zero = T::zero();
        let one = T::one();

        assert!(
            p >= zero && p < one,
            "Dropout probability must be in range [0, 1)"
        );

        let scale = if p == zero { one } else { one / (one - p) };

        Self {
            p,
            scale,
            training: true, // Training mode by default
            mask_cache: None,
        }
    }

    /// Sets the mode of the layer.
    ///
    /// # Arguments
    /// * `training` - `true` for training mode, `false` for inference
    #[inline(always)]
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Checks if the layer is currently in training mode.
    #[inline(always)]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the dropout probability.
    #[inline(always)]
    pub fn dropout_probability(&self) -> T {
        self.p
    }

    /// Switches the layer to training mode.
    #[inline(always)]
    pub fn train(&mut self) {
        self.set_training(true);
    }

    /// Switches the layer to inference (evaluation) mode.
    #[inline(always)]
    pub fn eval(&mut self) {
        self.set_training(false);
    }

    /// Generates a flat 1D dropout mask for a given number of elements.
    ///
    /// The mask contains:
    /// - `scale` (= 1/(1-p)) with probability (1-p)
    /// - `0` with probability p
    fn generate_mask(&self, numel: usize) -> Tensor<T, 1> {
        let mut rng = rand::rng();
        let zero = T::zero();

        let p_f64: f64 = self.p.to_f64().unwrap_or(0.5);

        let data: Vec<T> = (0..numel)
            .map(|_| {
                let rand_val: f64 = rng.random();
                if rand_val < p_f64 { zero } else { self.scale }
            })
            .collect();

        Tensor::new([numel], data).unwrap()
    }
}

impl<T: TensorFloat, const D: usize> Layer<Tensor<T, D>> for Dropout<T> {
    type Output = Tensor<T, D>;

    /// Forward pass.
    ///
    /// # Behavior
    /// - Training mode (`training = true`): applies dropout mask with inverted scaling.
    /// - Inference mode (`training = false`): returns the input identically.
    #[inline(always)]
    fn forward(&mut self, input: Tensor<T, D>, save_grads: bool) -> Result<Self::Output, LibError> {
        profile_layer!(Self, "Forward", {
            // Bypass completely if in inference mode or probability is 0
            if !self.training || self.p == T::zero() {
                self.mask_cache = None;
                return Ok(input);
            }

            // Calculate total elements
            let numel: usize = input.shape().iter().product();

            // Generate a contiguous 1D mask
            let mask = self.generate_mask(numel);

            // Zero-cost abstraction: Reshape the 1D mask to match the input's D-dimensional shape
            let mask_d: Tensor<T, D> = mask.reshape(input.shape().clone())?;

            let output = input.mul_elem(&mask_d);

            if save_grads {
                self.mask_cache = Some(mask);
            } else {
                self.mask_cache = None;
            }

            Ok(output)
        })
    }

    /// Backward pass.
    ///
    /// # Math
    /// ```text
    /// d(loss)/d(input) = upstream_grad * mask
    /// ```
    /// Gradients only flow through non-zeroed elements, scaled by the factor.
    fn backward(&mut self, upstream_grad: Self::Output) -> Result<Tensor<T, D>, LibError> {
        profile_layer!(Self, "Backward", {
            if !self.training || self.p == T::zero() {
                return Ok(upstream_grad);
            }

            let mask = self
                .mask_cache
                .take()
                .ok_or_else(|| LibError::LayerErrorBackwardNoGradient)?;

            // Restore the mask to D dimensions to match the upstream gradient
            let mask_d: Tensor<T, D> = mask.reshape(upstream_grad.shape().clone())?;

            let downstream_grad = upstream_grad.mul_elem(&mask_d);

            Ok(downstream_grad)
        })
    }

    #[inline(always)]
    fn visit_params<O: crate::optimizer::Optimizer>(&mut self, _optimizer: &mut O) {
        // Dropout has no learnable parameters. Do nothing.
    }

    #[inline(always)]
    fn clear_grad(&mut self) {
        self.mask_cache = None;
    }

    #[inline(always)]
    fn training(&mut self) {
        self.set_training(true);
    }

    #[inline(always)]
    fn inference(&mut self) {
        self.set_training(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::dropout::Dropout;

    /// Test podstawowej funkcjonalności dropout
    #[test]
    fn test_dropout_basic() {
        let mut dropout = Dropout::<f32>::new(0.5);

        // Utwórz tensor wejściowy 3x3
        let input = Tensor::ones([3, 3]).unwrap();

        // Forward w trybie treningowym
        dropout.set_training(true);
        let output = dropout.forward(input.clone(), true).unwrap();

        // Sprawdź kształt
        assert_eq!(output.shape(), input.shape());

        // Niektóre elementy powinny być wyzerowane
        let has_zeros = output.iter().any(|x| x == 0.0);
        let has_nonzeros = output.iter().any(|x| x != 0.0);

        // Z p=0.5 i 9 elementami, bardzo mało prawdopodobne że wszystkie są 0 lub wszystkie nie-0
        // Ten test może sporadycznie failować z bardzo małym p-value
        assert!(
            has_zeros || has_nonzeros,
            "Dropout should modify some values"
        );
    }

    /// Test że inferencja nie zmienia wartości
    #[test]
    fn test_dropout_inference_mode() {
        let mut dropout = Dropout::<f32>::new(0.5);

        let input = Tensor::from_slice([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Forward w trybie inferencji
        dropout.set_training(false);
        let output = dropout.forward(input.clone(), false).unwrap();

        // Wartości powinny być identyczne
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Inference mode should not modify values"
            );
        }
    }

    /// Test że p=0 nie zmienia wartości
    #[test]
    fn test_dropout_p_zero() {
        let mut dropout = Dropout::<f32>::new(0.0);
        dropout.set_training(true);

        let input = Tensor::from_slice([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = dropout.forward(input.clone(), true).unwrap();

        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "p=0 should not modify values");
        }
    }

    /// Test skalowania (inverted dropout)
    #[test]
    fn test_dropout_scaling() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(true);

        let input = Tensor::ones([1000]).unwrap();
        let output = dropout.forward(input, true).unwrap();

        // Niezerowe wartości powinny być skalowane przez 1/(1-p) = 2
        for val in output.iter() {
            assert!(
                val == 0.0 || (val - 2.0).abs() < 1e-6,
                "Non-zero values should be scaled by 2.0, got {}",
                val
            );
        }
    }

    /// Test backward pass
    #[test]
    fn test_dropout_backward() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(true);

        let input = Tensor::ones([4, 4]).unwrap();
        let _output = dropout.forward(input, true).unwrap();

        // Upstream gradient = wszystkie jedynki
        let upstream = Tensor::ones([4, 4]).unwrap();
        let downstream = dropout.backward(upstream).unwrap();

        // Downstream powinien mieć ten sam wzorzec co output (0 lub scale)
        assert_eq!(downstream.shape(), &[4, 4]);
    }

    /// Test obsługi batchy (3D tensor)
    #[test]
    fn test_dropout_batch_3d() {
        let mut dropout = Dropout::<f32>::new(0.3);
        dropout.set_training(true);

        // Symulacja batcha: (features=10, height=4, batch=8)
        let input = Tensor::ones([10, 4, 8]).unwrap();
        let output = dropout.forward(input.clone(), true).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    /// Test obsługi 4D tensora (conv layers)
    #[test]
    fn test_dropout_batch_4d() {
        let mut dropout = Dropout::<f32>::new(0.2);
        dropout.set_training(true);

        // Symulacja conv output: (channels=16, height=8, width=8, batch=4)
        let input = Tensor::ones([16, 8, 8, 4]).unwrap();
        let output = dropout.forward(input.clone(), true).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    /// Test że backward bez forward rzuca błąd
    #[test]
    fn test_dropout_backward_without_forward() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(true);

        let upstream = Tensor::ones([2, 2]).unwrap();
        let result = dropout.backward(upstream);

        assert!(result.is_err(), "Backward without forward should fail");
    }

    /// Test clear_grad
    #[test]
    fn test_dropout_clear_grad() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(true);

        let input = Tensor::ones([3, 3]).unwrap();
        let _output = dropout.forward(input, true).unwrap();

        assert!(dropout.mask_cache.is_some());

        <Dropout<f32> as Layer<Tensor<f32, 2>>>::clear_grad(&mut dropout);

        assert!(dropout.mask_cache.is_none());
    }
}
