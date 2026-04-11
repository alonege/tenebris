use rand::rng;
use rand_distr::{Distribution, Normal as NormalDist};

use crate::{
    error::LibError,
    initialization::{Initialization, InitializationRawData},
    tensor::tensor::TensorFloat,
};

/// Xavier/Glorot initialization
///
/// Best for: Tanh, Sigmoid activations
///
/// Variance = 2 / (fan_in + fan_out)
///
/// # References
/// - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
#[derive(Clone, Copy, Debug)]
pub struct Glorot;

impl Glorot {
    pub fn new() -> Self {
        Self
    }
}

impl<T: TensorFloat> InitializationRawData<T> for Glorot {
    fn initialize_data(
        &self,
        fan_in: usize,
        fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, LibError> {
        let std = (2.0 / (fan_in + fan_out) as f64).sqrt();

        let mut rng = rng();
        let normal = NormalDist::new(0.0, std).map_err(|e| {
            LibError::OtherError(format!("Failed to create normal distribution: {}", e))
        })?;

        let data: Vec<T> = (0..n_elements)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();

        Ok(data)
    }
}

impl<T: TensorFloat> Initialization<T> for Glorot {}

impl Default for Glorot {
    fn default() -> Self {
        Self::new()
    }
}
