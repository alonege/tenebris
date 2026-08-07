use rand::distr::Distribution;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::Normal as NormalDist;

use crate::{error::LibError, initialization::Initialization, tensor::tensor::TensorFloat};

/// He/Kaiming initialization
///
/// Best for: ReLU, Leaky ReLU activations
///
/// Variance = 2 / fan_in
#[derive(Clone, Copy, Debug)]
pub struct He;

impl He {
    pub fn new() -> Self {
        Self
    }
}

impl<T: TensorFloat> Initialization<T> for He {
    fn initialize_data(
        &self,
        fan_in: usize,
        _fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, LibError> {
        let std = (2.0 / fan_in as f64).sqrt();

        let mut rng = SmallRng::from_rng(&mut rand::rng());
        let normal = NormalDist::new(0.0, std).map_err(|e| {
            LibError::OtherError(format!("Failed to create normal distribution: {}", e))
        })?;

        let data: Vec<T> = (0..n_elements)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();

        Ok(data)
    }
}

impl Default for He {
    fn default() -> Self {
        Self::new()
    }
}
