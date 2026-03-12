use rand::distr::Distribution;
use rand::rng;
use rand_distr::Normal as NormalDist;

use crate::{error::LibError, initialization::Initialization, tensor::tensor::TensorFloat};

/// LeCun initialization
///
/// Best for: SELU activation
///
/// Variance = 1 / fan_in
#[derive(Clone, Copy, Debug)]
pub struct LeCun;

impl LeCun {
    pub fn new() -> Self {
        Self
    }
}

impl<T: TensorFloat> Initialization<T> for LeCun {
    fn initialize_data(
        &self,
        fan_in: usize,
        _fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, LibError> {
        let std = (1.0 / fan_in as f64).sqrt();

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

impl Default for LeCun {
    fn default() -> Self {
        Self::new()
    }
}
