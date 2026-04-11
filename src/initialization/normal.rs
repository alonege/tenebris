use rand::distr::Distribution;
use rand::rng;
use rand_distr::Normal as NormalDist;

use crate::{error::LibError, initialization::InitializationRawData, tensor::tensor::TensorFloat};

/// Normal (Gaussian) distribution initialization with custom parameters
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    pub mean: f64,
    pub std: f64,
}

impl Normal {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl<T: TensorFloat> InitializationRawData<T> for Normal {
    fn initialize_data(
        &self,
        _fan_in: usize,
        _fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, LibError> {
        let mut rng = rng();
        let normal = NormalDist::new(self.mean, self.std).map_err(|e| {
            LibError::OtherError(format!("Failed to create normal distribution: {}", e))
        })?;

        let data: Vec<T> = (0..n_elements)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();

        Ok(data)
    }
}

impl Default for Normal {
    fn default() -> Self {
        Self::standard()
    }
}
