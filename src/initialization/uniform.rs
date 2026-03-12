use rand::distr::{Distribution, Uniform as UniformDist};
use rand::rng;

use crate::{initialization::Initialization, tensor::tensor::TensorFloat};

#[derive(Clone, Copy)]
pub struct Uniform;

impl Uniform {
    pub fn new() -> Self {
        Self
    }
}

impl<T: TensorFloat> Initialization<T> for Uniform {
    fn initialize_data(
        &self,
        _fan_in: usize,
        _fan_out: usize,
        n_elements: usize,
    ) -> Result<Vec<T>, crate::error::LibError> {
        let mut rng = rng();
        let u = UniformDist::new(0.0, 1.0).map_err(|e| {
            crate::error::LibError::OtherError(format!(
                "Failed to create uniform distribution: {}",
                e
            ))
        })?;
        (0..n_elements)
            .map(|_| {
                let sample = u.sample(&mut rng);
                T::from(sample).ok_or_else(|| {
                    crate::error::LibError::OtherError(
                        "Failed to convert sample to target type".to_string(),
                    )
                })
            })
            .collect()
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Self::new()
    }
}
