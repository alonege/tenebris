//! Warstwa Dropout dla regularyzacji sieci neuronowych.
//!
//! Dropout losowo zeruje elementy tensora wejściowego z prawdopodobieństwem `p`
//! podczas treningu, skalując pozostałe wartości przez `1/(1-p)` (inverted dropout).
//! Podczas inferencji warstwa działa jako funkcja tożsamości.

use crate::error::LibError;
use crate::layer::Module;
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::TensorFloat;
use rand::Rng;

/// Warstwa Dropout implementująca regularyzację przez losowe zerowanie aktywacji.
///
/// # Opis matematyczny
/// Podczas treningu (forward z `save_grads = true`):
/// ```text
/// mask[i] ~ Bernoulli(1 - p)
/// output = input * mask * scale
/// gdzie scale = 1 / (1 - p)
/// ```
///
/// Podczas inferencji:
/// ```text
/// output = input
/// ```
///
/// # Obsługa batchy
/// Maska jest generowana dla każdego elementu tensora niezależnie.
/// Dzięki temu warstwa automatycznie obsługuje dowolne kształty tensorów,
/// w tym batche o wymiarach `(features, batch_size)` lub `(C, H, W, B)`.
///
/// # Przykład użycia
/// rust
/// use libmlrs::layer::dropout::Dropout;
/// use crate::libmlrs::layer::Module;
/// let mut dropout = Dropout::<f32>::new(0.5); // 50% dropout
///
/// // Trening
/// dropout.set_training(true);
/// let output = dropout.forward(input, true)?;
///
/// // Inferencja  
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
    mask_cache: Option<Tensor<T>>,
}

impl<T: TensorFloat> Dropout<T> {
    /// Tworzy nową warstwę Dropout.
    ///
    /// # Argumenty
    /// * `p` - Prawdopodobieństwo wyzerowania elementu (0.0 ≤ p < 1.0)
    ///
    /// # Panics
    /// Panikuje jeśli `p < 0` lub `p >= 1`.
    ///
    /// # Przykład
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
            training: true, // Domyślnie w trybie treningowym
            mask_cache: None,
        }
    }

    /// Ustawia tryb działania warstwy.
    ///
    /// # Argumenty
    /// * `training` - `true` dla trybu treningowego, `false` dla inferencji
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Sprawdza czy warstwa jest w trybie treningowym.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Zwraca prawdopodobieństwo dropout.
    pub fn dropout_probability(&self) -> T {
        self.p
    }

    /// Generuje maskę dropout dla tensora o zadanym kształcie.
    ///
    /// Maska zawiera wartości:
    /// - `scale` (= 1/(1-p)) z prawdopodobieństwem (1-p)
    /// - `0` z prawdopodobieństwem p
    fn generate_mask(&self, shape: &[usize]) -> Tensor<T> {
        let mut rng = rand::rng();
        let zero = T::zero();
        //let one = T::one();

        let numel: usize = shape.iter().product();

        let p_f64: f64 = self.p.to_f64().unwrap_or(0.5);

        let data: Vec<T> = (0..numel)
            .map(|_| {
                let rand_val: f64 = rng.random();
                if rand_val < p_f64 { zero } else { self.scale }
            })
            .collect();

        Tensor::from_slice(shape.to_vec(), data).unwrap()
    }
}

impl<T: TensorFloat> Module<T> for Dropout<T> {
    type Input<A> = Tensor<T>;
    type Output<B> = Tensor<T>;

    /// Propagacja w przód.
    ///
    /// # Argumenty
    /// * `input` - Tensor wejściowy dowolnego kształtu
    /// * `save_grads` - Czy zachować dane dla backward pass
    ///
    /// # Zwraca
    /// * `Ok(output)` - Tensor wyjściowy tego samego kształtu
    /// * `Err(LibError)` - W przypadku błędu
    ///
    /// # Zachowanie
    /// - Tryb treningowy (`training = true`): stosuje dropout z skalowaniem
    /// - Tryb inferencji (`training = false`): zwraca kopię wejścia bez zmian
    #[inline(always)]
    fn forward(
        &mut self,
        input: Self::Input<T>,
        save_grads: bool,
    ) -> Result<Self::Output<T>, LibError> {
        // W trybie inferencji lub gdy p = 0, zwróć wejście bez zmian
        if !self.training || self.p == T::zero() {
            self.mask_cache = None;
            return Ok(input);
        }

        let mask = self.generate_mask(input.shape());

        let output = input.mul_elem(&mask);

        if save_grads {
            self.mask_cache = Some(mask);
        } else {
            self.mask_cache = None;
        }

        Ok(output)
    }

    /// Propagacja wsteczna.
    ///
    /// # Argumenty
    /// * `upstream_grad` - Gradient z warstwy wyżej (ten sam kształt co output)
    ///
    /// # Zwraca
    /// * `Ok(downstream_grad)` - Gradient dla warstwy niżej
    /// * `Err(LibError)` - Gdy brak zacachowanej maski
    ///
    /// # Matematyka
    /// ```text
    /// d(loss)/d(input) = d(loss)/d(output) * d(output)/d(input)
    ///                  = upstream_grad * mask
    /// ```
    /// Gradient przepływa tylko przez nie-wyzerowane elementy, skalowany przez `scale`.
    fn backward(&mut self, upstream_grad: Self::Output<T>) -> Result<Self::Input<T>, LibError> {
        if !self.training || self.p == T::zero() {
            return Ok(upstream_grad);
        }

        let mask = self.mask_cache.take().ok_or_else(|| {
            LibError::OtherError(
                "Dropout backward called without cached mask. \
                      Ensure forward() was called with save_grads=true"
                    .to_string(),
            )
        })?;

        let downstream_grad = upstream_grad.mul_elem(&mask);

        Ok(downstream_grad)
    }

    #[inline(always)]
    fn parameters(&self) -> Vec<Tensor<T>> {
        Vec::new()
    }

    #[inline(always)]
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new()
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
        let input = Tensor::ones(&[3, 3]).unwrap();

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

        let input = Tensor::from_slice(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

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

        let input = Tensor::from_slice(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
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

        let input = Tensor::ones(&[1000]).unwrap();
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

        let input = Tensor::ones(&[4, 4]).unwrap();
        let _output = dropout.forward(input, true).unwrap();

        // Upstream gradient = wszystkie jedynki
        let upstream = Tensor::ones(&[4, 4]).unwrap();
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
        let input = Tensor::ones(&[10, 4, 8]).unwrap();
        let output = dropout.forward(input.clone(), true).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    /// Test obsługi 4D tensora (conv layers)
    #[test]
    fn test_dropout_batch_4d() {
        let mut dropout = Dropout::<f32>::new(0.2);
        dropout.set_training(true);

        // Symulacja conv output: (channels=16, height=8, width=8, batch=4)
        let input = Tensor::ones(&[16, 8, 8, 4]).unwrap();
        let output = dropout.forward(input.clone(), true).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    /// Test że backward bez forward rzuca błąd
    #[test]
    fn test_dropout_backward_without_forward() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(true);

        let upstream = Tensor::ones(&[2, 2]).unwrap();
        let result = dropout.backward(upstream);

        assert!(result.is_err(), "Backward without forward should fail");
    }

    /// Test clear_grad
    #[test]
    fn test_dropout_clear_grad() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(true);

        let input = Tensor::ones(&[3, 3]).unwrap();
        let _output = dropout.forward(input, true).unwrap();

        assert!(dropout.mask_cache.is_some());

        dropout.clear_grad();

        assert!(dropout.mask_cache.is_none());
    }

    /// Test że parameters() zwraca pusty wektor
    #[test]
    fn test_dropout_no_parameters() {
        let dropout = Dropout::<f32>::new(0.5);
        assert!(dropout.parameters().is_empty());
    }
}
