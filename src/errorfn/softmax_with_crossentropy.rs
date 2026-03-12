use crate::{
    error::LibError,
    errorfn::ErrorFn,
    tensor::tensor::{Tensor, TensorFloat},
};

/// Implementacja Fused Softmax + Cross Entropy.
/// Działa na surowych logitach (y_pred) i one-hot targetach (y_true).
pub struct SoftmaxCrossEntropy;

impl SoftmaxCrossEntropy {
    pub fn new() -> Self {
        Self
    }

    /// Helper fn -- returns max in 0 axis (for each batch element)
    /// ## Returns
    /// tensor [1, N].
    fn max_axis_0<T: TensorFloat>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>, LibError> {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        let mut max_data = Vec::with_capacity(cols);

        for c in 0..cols {
            let mut col_max = T::neg_infinity();
            for r in 0..rows {
                if let Some(val) = tensor.get(&[r, c]) {
                    if val > col_max {
                        col_max = val;
                    }
                }
            }
            max_data.push(col_max);
        }
        Tensor::new(vec![1, cols], max_data)
    }
}

impl<T: TensorFloat> ErrorFn<T> for SoftmaxCrossEntropy {
    fn compute(&self, logits: &Tensor<T>, targets: &Tensor<T>) -> Result<(T, Tensor<T>), LibError> {
        if logits.shape() != targets.shape() {
            return Err(LibError::ShapeMismatch {
                operation: "SoftmaxCrossEntropy::compute".into(),
                expected: logits.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        // SOFTMAX

        //println!("=================================");
        //println!("Logits: {:?}", logits);
        //println!("Targets: {:?}", targets);
        // Log-Sum-Exp trick: M = max(logits)
        // dla każdego elementu batcha dostajemy max dla danej próbki
        let max_vals = self.max_axis_0(logits)?;
        //println!("Max vals: {:?}", max_vals);
        let max_vals_expanded = max_vals.expand(logits.shape())?;
        //println!("Max vals expanded: {:?}", max_vals_expanded);

        // shifted = logits - M
        let shifted = logits.sub(&max_vals_expanded)?;
        //println!("Shifted logits: {:?}", shifted);

        // exp_shifted = exp(logits - M)
        let exp_shifted = shifted.map(|x| x.exp());
        //println!("Exp shifted: {:?}", exp_shifted);

        // sum_exp = sum(exp_shifted)
        let sum_exp = exp_shifted.sum(0);
        //println!("Sum exp: {:?}", sum_exp);
        let log_sum_exp = sum_exp.map(|x| x.ln());
        let log_sum_exp_expanded = log_sum_exp.expand(logits.shape())?;

        // log_probs = shifted - log(sum(exp))
        let log_probs = shifted.sub(&log_sum_exp_expanded)?;

        // LOSS

        // Loss = -sum(targets * log_probs) / batch_size
        let element_loss = targets.mul_elem(&log_probs);
        let total_sum = element_loss.sum_all();

        let batch_size = logits.shape()[1];
        //let mean_loss = -total_sum / (T::from(batch_size).unwrap());
        let loss = -total_sum;

        // GRAD

        // Gradient = (probs - targets) / batch_size
        // probs = exp(log_probs)
        let probs = log_probs.map(|x| x.exp());

        let diff = probs.sub(targets)?;
        let scale = T::one() / (T::from(batch_size).unwrap());
        let grad = diff.map(|x| *x * scale);

        //println!("grad: {:?}", grad);
        //println!("==============================");
        Ok((loss, grad))
    }
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    use super::*;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_error_fn_trait_usage() {
        // Setup: Batch 1, 2 klasy
        // Logity: [0, 0] -> Probs: [0.5, 0.5]
        let logits = Tensor::new(vec![2, 1], vec![0.0, 0.0]).unwrap();
        // Target: Klasa 0
        let targets = Tensor::new(vec![2, 1], vec![1.0, 0.0]).unwrap();

        let criterion = SoftmaxCrossEntropy::new();

        // Test compute (loss + grad)
        let (loss, grad) = criterion.compute(&logits, &targets).unwrap();

        // Oczekiwana strata: -log(0.5) ≈ 0.693
        println!("Loss: {}", loss);
        assert!((loss.to_f32().unwrap() - 0.6931).abs() < 0.001);

        // Oczekiwany gradient: (probs - target) / 1
        // [0.5 - 1.0, 0.5 - 0.0] = [-0.5, 0.5]
        let g0 = grad.get(&[0, 0]).unwrap();
        let g1 = grad.get(&[1, 0]).unwrap();

        assert!((g0.to_f32().unwrap() - (-0.5)).abs() < 0.001);
        assert!((g1.to_f32().unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_softmax_loss_magnitude() {
        use num_traits::ToPrimitive;

        let criterion = SoftmaxCrossEntropy::new();

        // Uniform random logits (model nie wie nic)
        let logits = Tensor::zeros(&[10, 128]).unwrap(); // Batch=128, classes=10

        // Random targets (one-hot)
        let mut targets_data = vec![0.0; 10 * 128];
        for b in 0..128 {
            targets_data[b] = 1.0; // Klasa 0 dla wszystkich (dla prostoty)
        }
        let targets = Tensor::from_slice(vec![10, 128], targets_data).unwrap();

        let (loss, _grad) = criterion.compute(&logits, &targets).unwrap();

        println!("Loss for uniform logits: {}", loss);

        // Dla uniform logits (wszystkie 0), softmax daje p=1/10 dla każdej klasy
        // Loss = -log(1/10) ≈ 2.3026
        let expected_loss = 2.3026_f32;

        assert!(
            (loss.to_f32().unwrap() - expected_loss).abs() < 0.01,
            "Loss {} should be close to {} for uniform predictions!",
            loss.to_f32().unwrap(),
            expected_loss
        );
    }
    #[test]
    fn test_softmax_gradient_scale() {
        let criterion = SoftmaxCrossEntropy::new();

        // Batch=2, classes=3
        let logits = Tensor::from_slice(
            vec![3, 2],
            vec![
                1.0, 2.0, // class 0
                2.0, 1.0, // class 1
                0.5, 0.5, // class 2
            ],
        )
        .unwrap();

        let targets = Tensor::from_slice(
            vec![3, 2],
            vec![
                1.0, 0.0, // sample 0: class 0
                0.0, 1.0, // sample 1: class 1
                0.0, 0.0,
            ],
        )
        .unwrap();

        let (_loss, grad) = criterion.compute(&logits, &targets).unwrap();

        println!("Gradient:\n{:?}", grad);

        // Gradient[i, j] = (p[i, j] - target[i, j]) / batch_size
        // NIE: (p[i, j] - target[i, j]) / (batch_size × num_classes)

        // Sprawdź magnitude
        let grad_max = grad.iter().map(|x: f32| x.abs()).fold(0.0, f32::max);
        println!("Max |grad|: {}", grad_max);

        // Z poprawną normalizacją, max grad powinien być ~O(0.5)
        // Z błędną (÷ num_classes), byłoby ~O(0.17)
        assert!(
            grad_max > 0.2,
            "Gradient too small! Still divided by num_classes?"
        );
    }
    #[test]
    fn test_single_vs_batch_gradient() {
        let criterion = SoftmaxCrossEntropy::new();

        // Single sample
        let logits_1: Tensor<f32> = Tensor::new(
            vec![10, 1],
            vec![0.5, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        )
        .unwrap();
        let targets_1 = Tensor::new(
            vec![10, 1],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let (loss1, grad1) = criterion.compute(&logits_1, &targets_1).unwrap();

        // Batch of 2 identical samples
        let logits_2 = Tensor::new(
            vec![10, 2],
            vec![
                0.5, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.5, 0.1, 0.2, 0.3, -0.1,
                -0.2, -0.3, -0.4, -0.5, -0.6,
            ],
        )
        .unwrap();
        let targets_2 = Tensor::new(
            vec![10, 2],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let (loss2, grad2) = criterion.compute(&logits_2, &targets_2).unwrap();

        println!("Loss single: {}, Loss batch: {}", loss1, loss2);
        println!("Grad single: {:?}", grad1.get(&[0, 0]));
        println!("Grad batch[0]: {:?}", grad2.get(&[0, 0]));
        println!("Grad batch[1]: {:?}", grad2.get(&[0, 1]));

        // Loss powinien być IDENTYCZNY (mean loss per sample)
        // Gradient dla każdego sample w batch powinien być IDENTYCZNY jak single

        assert!((loss1 - loss2).abs() < 1e-5, "Loss mismatch!");
        assert!((grad1.get(&[0, 0]).unwrap() - grad2.get(&[0, 0]).unwrap()).abs() < 1e-5);
    }
}
