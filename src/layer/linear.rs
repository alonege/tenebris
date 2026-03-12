use crate::error::LibError;
use crate::layer::Module;
use crate::tensor::tensor::{Tensor, TensorFloat};
use crate::tensor::tensorops::MatMul;
use crate::tensor::tensorops::TensorAdd;

#[derive(Debug)]
pub struct Linear<T: TensorFloat> {
    weights: Tensor<T>,
    bias: Tensor<T>,

    cache_input: Option<Tensor<T>>,
}

impl<T: TensorFloat> Linear<T> {
    pub fn new(
        weights: Tensor<T>,
        bias: Tensor<T>,
        cache_input: Option<Tensor<T>>,
    ) -> Result<Self, LibError> {
        if weights.shape().len() != 2 || bias.shape().len() != 2 {
            return Err(LibError::InvalidDimensionalityTwoTensors {
                operation: "Linear::new()".to_string(),
                expected_a: 2,
                actual_a: weights.shape().len(),
                expected_b: 2,
                actual_b: bias.shape().len(),
            });
        }
        if weights.shape()[0] != bias.shape()[0] {
            return Err(LibError::InvalidDimensionality {
                operation: "Layer::new() weights.shape()[0] != bias.shape()[0]".to_string(),
                expected: weights.shape()[0],
                actual: bias.shape()[0],
            });
        }
        Ok(Self {
            weights,
            bias,
            cache_input,
        })
    }

    pub fn new_rand(in_size: usize, out_size: usize) -> Self {
        //let mut rng = rand::rng();
        //let mut random_cl = |_| rng.random();
        //let weights = Tensor::from_fn(vec![out_size, in_size], &mut random_cl).unwrap();
        //let bias = Tensor::from_fn(vec![out_size, 1], &mut random_cl).unwrap();
        let weights = Tensor::random(&[out_size, in_size]);
        let bias = Tensor::random(&[out_size, 1]);
        //let weights = Tensor::new(&[out_size, in_size], );

        Self {
            weights,
            bias,
            cache_input: None,
        }
    }

    /// TODO: FIX THIS MOCKUP
    pub fn new_val(in_size: usize, out_size: usize, val: T) -> Self {
        //let mut random_cl = |_, _| val;
        //let weights = Matrix::new((out_size, in_size), &mut random_cl);
        //let bias = Matrix::new((out_size, 1), &mut random_cl);

        let weights = Tensor::new_from_val(&[out_size, in_size], val).unwrap();
        let bias = Tensor::new_from_val(&[out_size, 1], val).unwrap();

        Self {
            weights,
            bias,
            cache_input: None,
        }
    }

    pub fn new_with_init(
        in_size: usize,
        out_size: usize,
        init: &impl crate::initialization::Initialization<T>,
    ) -> Result<Self, LibError> {
        let weights = init
            .initialize_tensor(in_size, out_size, vec![out_size, in_size])
            .unwrap();

        let bias = Tensor::zeros(vec![out_size, 1])?;
        //let bias = init.initialize_tensor(out_size, out_size, vec![out_size, 1])?;

        Ok(Self {
            weights,
            bias,
            cache_input: None,
        })
    }
}

impl<T: TensorFloat> Module<T> for Linear<T> {
    type Input<A> = Tensor<T>;
    type Output<B> = Tensor<T>;

    #[inline(always)]
    fn forward(
        &mut self,
        input: Self::Input<T>,
        for_backward: bool,
    ) -> Result<Self::Output<T>, LibError> {
        if for_backward {
            self.cache_input = Some(input.clone());
        }
        if input.shape().len() == 2 {
            // We have single Matrix as input
            let bias_broad = &self
                .bias
                .expand([self.weights.shape()[0], input.shape()[1]])?;
            self.weights.matmul(&input).unwrap().tensoradd(&bias_broad)
        } else {
            return Err(LibError::ShapeError(format!(
                "Input shape not supported for Linear layer: {:?}",
                input.shape()
            )));
        }
    }

    #[inline(always)]
    fn backward(&mut self, grad_output: Self::Input<T>) -> Result<Self::Output<T>, LibError> {
        let input = self.cache_input.as_ref().unwrap();

        //println!("{grad_output}");
        //println!("{}", input.t());
        let weights_grad = grad_output.matmul(&input.t())?;
        if let Some(existing_weights_grad) = self.weights.grad.take() {
            self.weights.grad = Some(Box::new(existing_weights_grad.tensoradd(&weights_grad)?));
        } else {
            self.weights.grad = Some(Box::new(weights_grad));
        }

        let bias_grad = grad_output.sum(1);
        if let Some(existing_bias_grad) = self.bias.grad.take() {
            self.bias.grad = Some(Box::new(existing_bias_grad.tensoradd(&bias_grad)?));
        } else {
            self.bias.grad = Some(Box::new(bias_grad));
        }

        let downstream_grad = self.weights.t().matmul(&grad_output)?;
        Ok(downstream_grad)
    }

    #[inline(always)]
    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    #[inline(always)]
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weights, &mut self.bias]
    }

    #[inline(always)]
    fn clear_grad(&mut self) {
        self.weights.grad = None;
        self.bias.grad = None;
    }
}

#[allow(unused_imports)]
mod tests {
    use crate::layer::linear::Linear;

    use crate::tensor::tensor::Tensor;
    use crate::tensor::tensorops::MatMul;
    use crate::tensor::tensorops::TensorAdd;

    #[test]
    fn test_linear_forward() {
        use crate::layer::Module;
        let mut linear = Linear::<f32>::new_val(3, 2, 1.0);
        println!("Weights: {:?}", linear.weights);
        println!("Bias: {:?}", linear.bias);
        let input = Tensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        println!("Input: {:?}", input);
        let output = linear.forward(input, false).unwrap();
        println!("Output: {:?}", output);
        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn backpropagation_notation_denominator_shapes() {
        use crate::layer::Module;
        let mut linear = Linear::<f32>::new_val(3, 2, 1.0);
        let input = Tensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let output = linear.forward(input, true);
        println!("Output: {:?}", output);

        let grad = Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap();
        let downstream_grad = linear.backward(grad).unwrap();
        println!("Downstream Grad: {:?}", downstream_grad);

        assert_eq!(downstream_grad.shape(), &[3, 1]);
        assert_eq!(linear.weights.grad.as_ref().unwrap().shape(), &[2, 3]);
        assert_eq!(linear.bias.grad.as_ref().unwrap().shape(), &[2, 1]);
    }

    /// Sprawdza poprawność matematyczną implementacji `backward` dla warstwy Linear.
    ///
    /// Test opiera się na prostym przypadku z ustalonymi wagami, wejściem i gradientem
    /// z kolejnej warstwy, a następnie porównuje wyniki z oczekiwanymi wartościami
    /// obliczonymi ręcznie na podstawie wzorów na pochodne.
    #[test]
    fn test_linear_backward_correctness() {
        use crate::layer::Module;
        // === Krok 1: ARRANGE (Przygotowanie danych) ===

        // Tworzymy warstwę Linear 2 -> 3 z prostymi, znanymi wagami.
        // Używamy typu f32, ponieważ jest najczęstszy i łatwy w testowaniu.
        let mut linear = Linear::<f32>::new_val(3, 2, 0.0);
        linear.weights = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        linear.bias = Tensor::new(vec![3, 1], vec![0.1, 0.2, 0.3]).unwrap();

        // Tworzymy tensor wejściowy (kształt [2, 1] - jedna próbka, 2 cechy)
        let input = Tensor::new(vec![2, 1], vec![10.0, 20.0]).unwrap();

        // Definiujemy gradient "płynący" z następnej warstwy (dL/dY).
        // Musi mieć taki sam kształt jak wyjście warstwy (dla Y=W*X+B, wyjście ma kształt [3, 1]).
        let grad_output = Tensor::new(vec![3, 1], vec![0.1, 0.2, 0.3]).unwrap();

        // === Krok 2: ACT (Wykonanie testowanej logiki) ===

        // Wykonujemy forward, aby zapisać `input` w `cache_input`.
        let _ = linear.forward(input.clone(), true);

        // Wykonujemy backward, aby obliczyć gradienty.
        let downstream_grad = linear.backward(grad_output.clone()).unwrap();

        // === Krok 3: ASSERT (Sprawdzenie wyników) ===

        // --- 3a. Sprawdzenie gradientu wag (dL/dW) ---
        // Wzór: dL/dW = dL/dY * X^T
        let expected_weights_grad =
            Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]).unwrap();
        match &linear.weights.grad {
            Some(wg) => {
                println!("Weights grad: {:?}", wg);
                assert_eq!(wg.get_data(), expected_weights_grad.get_data());
            }
            None => panic!("Weights grad is None"),
        }

        // --- 3b. Sprawdzenie gradientu biasu (dL/dB) ---
        // Wzór: dL/dB = dL/dY (sumowane po batchu, ale dla jednej próbki to po prostu dL/dY)
        // Twoja funkcja `sum(1)` dla wejścia [3,1] powinna zwrócić to samo.
        let expected_bias_grad = grad_output; // `grad.sum(1)` dla jednej próbki
        match &linear.bias.grad {
            Some(bg) => {
                println!("Bias grad: {:?}", bg);
                assert_eq!(bg.get_data(), expected_bias_grad.get_data());
            }
            None => panic!("Bias grad is None"),
        }

        // --- 3c. Sprawdzenie gradientu wejścia (dL/dX) ---
        // Wzór: dL/dX = W^T * dL/dY
        let expected_downstream_grad = Tensor::new(
            vec![2, 1],
            vec![1.4, 3.2], // [1*0.1+3*0.2+5*0.3, 2*0.1+4*0.2+6*0.3]
        )
        .unwrap();
        assert_eq!(
            &downstream_grad.get_data(),
            &expected_downstream_grad.get_data()
        );

        println!("Test `Linear::backward` zakończony sukcesem!");
    }

    #[test]
    fn test_module_linear_backward_correctness() {
        // === Krok 1: ARRANGE (Przygotowanie danych) ===
        use crate::layer::Module;

        // Tworzymy warstwę Linear 2 -> 3.
        //let mut linear = Linear::<f32>::new_val(2, 3, 1.0);
        let weights = Tensor::ones(vec![3, 2]).unwrap();
        let bias = Tensor::zeros(vec![3, 0]).unwrap();
        let mut linear = Linear::<f32>::new(weights, bias, None).unwrap();

        // Ustawiamy wagi. Dla kształtu [3, 2] w układzie column-major,
        // wektor `vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]` tworzy macierz:
        // W = [[1.0, 4.0],
        //      [2.0, 5.0],
        //      [3.0, 6.0]]
        linear.weights = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        linear.bias = Tensor::new(vec![3, 1], vec![0.1, 0.2, 0.3]).unwrap();

        // Wejście X (kształt [2, 1])
        // X = [[10.0],
        //      [20.0]]
        let input = Tensor::new(vec![2, 1], vec![10.0, 20.0]).unwrap();

        // Gradient z góry, dL/dY (kształt [3, 1])
        // dL/dY = [[0.1],
        //          [0.2],
        //          [0.3]]
        let grad_output = Tensor::new(vec![3, 1], vec![0.1, 0.2, 0.3]).unwrap();

        // === Krok 2: ACT (Wykonanie testowanej logiki) ===

        println!("{input}");
        println!("FORWARD");
        let _ = linear.forward(input.clone(), true).unwrap();
        println!("{input}");
        let downstream_grad = linear.backward(grad_output.clone()).unwrap();
        println!("{input}");

        // === Krok 3: ASSERT (Sprawdzenie wyników) ===

        // --- 3a. Sprawdzenie gradientu wag (dL/dW) ---
        // Wzór: dL/dW = dL/dY * X^T
        // dL/dY [3, 1] * X^T [1, 2] -> [3, 2]
        // [[0.1],   * [[10.0, 20.0]] = [[1.0, 2.0],
        //  [0.2],                      [2.0, 4.0],
        //  [0.3]]                      [3.0, 6.0]]
        // Oczekiwana macierz dL/dW.
        // W formacie column-major, dane w wektorze to: [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
        let expected_weights_grad =
            Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]).unwrap();
        match &linear.weights.grad {
            Some(wg) => {
                println!("Obliczony gradient wag: {:?}", wg.get_data());
                println!(
                    "Oczekiwany gradient wag: {:?}",
                    expected_weights_grad.get_data()
                );
                assert_eq!(wg.get_data(), expected_weights_grad.get_data());
            }
            None => panic!("Gradient wag (weights_grad) nie został obliczony."),
        }

        // --- 3b. Sprawdzenie gradientu biasu (dL/dB) ---
        // Wzór: dL/dB = dL/dY
        let expected_bias_grad = grad_output;
        match &linear.bias.grad {
            Some(bg) => {
                assert_eq!(bg.get_data(), expected_bias_grad.get_data());
            }
            None => panic!("Gradient biasu (bias_grad) nie został obliczony."),
        }

        // --- 3c. Sprawdzenie gradientu wejścia (dL/dX) ---
        // Wzór: dL/dX = W^T * dL/dY
        // W^T [2, 3] * dL/dY [3, 1] -> [2, 1]
        // W^T = [[1.0, 2.0, 3.0],
        //        [4.0, 5.0, 6.0]]
        // [[1, 2, 3],   * [[0.1], = [[1*0.1 + 2*0.2 + 3*0.3], = [[0.1 + 0.4 + 0.9], = [[1.4],
        //  [4, 5, 6]]      [0.2],    [4*0.1 + 5*0.2 + 6*0.3]]    [0.4 + 1.0 + 1.8]]    [3.2]]
        // Oczekiwana macierz dL/dX.
        // W formacie column-major, dane w wektorze to: [1.4, 3.2]
        let expected_downstream_grad = Tensor::new(vec![2, 1], vec![1.4, 3.2]).unwrap();
        println!(
            "Obliczony gradient wejścia: {:?}",
            downstream_grad.get_data()
        );
        println!(
            "Oczekiwany gradient wejścia: {:?}",
            expected_downstream_grad.get_data()
        );
        assert_eq!(
            downstream_grad.get_data(),
            expected_downstream_grad.get_data()
        );

        println!("Test `Module for Linear::backward` zakończony sukcesem!");
    }

    #[test]
    fn test_linear_forward_batch_broadcasting() {
        use crate::layer::Module;

        // 1. Setup: Warstwa Linear (2 wejścia -> 3 wyjścia)
        // Wagi: [[1, 1], [2, 2], [3, 3]] (w układzie column-major)
        let weights = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();

        // Bias: [[10], [20], [30]] - to będzie broadcastowane
        let bias = Tensor::new(vec![3, 1], vec![10.0, 20.0, 30.0]).unwrap();

        let mut linear = Linear::new(weights, bias, None).unwrap();

        // 2. Input: Batch wielkości 2
        // Próbka 1 (kolumna 0): [1, 0]
        // Próbka 2 (kolumna 1): [0, 1]
        // Kształt: [In, Batch] = [2, 2]
        let input = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        // 3. Forward
        // Oczekujemy, że wewnątrz zadziała: (Weights * Input) + Bias_Broadcasted
        let output = linear.forward(input, false).unwrap();

        // 4. Weryfikacja
        // Oczekiwany kształt: [Out, Batch] = [3, 2]
        assert_eq!(output.shape(), &[3, 2]);

        // Obliczenia ręczne:
        // Próbka 1 (W * [1, 0] + b):
        // [1*1 + 1*0] + 10 = 1 + 10 = 11
        // [2*1 + 2*0] + 20 = 2 + 20 = 22
        // [3*1 + 3*0] + 30 = 3 + 30 = 33

        // Próbka 2 (W * [0, 1] + b):
        // [1*0 + 1*1] + 10 = 1 + 10 = 11
        // [2*0 + 2*1] + 20 = 2 + 20 = 22
        // [3*0 + 3*1] + 30 = 3 + 30 = 33

        let expected_data = vec![
            11.0, 22.0, 33.0, // Kolumna 1
            11.0, 22.0, 33.0, // Kolumna 2
        ];

        assert_eq!(output.get_data(), &expected_data);
        println!("Linear batch broadcasting works!");
    }
}
