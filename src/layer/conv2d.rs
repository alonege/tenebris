// libmlrs/src/layer/conv2d.rs

use crate::initialization::Initialization;
use crate::profile_layer;
use crate::tensor::tensorops::{MatMul, TensorAdd};
use crate::{
    error::LibError,
    layer::{Module, Tensor, TensorFloat},
    tensor::tensorops::TensorConv2D,
};

pub struct Conv2D<T: TensorFloat> {
    pub weights: Tensor<T>,
    //pub bias: Tensor<T>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),

    input_cache: Option<Tensor<T>>,
}

impl<T: TensorFloat> Conv2D<T> {
    pub fn new(
        c_in: usize,
        c_out: usize,
        (kernel_h, kernel_w): (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        init: &dyn Initialization<T>,
    ) -> Result<Self, LibError> {
        let weights = Tensor::new_with_init(
            &[c_in, kernel_w, kernel_h, c_out],
            (c_in * kernel_h * kernel_w, c_out * kernel_h * kernel_w),
            init,
        )?;
        Ok(Self::new_raw(weights, stride, padding))
    }

    pub fn new_raw(
        weights: Tensor<T>,
        //bias: Tensor<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        //let c_out = weights.shape()[3];
        //let reshaped_bias = bias.reshape(vec![c_out, 1, 1, 1]).unwrap();

        Self {
            weights,
            //bias: reshaped_bias,
            stride,
            padding,
            input_cache: None,
        }
    }

    pub fn new_rand(
        c_in: usize,
        c_out: usize,
        (kernel_h, kernel_w): (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let weights = Tensor::random(&[c_in, kernel_w, kernel_h, c_out]);
        Self::new_raw(weights, stride, padding)
    }
}

impl<T: TensorFloat + 'static> Module<T> for Conv2D<T> {
    type Input<A> = Tensor<T>;
    type Output<B> = Tensor<T>;

    #[inline(always)]
    fn forward(
        &mut self,
        input: Self::Input<T>,
        for_backward: bool,
    ) -> Result<Self::Output<T>, LibError> {
        profile_layer!(Self, "Forward", {
            // unpacking of dim data
            let (c_image, w_image, h_image, n) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
            );

            let (c_in, w_kernel, h_kernel, c_out) = (
                self.weights.shape()[0],
                self.weights.shape()[1],
                self.weights.shape()[2],
                self.weights.shape()[3],
            );

            let (padding_h, padding_w) = self.padding;
            let (stride_h, stride_w) = self.stride;

            //check for dim mismatch

            if c_image != c_in {
                return Err(LibError::InvalidDimensionality {
                    operation: "conv2d module forward; bad image input c_in".to_string(),
                    expected: c_in,
                    actual: c_image,
                });
            }

            // calculate output shape

            let h_out = (h_image - h_kernel + 2 * padding_h) / stride_h + 1;
            let w_out = (w_image - w_kernel + 2 * padding_w) / stride_w + 1;

            // save input for backward pass

            if for_backward {
                self.input_cache = Some(input.clone());
            }

            // let's get x_col by using im2col
            // x_col shape = [w_kernel * h_kernel * c_in, w_out * h_out * n]
            let x_col = <crate::tensor::tensor::Tensor<T> as TensorConv2D<
                crate::tensor::tensor::Tensor<T>,
            >>::im2col(
                input.clone(), w_kernel, h_kernel, self.stride, self.padding
            )?;
            //println!("x_col: {}", x_col);

            // and let's get k_mat by using kn2k_mat
            // k_mat shape = [c_out, w_kernel * h_kernel * c_in]
            let k_mat = <crate::tensor::tensor::Tensor<T> as TensorConv2D<
                crate::tensor::tensor::Tensor<T>,
            >>::kn2k_mat(self.weights.clone())?;

            // calculate output
            // output shape = [c_out, w_kernel * h_kernel * c_in] * [w_kernel * h_kernel * c_in, w_out * h_out * n] =
            // = [c_out, w_out * h_out * n]
            let output = k_mat.matmul(&x_col)?;

            //reshape output

            let output = output.reshape(vec![c_out, w_out, h_out, n])?;

            Ok(output)
        })
    }

    fn backward(&mut self, y_out: Self::Input<T>) -> Result<Self::Output<T>, LibError> {
        profile_layer!(Self, "Backward", {
            if self.input_cache.is_none() {
                return Err(LibError::LayerErrorBackwardNoGradient);
            }

            let (c_in, w_kernel, h_kernel, c_out_kernel) = (
                self.weights.shape()[0],
                self.weights.shape()[1],
                self.weights.shape()[2],
                self.weights.shape()[3],
            );

            // shape of y_out is [c_out, w_out, h_out, n]
            let (c_out, w_out, h_out, n) = (
                y_out.shape()[0],
                y_out.shape()[1],
                y_out.shape()[2],
                y_out.shape()[3],
            );

            if c_out != c_out_kernel {
                return Err(LibError::InvalidDimensionality {
                    operation: "conv2d module backward; bad y_out c dimension".to_string(),
                    expected: c_out_kernel,
                    actual: c_out,
                });
            }

            //println!("y_out: {}", y_out);
            let y_out_cont = y_out.make_contiguous()?;
            //println!("y_out_cont: {}", y_out_cont);
            let y_out_reshaped = y_out_cont.reshape(vec![c_out, w_out * h_out * n])?;
            //println!("y_out_reshaped: {}", y_out_reshaped);
            // y_out reshaped to [c_out, w_out * h_out * n]

            // let's calculate dL/dk
            let x_col = <crate::tensor::tensor::Tensor<T> as TensorConv2D<
                crate::tensor::tensor::Tensor<T>,
            >>::im2col(
                self.input_cache.as_ref().unwrap().clone(),
                self.weights.shape()[1],
                self.weights.shape()[2],
                self.stride,
                self.padding,
            )?; // shape of x_col: [w_kernel * h_kernel * c_in, w_out * h_out, n]

            let x_col_t = x_col.t();
            // x_col_t shape: [w_out * h_out * n, w_kernel * h_kernel * c_in]

            let k_grad = y_out_reshaped.matmul(&x_col_t)?;
            // k_grad shape: [c_out, w_out * h_out * n] * [w_out * h_out * n, w_kernel * h_kernel * c_in] =
            // = [c_out, w_kernel * h_kernel * c_in]
            //println!("k_grad: {}", k_grad);

            let k_grad_reshaped = k_grad
                .reshape(vec![c_out, w_kernel, h_kernel, c_in])?
                .permute(&[3, 1, 2, 0]);
            //println!("k_grad_reshaped: {}", k_grad_reshaped);
            // k_grad_reshaped reshaped to [c_out, w_kernel, h_kernel, c_in]
            // k_grad_reshaped then permuted to [c_in, w_kernel, h_kernel, c_out]
            if let Some(existing_weights_grad) = self.weights.grad.take() {
                self.weights.grad =
                    Some(Box::new(existing_weights_grad.tensoradd(&k_grad_reshaped)?));
            } else {
                self.weights.grad = Some(Box::new(k_grad_reshaped));
            }

            // dL/dX = K_mat^T * dL/dY
            let k_mat = <crate::tensor::tensor::Tensor<T> as TensorConv2D<
                crate::tensor::tensor::Tensor<T>,
            >>::kn2k_mat(self.weights.clone())?;
            // k_mat shape: [c_out, w_kernel * h_kernel * c_in]

            let k_mat_t = k_mat.t();
            // k_mat_t shape: [w_kernel * h_kernel * c_in, c_out]

            //println!("k_mat shape: {:?}", k_mat.shape());
            //println!("y_out shape: {:?}", y_out.shape());

            // grad_x_col = k_mat^T * y_out_reshaped
            let grad_x_col = k_mat_t.matmul(&y_out_reshaped)?;
            // grad_x_col shape: [w_kernel * h_kernel * c_in, c_out] * [c_out, w_out * h_out * n] =
            // = [w_kernel * h_kernel * c_in, w_out * h_out * n]
            //println!("grad_x_col {}", grad_x_col);

            // grad_x = col2im(grad_x_col)
            // grad_x shape = [c, w, h, n]
            let grad_x = <crate::tensor::tensor::Tensor<T> as TensorConv2D<
                crate::tensor::tensor::Tensor<T>,
            >>::col2im(
                grad_x_col.clone(),
                (
                    self.input_cache.as_ref().unwrap().shape()[0],
                    self.input_cache.as_ref().unwrap().shape()[1],
                    self.input_cache.as_ref().unwrap().shape()[2],
                    self.input_cache.as_ref().unwrap().shape()[3],
                ),
                w_kernel,
                h_kernel,
                self.stride,
                self.padding,
            )?;

            Ok(grad_x)
        })
    }

    #[inline(always)]
    fn parameters(&self) -> Vec<Tensor<T>> {
        //vec![self.weights.clone(), self.bias.clone()]
        vec![self.weights.clone()]
    }

    #[inline(always)]
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        //vec![&mut self.weights, &mut self.bias]
        vec![&mut self.weights]
    }

    #[inline(always)]
    fn clear_grad(&mut self) {
        self.input_cache = None;
    }
}

#[cfg(test)]
mod tests {
    use crate::layer::{Module, conv2d::Conv2D};
    use crate::tensor::tensor::Tensor;

    /// Helper function to create a consistent layer for tests.
    fn setup_layer() -> Conv2D<f32> {
        let weights = Tensor::new(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        println!("Weights: {}", weights);

        // Bias is ignored for now.
        //let bias = Tensor::zeros(&[1]).unwrap(); // Minimal valid bias

        Conv2D::new_raw(weights, (1, 1), (0, 0))
    }

    // calculated manually
    #[test]
    fn test_conv2d_module_forward() {
        // --- Setup ---
        let mut conv_layer = setup_layer();
        // Input shape: [C, W, H, N] = [1, 3, 3, 1]
        let input = Tensor::new(
            vec![1, 3, 3, 1],
            (1..=9).map(|x| x as f32).collect(), // Stored column-major
        )
        .unwrap();

        // --- Action ---
        let output = conv_layer.forward(input, true).unwrap();

        // --- Assert ---
        // Output shape according to your conv2d implementation: [C_out, W_out, H_out, N]
        let expected_shape = vec![1, 2, 2, 1];
        assert_eq!(output.shape(), &expected_shape);

        // Manually calculated convolution result for a column-major tensor
        // with (C, W, H, N) layout.
        // Output layout is also column-major: [out(W0,H0), out(W0,H1), out(W1,H0), out(W1,H1)]
        let expected_data = vec![37.0, 47.0, 67.0, 77.0];
        assert_eq!(output.get_data(), &expected_data);
    }

    // calculated manually
    #[test]
    fn test_conv2d_module_forward_c_in2() {
        let image = Tensor::new(
            vec![2, 5, 5, 1],
            vec![
                0.0, 25.0, 1.0, 26.0, 2.0, 27.0, 3.0, 28.0, 4.0, 29.0, 5.0, 30.0, 6.0, 31.0, 7.0,
                32.0, 8.0, 33.0, 9.0, 34.0, 10.0, 35.0, 11.0, 36.0, 12.0, 37.0, 13.0, 38.0, 14.0,
                39.0, 15.0, 40.0, 16.0, 41.0, 17.0, 42.0, 18.0, 43.0, 19.0, 44.0, 20.0, 45.0, 21.0,
                46.0, 22.0, 47.0, 23.0, 48.0, 24.0, 49.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);

        let kernel = Tensor::new(
            vec![2, 3, 3, 2], // [H_f, W_f, C_in, C_out]
            vec![
                // row, column
                // out channel 0
                0.0, 0.0, // 0, 0
                1.0, 0.0, // 0, 1
                0.0, 1.0, // 0, 2
                0.0, 0.0, // 1, 0
                1.0, 0.0, // 1, 1
                0.0, 0.0, // 1, 2
                0.0, 0.0, // 2, 0
                0.0, 0.0, // 2, 1
                0.0, 0.0, // 2, 2
                //0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                //0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

                // out channel 1
                0.0, 0.0, // 0, 0
                0.0, 0.0, // 0, 1
                0.0, 0.0, // 0, 2
                0.0, 0.0, // 1, 0
                0.0, 1.0, // 1, 1
                0.0, 0.0, // 1, 2
                0.0, 0.0, // 2, 0
                0.0, 0.0, // 2, 1
                0.0,
                0.0, // 2, 2
                     //0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     //0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        println!("Kernel: {}", kernel);

        let mut conv2d = Conv2D::new_raw(kernel.clone(), (1, 1), (0, 0));

        let out = conv2d.forward(image.clone(), false).unwrap();

        println!("Output: {}", out);
        let out_expected = Tensor::new(
            vec![2, 3, 3, 1], // [C_out, W_out, H_out]
            vec![
                34.0, 31.0, 37.0, 32.0, 40.0, 33.0, 49.0, 36.0, 52.0, 37.0, 55.0, 38.0, 64.0, 41.0,
                67.0, 42.0, 70.0, 43.0,
            ],
        )
        .unwrap();

        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
    }

    // hand written test for 2 input channels and 3 output channels
    // calculated manually
    #[test]
    fn test_conv2d_module_forward_multi_channel() {
        // Obraz wejściowy o kształcie [C, W, H], czyli [1, 5, 5]
        let image = Tensor::new(
            vec![2, 5, 5, 1],
            vec![
                0.0, 25.0, 1.0, 26.0, 2.0, 27.0, 3.0, 28.0, 4.0, 29.0, 5.0, 30.0, 6.0, 31.0, 7.0,
                32.0, 8.0, 33.0, 9.0, 34.0, 10.0, 35.0, 11.0, 36.0, 12.0, 37.0, 13.0, 38.0, 14.0,
                39.0, 15.0, 40.0, 16.0, 41.0, 17.0, 42.0, 18.0, 43.0, 19.0, 44.0, 20.0, 45.0, 21.0,
                46.0, 22.0, 47.0, 23.0, 48.0, 24.0, 49.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);

        let kernel = Tensor::new_row(
            vec![3, 3, 2, 3], // [H_f, W_f, C_in, C_out]
            vec![
                // channel out 0
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // channel in 0
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // channel in 1
                // channel out 1
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // channel in 0
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // channel in 1
                // channel out 2
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // channel in 0
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // channel in 1
            ],
        )
        .unwrap()
        .permute(&[2, 1, 0, 3])
        .make_contiguous()
        .unwrap();
        println!("Kernel: {}", kernel);

        let mut conv2d = Conv2D::new_raw(kernel.clone(), (1, 1), (0, 0));

        let out = conv2d.forward(image.clone(), false).unwrap();

        println!("Output: {}", out);
        let out_expected = Tensor::new(
            vec![3, 3, 3, 1], // [C_out, W_out, H_out]
            vec![
                // every row is one pixel's channels
                // row 0
                34.0, 31.0, 5.0, // column 0
                37.0, 32.0, 6.0, // column 1
                40.0, 33.0, 7.0, // column 2
                // row 1
                49.0, 36.0, 10.0, // column 0
                52.0, 37.0, 11.0, // column 1
                55.0, 38.0, 12.0, // column 2
                // row 2
                64.0, 41.0, 15.0, // column 0
                67.0, 42.0, 16.0, // column 1
                70.0, 43.0, 17.0, // column 2
            ],
        )
        .unwrap();

        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
    }

    #[test]
    fn test_conv2d_module_strides_padding_multchannels() {
        let weights = Tensor::new(
            vec![2, 2, 2, 3],
            vec![
                //channel out 0
                1.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, //
                //channel out 1
                0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 1.0, //
                //channel out 2
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
        )
        .unwrap();

        let mut conv_layer = Conv2D::new_raw(weights, (2, 2), (1, 1));

        let image = Tensor::new(
            vec![2, 3, 3, 1],
            vec![
                1.0, 10.0, 2.0, 11.0, 3.0, 12.0, 4.0, 13.0, 5.0, 14.0, 6.0, 15.0, //
                7.0, 16.0, 8.0, 17.0, 9.0, 18.0,
            ],
        );

        let output = conv_layer.forward(image.unwrap(), false).unwrap();
        println!("Output: {}", output);
        assert_eq!(output.shape(), vec![3, 2, 2, 1].as_slice());
    }

    #[test]
    fn test_conv2d_module_backward_dx() {
        // --- Setup ---
        let mut conv_layer = setup_layer();
        // Input shape: [C, W, H, N] = [1, 3, 3, 1]
        let input = Tensor::new(vec![1, 3, 3, 1], (1..=9).map(|x| x as f32).collect()).unwrap();
        // Perform a forward pass to cache the input shape.
        let _ = conv_layer.forward(input.clone(), true).unwrap();

        println!("-------------------------BACKWARD----------------------");
        // Upstream gradient (dY), shape: [C_out, W_out, H_out, N] = [1, 2, 2, 1]
        let grad_output =
            Tensor::new_row(vec![1, 2, 2, 1], vec![0.0, 10.0, 100.0, 1000.0]).unwrap();

        // --- Action ---
        // Calculate the gradient with respect to the input (dX)
        let d_input = conv_layer.backward(grad_output).unwrap();

        // --- Assert ---
        // 1. The primary check: dX must have the same shape as the original input X.
        assert_eq!(d_input.shape(), input.shape());

        // 2. Check the data. This is the result of a "full convolution" of grad_output
        //    with the spatially flipped (180-degree rotated) weights.
        // dX layout is column-major: [dX(W0,H0..2), dX(W1,H0..2), dX(W2,H0..2)]
        // TODO: check it
        let expected_data = vec![
            0.0, 10.0, 20.0, // Column for W=0
            100.0, 1230.0, 2040.0, // Column for W=1
            300.0, 3400.0, 4000.0, // Column for W=2
        ];
        assert_eq!(d_input.get_data(), &expected_data);
    }

    #[test]
    fn test_identity_conv() {
        // Kernel który powinien zwrócić input bez zmian
        let mut conv = Conv2D::<f32>::new_rand(1, 1, (1, 1), (1, 1), (0, 0));
        conv.weights[&[0]] = 1.0; // Identity kernel

        let input = Tensor::new(
            vec![1, 3, 3, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();

        let output = conv.forward(input.clone(), true).unwrap();

        // Output powinien być równy input (stride=1, padding=0, kernel=1x1)
        assert_eq!(input.get_data(), output.get_data());

        // Test backward
        let grad = output.clone();
        let grad_input = conv.backward(grad).unwrap();

        // Gradient input powinien też być równy output
        assert_eq!(grad_input.get_data(), output.get_data());
    }
}
