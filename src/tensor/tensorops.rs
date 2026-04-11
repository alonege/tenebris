use crate::{
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub trait TensorOps {}

pub trait MatMul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

pub trait TensorAdd<Rhs> {
    type Output;
    fn tensoradd(&self, rhs: &Rhs) -> Self::Output;
}

pub trait TensorConv2D<T: TensorFloat> {
    #[deprecated(note = "Use im2col and GEMM matmul instead for better performance")]
    fn conv2d(
        &self, // Expected to be implemented on the filter tensor: [C_out, C_in, W_k, H_k]
        images: Tensor<T, 4>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor<T, 4>, LibError>;

    fn im2col(
        images: Tensor<T, 4>,
        w_kernel: usize,
        h_kernel: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor<T, 2>, LibError>;
    fn kn2k_mat(filters: Tensor<T, 4>) -> Result<Tensor<T, 2>, LibError>;

    fn col2im(
        x_col: Tensor<T, 2>,
        image_shape: (usize, usize, usize, usize),
        w_kernel: usize,
        h_kernel: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor<T, 4>, LibError>;
}

pub trait TensorConv2DExperimental<Image> {
    type Output;
    fn conv2d_experimental(
        &self,
        images: impl AsRef<[Image]>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self::Output;
}
