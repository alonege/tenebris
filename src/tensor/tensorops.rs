use crate::error::LibError;

pub trait TensorOps {}

pub trait MatMul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

pub trait TensorAdd<Rhs> {
    type Output;
    fn tensoradd(&self, rhs: &Rhs) -> Self::Output;
}

pub trait TensorConv2D<T> {
    #[deprecated]
    fn conv2d(
        &self,
        images: T,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<T, LibError>;
    fn im2col(
        images: T,
        w_kernel: usize,
        h_kernel: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<T, LibError>;
    fn kn2k_mat(filters: T) -> Result<T, LibError>;
    fn col2im(
        x_col: T,
        image_shape: (usize, usize, usize, usize), // (C_in, W_in, H_in, N)
        w_kernel: usize,
        h_kernel: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<T, LibError>;
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
