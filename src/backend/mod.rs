use core::{
    clone::Clone,
    marker::{Copy, Sync},
};

pub mod error;
use crate::{backend::error::BackendError, tensor_number::TensorNumber, tensor2::layout::Layout};

/// Trait for selecting supported number types per backend.
pub trait SupportedBy<B: Backend>: TensorNumber {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;
}

pub trait SupportedTensorNumber<B>: TensorNumber + SupportedBy<B>
where
    B: Backend,
{
}

impl<T, B> SupportedTensorNumber<B> for T
where
    T: TensorNumber + SupportedBy<B>,
    B: Backend,
{
}

pub trait Backend: Clone + Copy + Sync + Send {
    /// Raw buffer for data
    type Storage<T>: Clone + Sync
    where
        T: SupportedTensorNumber<Self>;

    fn get_data_as_vector<T>(data: &Self::Storage<T>) -> Vec<T>
    where
        T: SupportedTensorNumber<Self>;

    fn create_storage_from_vec<T>(data: Vec<T>) -> Self::Storage<T>
    where
        T: SupportedTensorNumber<Self>;

    fn create_storage_from_slice<T>(data: &[T]) -> Self::Storage<T>
    where
        T: SupportedTensorNumber<Self>;

    fn zeros<T>(size: usize) -> Self::Storage<T>
    where
        T: SupportedTensorNumber<Self>;
    fn ones<T>(size: usize) -> Self::Storage<T>
    where
        T: SupportedTensorNumber<Self>;
    unsafe fn uninit<T>(size: usize) -> Self::Storage<T>
    where
        T: SupportedTensorNumber<Self>;
    //random!
}

pub trait BasicOperations<T: SupportedBy<Self>>: Backend {
    fn add<const D: usize>(
        lhs: &Self::Storage<T>,
        lhs_layout: Layout<D>,
        rhs: &Self::Storage<T>,
        rhs_layout: Layout<D>,
    ) -> Result<Self::Storage<T>, BackendError>;
    fn sub<const D: usize>(
        lhs: &Self::Storage<T>,
        lhs_layout: Layout<D>,
        rhs: &Self::Storage<T>,
        rhs_layout: Layout<D>,
    ) -> Result<Self::Storage<T>, BackendError>;
    fn mul_elementwise<const D: usize>(
        lhs: &Self::Storage<T>,
        lhs_layout: Layout<D>,
        rhs: &Self::Storage<T>,
        rhs_layout: Layout<D>,
    ) -> Result<Self::Storage<T>, BackendError>;
}

pub trait ElementWiseOperations<T: SupportedBy<Self>>: Backend {
    fn apply_unary<F>(input: &Self::Storage<T>, f: F) -> Self::Storage<T>
    where
        F: Fn(T) -> T + Sync + Send;

    fn apply_binary<F>(lhs: &Self::Storage<T>, rhs: &Self::Storage<T>, f: F) -> Self::Storage<T>
    where
        F: Fn(T, T) -> T + Sync + Send;
}

pub trait MatmulOperations<T: SupportedBy<Self>>: Backend {
    fn matmul(
        &self,
        lhs: &Self::Storage<T>,
        lhs_layout: Layout<2>,
        rhs: &Self::Storage<T>,
        rhs_layout: Layout<2>,
    ) -> Result<Self::Storage<T>, BackendError>;

    fn batched_matmul(
        &self,
        lhs: &Self::Storage<T>,
        lhs_layout: Layout<3>,
        rhs: &Self::Storage<T>,
        rhs_layout: Layout<3>,
    ) -> Result<Self::Storage<T>, BackendError>;
}
