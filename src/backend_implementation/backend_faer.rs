use std::{num::NonZero, sync::Arc};

use faer::{
    Par,
    traits::{ComplexField, Conjugate},
};
use tracing::{Level, event};

use crate::{
    backend::{Backend, MatmulOperations, SupportedBy, error::BackendError},
    tensor_number::TensorNumber,
};

#[derive(Copy, Clone)]
pub struct FaerBackend {
    parallelism: Par,
}

impl FaerBackend {
    pub fn new() -> Self {
        FaerBackend {
            parallelism: Par::Rayon(NonZero::new(num_cpus::get_physical().max(1)).unwrap_or(
                NonZero::new(1).expect(
                    "CRITICAL ERROR: NonZero::new(1) returned None, which should never happen.",
                ),
            )),
        }
    }
    pub fn set_parallelism(mut self, parallelism: Par) -> Self {
        self.parallelism = parallelism;
        self
    }
    pub fn parallelism(&self) -> Par {
        self.parallelism
    }
}

impl SupportedBy<FaerBackend> for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl SupportedBy<FaerBackend> for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

pub trait FaerNum:
    SupportedBy<FaerBackend> + Conjugate + ComplexField + Copy + Send + Sync
{
}
impl FaerNum for f32 {}
impl FaerNum for f64 {}

impl Backend for FaerBackend {
    type Storage<T>
        = Arc<Vec<T>>
    where
        T: TensorNumber + SupportedBy<Self>;

    fn get_data_as_vector<T>(data: &Self::Storage<T>) -> Vec<T>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        data.to_vec()
    }

    fn create_storage_from_vec<T>(data: Vec<T>) -> Self::Storage<T>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        Arc::new(data)
    }

    fn create_storage_from_slice<T>(data: &[T]) -> Self::Storage<T>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        Arc::new(data.to_vec())
    }

    fn zeros<T>(size: usize) -> Self::Storage<T>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        Arc::new(vec![T::zero(); size])
    }

    fn ones<T>(size: usize) -> Self::Storage<T>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        Arc::new(vec![T::one(); size])
    }
    unsafe fn uninit<T>(size: usize) -> Self::Storage<T>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        let mut vec: Vec<T> = Vec::with_capacity(size);
        unsafe {
            vec.set_len(size);
        }
        Arc::new(vec)
    }
}

impl<T: FaerNum> MatmulOperations<T> for FaerBackend {
    fn matmul(
        &self,
        lhs: &Self::Storage<T>,
        lhs_layout: crate::tensor2::layout::Layout<2>,
        rhs: &Self::Storage<T>,
        rhs_layout: crate::tensor2::layout::Layout<2>,
    ) -> Result<Self::Storage<T>, BackendError>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        let output_row_count = lhs_layout.shape()[0];
        let output_col_count = rhs_layout.shape()[1];

        let size = output_row_count * output_col_count;
        let mut result = unsafe {
            let mut vec: Vec<T> = Vec::with_capacity(size);
            vec.set_len(size);
            vec
        };

        let lhs_faer = unsafe {
            faer::MatRef::from_raw_parts(
                lhs.as_ptr(),
                lhs_layout.shape()[0],
                lhs_layout.shape()[1],
                lhs_layout.strides()[0],
                lhs_layout.strides()[1],
            )
        };
        let rhs_faer = unsafe {
            faer::MatRef::from_raw_parts(
                rhs.as_ptr(),
                rhs_layout.shape()[0],
                rhs_layout.shape()[1],
                rhs_layout.strides()[0],
                rhs_layout.strides()[1],
            )
        };
        let mut result_faer = unsafe {
            faer::MatMut::from_raw_parts_mut(
                result.as_mut_ptr(),
                output_row_count,
                output_col_count,
                1,
                output_col_count as isize,
            )
        };

        faer::linalg::matmul::matmul(
            &mut result_faer,
            faer::Accum::Replace,
            lhs_faer,
            rhs_faer,
            T::one(),
            self.parallelism(),
        );

        event!(Level::TRACE, "Result data: {:?}", result.as_slice());

        let result = Self::create_storage_from_vec(result);
        Ok(result)
    }
    fn batched_matmul(
        &self,
        lhs: &Self::Storage<T>,
        lhs_layout: crate::tensor2::layout::Layout<3>,
        rhs: &Self::Storage<T>,
        rhs_layout: crate::tensor2::layout::Layout<3>,
    ) -> Result<Self::Storage<T>, BackendError>
    where
        T: TensorNumber + SupportedBy<Self>,
    {
        // Implement the batched matrix multiplication logic here using the faer library
        // For now, we will just return a placeholder
        return Err(BackendError::Unsupported {
            operation: "Batched matrix multiplication",
        });
    }
}
