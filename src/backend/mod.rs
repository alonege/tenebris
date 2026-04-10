use core::{
    clone::Clone,
    default::Default,
    marker::{Copy, Sync},
};

pub mod faer_backend;

pub trait DType: Default + Copy {}
impl DType for f32 {}
impl DType for f64 {}

pub trait Backend {
    /// Device representation
    type Device: Default + Clone;

    /// Raw buffer for data
    type Storage<T>: Clone + Copy + Sync
    where
        T: DType + Sync + SupportedBy<Self>;
}

pub trait SupportedBy<B: Backend>: DType {}
