use core::marker::Sync;

use crate::backend::{Backend, DType, SupportedBy};

pub struct FaerBackend;

impl Backend for FaerBackend {
    type Device = ();
    type Storage<T>
        = T
    where
        T: DType + Sync + SupportedBy<Self>;
}

impl SupportedBy<FaerBackend> for f32 {}
impl SupportedBy<FaerBackend> for f64 {}
