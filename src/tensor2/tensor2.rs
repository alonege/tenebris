use crate::{
    backend::{self, Backend},
    tensor_number::TensorNumber,
    tensor2::layout,
    tensorid::TensorId,
};

pub struct Tensor2<B, T, const D: usize>
where
    B: Backend,
    T: TensorNumber + backend::SupportedBy<B>,
{
    pub(super) id: TensorId,
    pub(super) storage: B::Storage<T>,
    pub(super) layout: layout::Layout<D>,
}

impl<B, T, const D: usize> Tensor2<B, T, D>
where
    B: Backend,
    T: TensorNumber + backend::SupportedBy<B>,
{
    pub fn id(&self) -> &TensorId {
        &self.id
    }

    pub fn storage(&self) -> &B::Storage<T> {
        &self.storage
    }

    pub fn layout(&self) -> &layout::Layout<D> {
        &self.layout
    }
}
