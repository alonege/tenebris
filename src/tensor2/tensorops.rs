use std::ops::Add;

use crate::{
    backend::{self, Backend},
    tensor_number::TensorNumber,
    tensor2::tensor2::Tensor2,
    tensorid::TensorId,
};

impl<B, T, const D: usize> Add<&Tensor2<B, T, D>> for Tensor2<B, T, D>
where
    B: Backend + backend::BasicOperations<T>,
    T: TensorNumber + backend::SupportedBy<B>,
{
    type Output = Self;

    // WARN: check later - written prior to handling id
    // ALSO - change to inplace add for owned if needed
    fn add(self, rhs: &Tensor2<B, T, D>) -> Self::Output {
        let new_storage = B::add(&self.storage, self.layout, &rhs.storage, rhs.layout);
        let new_layout = self.layout; // Assuming the layout remains the same for addition
        Tensor2 {
            id: TensorId::new(self.id.get_id()), // Generate a new ID if needed
            storage: new_storage,
            layout: new_layout,
        }
    }
}

impl<B, T, const D: usize> Add for Tensor2<B, T, D>
where
    B: Backend + backend::BasicOperations<T>,
    T: TensorNumber + backend::SupportedBy<B>,
{
    type Output = Self;

    // WARN: check later - written prior to handling id
    // ALSO - change to inplace add for owned if needed
    fn add(self, rhs: Self) -> Self::Output {
        let new_storage = B::add(&self.storage, self.layout, &rhs.storage, rhs.layout);
        let new_layout = self.layout; // Assuming the layout remains the same for addition
        Tensor2 {
            id: TensorId::new(self.id.get_id()), // Generate a new ID if needed
            storage: new_storage,
            layout: new_layout,
        }
    }
}

impl<B, T, const D: usize> Add for &Tensor2<B, T, D>
where
    B: Backend + backend::BasicOperations<T>,
    T: TensorNumber + backend::SupportedBy<B>,
{
    type Output = Tensor2<B, T, D>;

    // WARN: check later - written prior to handling id
    fn add(self, rhs: Self) -> Self::Output {
        let new_storage = B::add(&self.storage, self.layout, &rhs.storage, rhs.layout);
        let new_layout = self.layout; // Assuming the layout remains the same for addition
        Tensor2 {
            id: TensorId::new(self.id.get_id()), // Generate a new ID if needed
            storage: new_storage,
            layout: new_layout,
        }
    }
}
