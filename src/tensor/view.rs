use crate::tensor::tensor::TensorFloat;

/// Wewnętrzny trait, który abstrahuje nad strukturami zachowującymi się jak tensor.
/// Internakl trait, which abstract over Tensor - like structs
pub trait TensorView<'a> {
    type Dtype: TensorFloat;

    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn data_ptr(&self) -> *const Self::Dtype;

    fn physical_offset(&self, logical_idx: &[usize]) -> isize {
        logical_idx
            .iter()
            .zip(self.strides().iter())
            .fold(0isize, |acc, (&ind, &stride)| acc + ind as isize * stride)
    }
}

use std::marker::PhantomData;

pub struct TensorIter<'a, V: TensorView<'a>> {
    view: &'a V,
    logical_pos: Vec<usize>,
    finished: bool,
    _marker: PhantomData<&'a V::Dtype>,
}

/// Method to create an iterator
pub fn iter<'a, V>(view: &'a V) -> TensorIter<'a, V>
where
    V: TensorView<'a> + Clone,
{
    let view_clone = view.clone();
    let finished = view.shape().iter().any(|&d| d == 0);
    TensorIter {
        view,
        logical_pos: vec![0; view_clone.shape().len()],
        finished,
        _marker: PhantomData,
    }
}

impl<'a, V> Iterator for TensorIter<'a, V>
where
    V: TensorView<'a>,
{
    type Item = V::Dtype;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let offset = self.view.physical_offset(&self.logical_pos);
        let item = unsafe { *self.view.data_ptr().offset(offset) };

        // position increment logic for column major
        let mut i = 0;
        while i < self.view.shape().len() {
            self.logical_pos[i] += 1;
            if self.logical_pos[i] < self.view.shape()[i] {
                return Some(item);
            }
            self.logical_pos[i] = 0;
            i += 1;
        }

        self.finished = true;
        Some(item)
    }
}
