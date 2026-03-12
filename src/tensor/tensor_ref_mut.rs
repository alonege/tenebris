use std::marker::PhantomData;

use faer::{MatMut, MatRef};

use crate::{error::LibError, tensor::tensor::TensorFloat};

pub struct TensorRefMut<'a, T: TensorFloat> {
    pub(crate) data: *mut T,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) _marker: PhantomData<&'a mut T>,
}

impl<'a, T> TensorRefMut<'a, T>
where
    T: TensorFloat,
{
    pub unsafe fn new(data: *mut T, shape: Vec<usize>, strides: Vec<isize>) -> Self {
        TensorRefMut {
            data,
            shape,
            strides,
            _marker: PhantomData,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn select(mut self, axis: usize, index: usize) -> Result<Self, LibError> {
        // let's validate if we can perform op
        if axis >= self.shape.len() {
            return Err(LibError::InvalidDimensionality {
                operation: "TensorMutRef::select".to_string(),
                expected: axis + 1,
                actual: self.shape.len(),
            });
        }
        if index >= self.shape[axis] {
            return Err(LibError::OutOfBounds {
                shape: self.shape.clone(),
                index: {
                    let mut idx = vec![0; self.shape.len()];
                    idx[axis] = index;
                    idx
                },
            });
        }

        let offset = self.strides[axis] * index as isize;
        self.data = unsafe { self.data.offset(offset) };

        self.shape.remove(axis);
        self.strides.remove(axis);

        if self.shape.is_empty() {
            self.shape.push(1);
            self.strides.push(1);
        }

        Ok(self)
    }

    pub fn get(&self, indices: impl AsRef<[usize]>) -> Result<T, LibError> {
        let indices = indices.as_ref();
        if indices.len() != self.shape.len() {
            return Err(LibError::ShapeMismatch {
                operation: "TensorRefMut::get".to_string(),
                expected: self.shape.clone(),
                actual: indices.to_vec(),
            });
        }
        let mut offset = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape[i] {
                return Err(LibError::OutOfBounds {
                    shape: self.shape.clone(),
                    index: indices.to_vec(),
                });
            }
            offset += index as isize * self.strides[i];
        }
        unsafe { Ok(*self.data.offset(offset)) }
    }

    pub fn get_mut(&mut self, indices: impl AsRef<[usize]>) -> Result<&mut T, LibError> {
        let indices = indices.as_ref();
        if indices.len() != self.shape.len() {
            return Err(LibError::ShapeMismatch {
                operation: "TensorRefMut::get_mut".to_string(),
                expected: self.shape.clone(),
                actual: indices.to_vec(),
            });
        }
        let mut offset = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape[i] {
                return Err(LibError::OutOfBounds {
                    shape: self.shape.clone(),
                    index: indices.to_vec(),
                });
            }
            offset += index as isize * self.strides[i];
        }
        unsafe { Ok(&mut *self.data.offset(offset)) }
    }

    pub fn as_faer_mut(&mut self) -> MatMut<'a, T> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.data,
                self.shape[0],
                self.shape[1],
                self.strides[0],
                self.strides[1],
            )
        }
    }

    pub fn as_faer_ref(&self) -> MatRef<'a, T> {
        unsafe {
            MatRef::from_raw_parts(
                self.data,
                self.shape[0],
                self.shape[1],
                self.strides[0],
                self.strides[1],
            )
        }
    }
}
