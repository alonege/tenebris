use crate::{
    error::LibError,
    tensor::{
        tensor::Tensor,
        view::{TensorIter, TensorView, iter},
    },
};
use faer::{MatRef, Par, linalg::matmul::matmul};
use std::{marker::PhantomData, num::NonZero};

use crate::tensor::tensor::TensorFloat;

#[derive(Debug, Clone)]
pub struct TensorRef<'a, T: TensorFloat> {
    pub(super) data: *const T,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<isize>,
    pub(super) _marker: PhantomData<&'a T>,
}

impl<'a, T: TensorFloat> TensorRef<'a, T> {
    pub unsafe fn new(data: *const T, shape: Vec<usize>, strides: Vec<isize>) -> Self {
        Self {
            data,
            shape,
            strides,
            _marker: PhantomData,
        }
    }

    pub fn data(&self) -> *const T {
        self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /*
    pub fn as_faer_ref(&self) -> Result<MatRef<'_, T>, String> {
        if self.shape.len() != 2 {
            return Err("Tensor have to be 2-dimensional to change it to faer ref".to_string());
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];
        Ok(MatRef::from_column_major_slice(&self.data, nrows, ncols))
    }
    */

    pub fn as_faer_ref_unsafe(&self) -> Result<MatRef<'_, T>, String> {
        if self.shape.len() != 2 {
            return Err("Tensor have to be 2-dimensional to change it to faer ref".to_string());
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];
        unsafe {
            Ok(MatRef::from_raw_parts(
                self.data,
                nrows,
                ncols,
                self.strides[0],
                self.strides[1],
            ))
        }
    }

    pub fn matmul(&self, b: TensorRef<'a, T>) -> Result<Tensor<T, 2>, String> {
        //let b = b.as_ref();

        if self.shape.len() != 2 || b.shape.len() != 2 {
            return Err("Both tensors have to be 2-dimensional".to_string());
        }
        if self.shape[1] != b.shape[0] {
            return Err("Tensors have to be the same size!".to_string());
        }

        let mut c: Tensor<T, 2> = Tensor::zeros([self.shape()[0], b.shape()[1]]).unwrap();
        println!("{c:?}");
        let c_faer = c.as_faer_mut_unsafe().unwrap();
        let a_faer = self.as_faer_ref_unsafe().unwrap();
        let b_faer = b.as_faer_ref_unsafe().unwrap();

        matmul(
            c_faer,
            faer::Accum::Add,
            a_faer,
            b_faer,
            T::one(),
            Par::Rayon(
                NonZero::new(num_cpus::get_physical().max(1))
                    .expect("this should never happen - library couldn't detect any CPU cores"),
            ),
        );
        Ok(c)
    }

    pub fn rotate_180(&self) -> Self {
        let mut new_strides = self.strides.clone();
        for stride in new_strides.iter_mut() {
            *stride = -*stride;
        }
        let mut new_ptr = self.data;
        let mut offset = 0;
        for (dim, stride) in self.shape.iter().zip(self.strides.iter()) {
            offset += (dim - 1) * (*stride as usize);
        }
        new_ptr = unsafe { new_ptr.add(offset) };
        Self {
            data: new_ptr,
            shape: self.shape.clone(),
            strides: new_strides,
            _marker: self._marker.clone(),
        }
    }

    pub fn get(&self, indices: impl AsRef<[usize]>) -> Result<T, LibError> {
        let indices = indices.as_ref();
        if indices.len() != self.shape.len() {
            return Err(LibError::ShapeMismatch {
                operation: "TensorRef::get".to_string(),
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

    pub fn iter(&'a self) -> TensorIter<'a, Self> {
        iter(&self)
    }

    pub fn to_owned<const D: usize>(&self) -> Result<Tensor<T, D>, LibError> {
        if self.shape.len() != D {
            return Err(LibError::InvalidDimensionality {
                operation: "TensorRef::to_owned".to_string(),
                expected: D,
                actual: self.shape.len(),
            });
        }

        let data: Vec<T> = self.iter().collect();

        let mut shape_arr = [0usize; D];
        shape_arr.copy_from_slice(&self.shape);

        Tensor::new(shape_arr, data)
    }

    /*
    pub fn as_faer_mut(&mut self) -> Result<MatMut<'_, f32>, String> {
        if self.shape.len() != 2 {
            return Err("Tensors have to be 2-dimensional to change it to faer ref".to_string());
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];

        Ok(MatMut::from_column_major_slice_mut(
            &mut self.data,
            nrows,
            ncols,
        ))
    }

    pub fn as_faer_mut_unsafe(&mut self) -> Result<MatMut<'_, f32>, String> {
        if self.shape.len() != 2 {
            return Err("Tensors have to be 2-dimensional to change it to faer ref".to_string());
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];

        unsafe {
            Ok(MatMut::from_raw_parts_mut(
                self.data,
                nrows,
                ncols,
                self.strides[0],
                self.strides[1],
            ))
        }
    }
    */
}

impl<'a, T: TensorFloat> TensorView<'a> for TensorRef<'a, T> {
    type Dtype = T;

    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[isize] {
        &self.strides
    }
    fn data_ptr(&self) -> *const T {
        self.data
    }
}

#[allow(unused_imports)]
pub mod tests {
    use faer::Par;
    use std::num::NonZero;

    use crate::tensor::tensor::Tensor;
    #[test]
    pub fn matmul_reversed() {
        let tensor_a =
            crate::tensor::tensor::Tensor::new_row([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let tensor_b =
            crate::tensor::tensor::Tensor::new_row([3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                .unwrap();

        let tensor_b_ref = tensor_b.as_ref().rotate_180();

        let a_faer = tensor_a.as_faer_ref_unsafe().unwrap();
        let b_faer = tensor_b_ref.as_faer_ref_unsafe().unwrap();
        let mut tensor_c = crate::tensor::tensor::Tensor::zeros([2, 2]).unwrap();
        let mut c_faer = tensor_c.as_faer_mut_unsafe().unwrap();
        faer::linalg::matmul::matmul(
            &mut c_faer,
            faer::Accum::Replace,
            a_faer,
            b_faer,
            1.0,
            Par::Rayon(NonZero::new(16).unwrap()),
        );
        println!("tensor_c: {}", tensor_c);
    }
    #[test]
    fn test_to_tensor_from_contiguous_ref() {
        // 1. Stwórz oryginalny, posiadający dane Tensor
        let original_tensor = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // 2. Stwórz z niego prosty TensorRef (widok)
        let tensor_ref = original_tensor.as_ref();

        // 3. Użyj testowanej metody, aby stworzyć nowy, posiadający dane Tensor
        let new_tensor = tensor_ref.to_owned().unwrap();

        // 4. Sprawdź, czy nowy Tensor jest identyczny z oryginałem
        assert_eq!(new_tensor.shape(), original_tensor.shape());
        assert_eq!(new_tensor.get_data(), original_tensor.get_data());
        // Nowy tensor powinien być ciągły
        assert_eq!(new_tensor.strides(), &[1, 2]);
    }

    #[test]
    fn test_to_tensor_from_non_contiguous_ref() {
        // 1. Stwórz oryginalny Tensor
        let original_tensor = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        println!("Original Tensor: {:?}", original_tensor);

        // 2. Stwórz widok nieciągły (transponowany)
        // Logicznie wygląda tak: [[1.0, 3.0], [2.0, 4.0]]
        let transposed_tensor = original_tensor.t();
        println!("Transposed Tensor: {:?}", transposed_tensor);
        let tensor_ref = transposed_tensor.as_ref();
        println!("TensorRef: {:?}", tensor_ref);

        // 3. Użyj testowanej metody
        let new_tensor = tensor_ref.to_owned().unwrap();
        println!("New Tensor from TensorRef: {:?}", new_tensor);

        // 4. Sprawdź, czy nowy Tensor ma poprawny (transponowany) kształt
        assert_eq!(new_tensor.shape(), &[2, 2]);
        // Nowy tensor powinien być teraz ciągły w pamięci
        assert_eq!(new_tensor.strides(), &[1, 2]);

        // 5. Sprawdź, czy dane w nowym, ciągłym tensorze są ułożone
        // w poprawnym porządku kolumnowym dla macierzy transponowanej.
        // Iterator powinien odczytać elementy w porządku: [0,0], [1,0], [0,1], [1,1]
        // co odpowiada wartościom: 1.0, 3.0, 2.0, 4.0
        let expected_data = vec![1.0, 3.0, 2.0, 4.0];
        assert_eq!(new_tensor.get_data(), &expected_data);
    }
}
