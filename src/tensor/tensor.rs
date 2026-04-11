use crate::error::LibError;
use crate::initialization::Initialization;
use crate::tensor::tensor_ref_mut::TensorRefMut;
use crate::tensor::tensorops::{MatMul, TensorAdd};
use crate::tensor::tensorref::TensorRef;
use crate::tensor::view::{TensorIter, TensorView, iter};
use faer::prelude::*;
use faer::traits::{ComplexField, Conjugate};
use num_traits::float::Float;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use faer::MatMut;

// Abandon hope, all ye who enter here

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub trait TensorFloat: Float + Clone + Copy + Conjugate + ComplexField + Display + 'static {
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self;
    //fn one() -> Self;
}

impl TensorFloat for f32 {
    #[inline(always)]
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.random::<f32>()
    }
}

impl TensorFloat for f64 {
    #[inline(always)]
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.random::<f64>()
    }
}

/// Tensor in column-major order.
/// Owns data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: TensorFloat, const D: usize> {
    id: usize,
    #[serde(with = "arc_vec")]
    data: Arc<Vec<T>>,
    #[serde(with = "serde_arrays")]
    shape: [usize; D],
    #[serde(with = "serde_arrays")]
    strides: [isize; D],

    pub grad: Option<Box<Tensor<T, D>>>,
}

mod arc_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub fn serialize<T: Serialize, S: Serializer>(
        arc: &Arc<Vec<T>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Vec::serialize(arc, serializer)
    }

    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<Vec<T>>, D::Error> {
        let vec = Vec::<T>::deserialize(deserializer)?;
        Ok(Arc::new(vec))
    }
}

impl<T: TensorFloat, const D: usize> Tensor<T, D> {
    /// Main constructor
    #[inline(always)]
    // WARN: REFACTOR: ADDED RETURN WITH NEW ID
    pub fn new(shape: [usize; D], data: Vec<T>) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        if expected_len != data.len() {
            return Err(LibError::InvalidShapeForData {
                shape: shape.to_vec(),
                expected_len,
                actual_len: data.len(),
            });
        }
        let strides = Self::compute_strides(&shape);
        let data = Arc::new(data);
        let grad = None;
        Ok(unsafe { Tensor::new_with_strides(shape, strides, data, grad) })
    }

    // WARN: REFACTOR: ADDED RETURN WITH NEW ID
    pub fn new_row(shape: [usize; D], data: Vec<T>) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        if expected_len != data.len() {
            return Err(LibError::InvalidShapeForData {
                shape: shape.to_vec(),
                expected_len,
                actual_len: data.len(),
            });
        }
        let strides = Self::compute_strides_row(&shape);
        let data = Arc::new(data);
        let grad = None;
        Ok(unsafe { Tensor::new_with_strides(shape, strides, data, grad) })
    }

    #[deprecated(note = "use `from_raw_parts`")]
    #[inline(always)]
    pub unsafe fn new_raw(
        shape: [usize; D],
        data: Vec<T>,
        strides: [isize; D],
        grad: Option<Box<Tensor<T, D>>>,
    ) -> Result<Self, LibError> {
        panic!(
            "`new_raw` is hard deprecated. Use `from_raw_parts` instead, which does not perform any checks and is more efficient."
        )
    }

    /// Reconstructs a tensor from its raw components while preserving its original unique ID.
    ///
    /// This function is primarily used internally to create **zero-copy views** of an existing
    /// tensor (e.g., during transposition, slicing, or reshaping). It keeps `id`, allowing
    /// creation of new tensor in place of old one, without changing the identity of the tensor in
    /// the computation graph.
    ///
    /// ### Memory Layout & Flat Index
    ///
    /// The mapping from an N-dimensional coordinate $(i_0, i_1, \dots, i_{D-1})$ to a 1D position
    /// in the flat `data` buffer is defined by the flat index $I$. Given the dimensions' strides
    /// $(s_0, s_1, \dots, s_{D-1})$, the index is calculated as the dot product:
    ///
    /// $$I = \sum_{k=0}^{D-1} i_k \cdot s_k$$
    ///
    /// For example, in a 4-dimensional tensor with coordinates $(c, w, h, n)$ and
    /// strides $(s_c, s_w, s_h, s_n)$, the flat memory offset is computed as:
    ///
    /// $$I = c \cdot s_c + w \cdot s_w + h \cdot s_h + n \cdot s_n$$
    ///
    /// # Safety
    ///
    /// This function is highly `unsafe` because it does not validate memory bounds.
    /// The caller must guarantee that:
    /// - For every valid coordinate where $0 \le i_k < \text{shape}[k]$, the computed
    ///   flat index $I$ will **never** exceed the allocated bounds of the `data` buffer.
    /// - The `data` buffer is fully initialized.
    #[inline(always)]
    pub unsafe fn from_raw_parts(
        id: usize,
        shape: [usize; D],
        data: Arc<Vec<T>>,
        strides: [isize; D],
        grad: Option<Box<Tensor<T, D>>>,
    ) -> Self {
        Self {
            id,
            data,
            shape,
            strides,
            grad,
        }
    }

    /// Creates a completely new tensor with a custom memory layout and generates a new unique ID.
    ///
    /// Unlike [`from_raw_parts`], this function assumes the tensor is a logically new entity
    /// within the computation graph. It is typically used when a backend operation produces
    /// a new data buffer that requires specific, non-contiguous strides to be read correctly.
    ///
    /// # Safety
    ///
    /// This function bypasses standard shape and memory validation.
    /// The caller is fully responsible for ensuring that:
    /// - Any index computed using the provided `shape` and `strides` stays strictly within
    ///   the bounds of the `data` buffer to prevent undefined behavior (UB) and segmentation faults.
    /// - *Note:* The product of the `shape` dimensions does **not** strictly need to match
    ///   the length of `data` (e.g., when the tensor is mapped to a sub-region of a larger buffer),
    ///   but out-of-bounds access must be mathematically impossible.
    #[inline(always)]
    pub unsafe fn new_with_strides(
        shape: [usize; D],
        strides: [isize; D],
        data: Arc<Vec<T>>,
        grad: Option<Box<Tensor<T, D>>>,
    ) -> Self {
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            data,
            shape,
            strides,
            grad,
        }
    }

    // TODO: CHANGE
    pub fn new_with_init<I: Initialization<T>>(
        shape: [usize; D],
        (fan_in, fan_out): (usize, usize),
        init: &I,
    ) -> Result<Self, LibError> {
        let data = init
            .initialize_tensor(fan_in, fan_out, shape)?
            .data
            .as_slice()
            .to_vec();
        Self::new(shape, data)
    }

    pub fn new_from_val(shape: [usize; D], val: T) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        let data = vec![val; expected_len];
        Self::new(shape, data)
    }

    const fn compute_strides<const N: usize>(shape: &[usize; N]) -> [isize; N] {
        let mut strides = [0isize; N];
        let mut prod = 1;
        let mut i = 0;

        while i < N {
            strides[i] = prod;
            prod *= shape[i] as isize;
            i += 1;
        }

        strides
    }

    const fn compute_strides_row<const N: usize>(shape: &[usize; N]) -> [isize; N] {
        let mut strides = [0isize; N];
        let mut prod = 1;
        let mut i = 0;
        strides.swap(0, 1);

        while i < D {
            strides[i] = prod;
            prod *= shape[i] as isize;
            i += 1;
        }

        strides.swap(0, 1);
        strides
    }

    pub fn ones(shape: [usize; D]) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        let data = vec![T::one(); expected_len];
        Self::new(shape, data)
    }

    pub fn zeros(shape: [usize; D]) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        let data = vec![T::zero(); expected_len];
        Self::new(shape, data)
    }

    pub unsafe fn uninitialized(shape: [usize; D]) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        let mut data: Vec<T> = Vec::with_capacity(expected_len);
        unsafe {
            data.set_len(expected_len);
        }
        Self::new(shape, data)
    }

    /// TODO: REFACTOR FOR PERFORMANCE
    pub fn from_fn(shape: [usize; D], mut f: impl FnMut(&[usize]) -> T) -> Result<Self, LibError> {
        let expected_len: usize = shape.iter().product();
        let data: Vec<T> = (0..expected_len).map(|_| f(&shape)).collect();
        Self::new(shape, data)
    }

    // WARN: REFACTOR: ADDED RETURN WITH NEW ID
    pub fn random(shape: [usize; D]) -> Self {
        let len: usize = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<T> = (0..len).map(|_| T::random(&mut rng)).collect();
        let data = Arc::new(data);
        let strides = Self::compute_strides(&shape);
        let grad = None;

        unsafe { Tensor::new_with_strides(shape, strides, data, grad) }
    }

    // WARN: REFACTOR: ADDED RETURN WITH NEW ID
    /// Constructor using slices
    pub fn from_slice(shape: [usize; D], data_slice: impl AsRef<[T]>) -> Result<Self, LibError> {
        let data = data_slice.as_ref();

        let expected_len: usize = shape.iter().product();
        if expected_len != data.len() {
            return Err(LibError::InvalidShapeForData {
                shape: shape.to_vec(),
                expected_len,
                actual_len: data.len(),
            });
        }

        let strides = Self::compute_strides(&shape);
        let grad = None;

        let data = Arc::new(data.to_vec());

        Ok(unsafe { Tensor::new_with_strides(shape, strides, data, grad) })
    }

    /// Returns the shape of the tensor
    #[inline(always)]
    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }

    #[inline(always)]
    pub fn strides(&self) -> &[isize; D] {
        &self.strides
    }

    #[inline(always)]
    pub fn data_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn change_tensor(&mut self, other: Tensor<T, D>) {
        self.data = other.data;
        self.shape = other.shape;
        self.strides = other.strides;
        self.grad = other.grad;
    }

    /*
    #[allow(dead_code)]
    fn make_mut(mut self) -> Tensor<T, D> {
        if let Some(data) = Arc::get_mut(&mut self.data) {
            let grad = self.grad.take();
            unsafe { Tensor::from_raw_parts(self.id, self.shape, data, self.strides, grad) }
            return Tensor {
                data: Arc::new(data.clone()),
                shape: self.shape,
                strides: self.strides,
                grad,
            };
        }
        self
    }
    */

    /// Gets an element by multi-dimensional index
    /// This operation is very expensive, as it requires calculating the flat index,
    /// and it involves bounds checking. It also requires locking the data for reading,
    /// so it is not recommended for performance-critical code.
    #[inline(always)]
    pub fn get(&self, idx: impl AsRef<[usize]>) -> Option<T> {
        let idx = idx.as_ref();
        if idx.len() != self.shape.len() {
            return None;
        }
        idx.iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
            .try_fold(0isize, |acc, (&ind, (&dim, &stride))| {
                if ind >= dim {
                    None
                } else {
                    Some(acc + ind as isize * stride)
                }
            })
            .and_then(|flat_idx| self.data.get(flat_idx as usize).copied())
    }
    #[inline(always)]
    pub fn set(&mut self, idx: impl AsRef<[usize]>, value: T) -> Result<(), String> {
        let idx = idx.as_ref();
        if idx.len() != self.shape.len() {
            return Err("Index dimensionality mismatch".to_string());
        }
        let flat_idx = idx
            .iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
            .try_fold(0isize, |acc, (&ind, (&dim, &stride))| {
                if ind >= dim {
                    None
                } else {
                    Some(acc + ind as isize * stride)
                }
            })
            .ok_or_else(|| "Index out of bounds".to_string())?;
        // Lock the data for writing TODO: IN FUTURE: handle lock errors
        match Arc::get_mut(&mut self.data) {
            Some(data) => {
                if let Some(elem) = data.get_mut(flat_idx as usize) {
                    *elem = value;
                    return Ok(());
                } else {
                    return Err("Index out of bounds".to_string());
                };
            }
            None => return Err("Cannot get mutable reference to data, it is shared".to_string()),
        };
    }

    /// Reshapes the tensor if compatible. This is a zero-copy operation.
    /// The tensor must be contiguous for this operation to be valid.
    pub fn reshape_to_tensorref<'a>(
        &'a self,
        shape: [usize; D],
    ) -> Result<TensorRef<'a, T>, LibError> {
        let new_len: usize = shape.iter().product();
        let old_len: usize = self.shape.iter().product();

        if new_len != old_len {
            return Err(LibError::ShapeMismatch {
                operation: "reshape".to_string(),
                expected: vec![new_len],
                actual: vec![old_len],
            });
        }

        let expected_strides = Self::compute_strides(&self.shape);
        if self.strides != expected_strides {
            // For now, we require the tensor to be contiguous.
            return Err(LibError::Unimplemented(
                "Reshape is only supported for contiguous tensors.".to_string(),
            ));
        }

        let strides = Self::compute_strides(&shape);

        Ok(TensorRef {
            data: self.data.as_ptr(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            _marker: PhantomData,
        })
    }

    /// Reshapes the tensor if compatible. This is a zero-copy operation.
    /// The tensor must be contiguous.
    // WARN: REFACTOR: ADDED RETURN WITH EXISTING ID
    #[inline(always)]
    pub fn reshape<const N: usize>(&self, shape: [usize; N]) -> Result<Tensor<T, N>, LibError> {
        let new_len: usize = shape.iter().product();
        let old_len: usize = self.shape.iter().product();

        if new_len != old_len {
            return Err(LibError::ShapeMismatch {
                operation: "reshape".to_string(),
                expected: vec![new_len],
                actual: vec![old_len],
            });
        }

        let expected_strides = Self::compute_strides(&self.shape);
        if self.strides != expected_strides {
            // For now, we require the tensor to be contiguous.
            return Err(LibError::Unimplemented(
                "Reshape is only supported for contiguous tensors.".to_string(),
            ));
        }

        let data = self.data.clone();
        let strides = Self::compute_strides(&shape);
        match self.grad {
            Some(ref g) => {
                let grad = g.reshape(shape)?;
                Ok(unsafe {
                    Tensor::from_raw_parts(self.id, shape, data, strides, Some(Box::new(grad)))
                })
            }
            None => Ok(unsafe { Tensor::from_raw_parts(self.id, shape, data, strides, None) }),
        }
    }

    /// Flattens the tensor to a 1D `TensorRef`.
    ///
    /// This operation is a zero-copy view, but it requires the tensor to be contiguous.
    /// It will panic if the tensor's memory layout is not contiguous.
    #[inline(always)]
    pub fn flatten_to_tensorref<'a>(&self) -> TensorRef<'a, T> {
        let expected_strides = Self::compute_strides(&self.shape);
        if self.strides != expected_strides {
            // TODO: Add a more robust `is_contiguous` check
            panic!("`flatten_to_tensorref` requires the tensor to be contiguous.");
        }
        let len = self.shape.iter().product();
        TensorRef {
            data: self.data.as_ptr(),
            shape: vec![len],
            strides: vec![1],
            _marker: PhantomData,
        }
    }

    /// Flattens the tensor into a new, contiguous 1D `Tensor`.
    ///
    /// This method provides a fast path for contiguous tensors (zero-copy) and
    /// handles non-contiguous tensors by iterating and collecting elements into a new buffer.
    // WARN: REFACTOR: ADDED RETURN WITH EXISTING ID
    #[inline(always)]
    pub fn flatten(&self) -> Tensor<T, 1> {
        let expected_strides = Self::compute_strides(&self.shape);
        let grad = match self.grad {
            Some(ref g) => {
                let g = g.flatten();
                Some(Box::new(g))
            }
            None => None,
        };
        if self.strides == expected_strides {
            // Fast-path for contiguous tensors (column-major).
            unsafe {
                Tensor::from_raw_parts(
                    self.id,
                    [self.shape.iter().product(); 1],
                    self.data.clone(),
                    [1],
                    grad,
                )
            }
        } else {
            // Slow-path for non-contiguous tensors.
            let new_data: Vec<T> = self.iter().collect();
            // This unwrap is safe because the length of new_data always matches the product of the new_shape.
            let mut out = Tensor::new([new_data.len(); 1], new_data).unwrap();
            out.grad = grad;
            out
        }
    }

    /*
    pub fn as_faer_mut(&mut self) -> Result<MatMut<'_, T>, String> {
        if self.shape.len() != 2 {
            return Err("Tensors have to be 2-dimensional to change it to faer ref".to_string());
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];

        let mut self_data = self.data.write().unwrap();

        Ok(MatMut::from_column_major_slice_mut(
            &mut self_data,
            nrows,
            ncols,
        ))
    }
    */
    #[inline(always)]
    pub fn as_faer_mut_unsafe(&mut self) -> Result<MatMut<'_, T>, LibError> {
        if self.shape.len() != 2 {
            return Err(LibError::InvalidDimensionality {
                operation: "as_faer_mut_unsafe".to_string(),
                expected: 2,
                actual: self.shape.len(),
            });
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];

        let data = match Arc::get_mut(&mut self.data) {
            Some(data) => data.as_mut_ptr(),
            None => return Err(LibError::SharedTensorMutability),
        };

        unsafe {
            Ok(MatMut::from_raw_parts_mut(
                data,
                nrows,
                ncols,
                self.strides[0],
                self.strides[1],
            ))
        }
    }

    #[inline(always)]
    pub fn as_faer_ref(&self) -> Result<faer::MatRef<'_, T>, LibError> {
        if self.shape.len() != 2 {
            return Err(LibError::InvalidDimensionality {
                operation: "as_faer_ref".to_string(),
                expected: 2,
                actual: self.shape.len(),
            });
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];

        Ok(faer::MatRef::from_column_major_slice(
            &self.data, nrows, ncols,
        ))
    }

    #[inline(always)]
    pub fn as_faer_ref_unsafe(&self) -> Result<faer::MatRef<'_, T>, LibError> {
        if self.shape.len() != 2 {
            return Err(LibError::InvalidDimensionality {
                operation: "as_faer_ref_unsafe".to_string(),
                expected: 2,
                actual: self.shape.len(),
            });
        }
        let nrows = self.shape[0];
        let ncols = self.shape[1];

        unsafe {
            Ok(faer::MatRef::from_raw_parts(
                self.data.as_ptr(),
                nrows,
                ncols,
                self.strides[0],
                self.strides[1],
            ))
        }
    }

    #[inline(always)]
    fn deep_copy(&self) -> Self {
        let data = Arc::new(self.data.as_slice().to_vec());
        let grad = self.grad.as_ref().map(|g| Box::new(g.as_ref().deep_copy()));
        unsafe { Tensor::new_with_strides(self.shape, self.strides, data, grad) }
    }

    // TODO: FIX IT
    #[inline(always)]
    pub fn make_unique(self) -> Self {
        if Arc::strong_count(&self.data) > 1 {
            self.deep_copy()
        } else {
            self
        }
    }

    #[inline(always)]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    #[inline(always)]
    pub fn physical_offset(&self, logical_idx: impl AsRef<[usize]>) -> isize {
        let logical_idx = logical_idx.as_ref();
        logical_idx
            .iter()
            .zip(self.strides.iter())
            .fold(0isize, |acc, (&ind, &stride)| acc + ind as isize * stride)
    }

    #[inline(always)]
    pub fn logical_index(&self, physical_idx: isize) -> Option<Vec<usize>> {
        let mut remaining = physical_idx;
        let mut logical_idx = vec![0; self.shape.len()];

        for (i, &stride) in self.strides.iter().enumerate().rev() {
            if stride == 0 {
                return None; // Avoid division by zero
            }
            logical_idx[i] = (remaining / stride) as usize;
            if logical_idx[i] >= self.shape[i] {
                return None; // Out of bounds
            }
            remaining -= logical_idx[i] as isize * stride;
        }

        Some(logical_idx)
    }

    #[inline(always)]
    pub fn sum_keepdim(&self, dim: usize) -> Tensor<T, D> {
        if dim >= self.shape.len() {
            panic!("Dimension {} out of bounds for shape {:?}", dim, self.shape);
        }
        let mut result_shape = self.shape.clone();
        result_shape[dim] = 1;

        let mut result = Tensor::zeros(result_shape).unwrap();

        if self.shape.iter().product::<usize>() == 0 {
            return result;
        }

        let result_strides = result.strides.clone();
        let result_data = Arc::get_mut(&mut result.data).unwrap();

        for (logical_idx, value) in self.iter_with_index() {
            let mut result_idx = logical_idx;
            result_idx[dim] = 0;

            let physical_idx = result_idx
                .iter()
                .zip(result_strides.iter())
                .fold(0, |acc, (&i, &s)| acc + i as isize * s);
            result_data[physical_idx as usize] = result_data[physical_idx as usize] + value;
        }

        result
    }

    /// Squeezes the specific dimension after summing over it. The programmer
    /// has to provide new size (`const generics` are unstable, so we won't use them here)
    // WARN: REFACTOR: ADDED RETURN WITH NEW ID
    #[inline(always)]
    pub fn sum_squeeze<const D_OUT: usize>(&self, dim: usize) -> Tensor<T, D_OUT> {
        assert_eq!(
            D,
            D_OUT + 1,
            "tensor::tensor::sum_squeeze: D_OUT has to be of size D - 1, where D is input size!"
        );
        if dim >= D {
            panic!("Dimension {} goes beyond Tensor shape!", dim);
        }

        // let's compute the sum with keeping the dim first
        let kept_dim_tensor = self.sum_keepdim(dim);

        // and let's declare our new shape and stride, making them smaller
        let mut new_shape = [0usize; D_OUT];
        let mut new_strides = [0isize; D_OUT];
        let mut idx = 0;

        for i in 0..D {
            if i != dim {
                new_shape[idx] = kept_dim_tensor.shape[i];
                new_strides[idx] = kept_dim_tensor.strides[i];
                idx += 1;
            }
        }

        unsafe {
            Tensor::new_with_strides(new_shape, new_strides, kept_dim_tensor.data.clone(), None)
        }
    }

    #[inline(always)]
    pub fn sum_all(&self) -> T {
        //self.iter().fold(T::zero(), |acc, x| acc + x)
        self.data
            .as_slice()
            .iter()
            .fold(T::zero(), |acc, x| acc + *x)
    }

    #[inline(always)]
    pub fn map<F: Fn(&T) -> T + Sync + Send>(&self, f: F) -> Tensor<T, D> {
        //let new_data: Vec<T> = self.iter().map(f).collect();
        let new_data = self.data.par_iter().map(f).collect();
        Tensor::new(self.shape.clone(), new_data).unwrap()
    }

    #[inline(always)]
    pub fn mul_elem(&self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        assert_eq!(self.shape, rhs.shape);
        if self.strides == rhs.strides {
            // fast path
            // we have the same memory layout for two Tensors, we can operate
            // on raw vector data
            let new_data: Vec<T> = self
                .data
                .par_iter()
                .zip(rhs.data.par_iter())
                .map(|(a, b)| *a * *b)
                .collect();
            return Tensor::new(*self.shape(), new_data).unwrap();
        }
        let new_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect();
        Tensor::new(self.shape.clone(), new_data).unwrap()
    }

    #[inline(always)]
    pub fn map_inplace<F: Fn(T) -> T + Send + Sync>(&mut self, f: F) {
        let data = Arc::get_mut(&mut self.data)
            .expect("Cannot map in-place on a shared tensor. Use `map()` to create a new tensor.");

        data.par_iter_mut().for_each(|x| *x = f(*x));
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn get_data(&self) -> &[T] {
        self.data.as_slice()
    }

    //pub fn iter(&self) -> TensorIter<'_, T> {
    //    TensorIter {
    //        tensor: self,
    //        logical_pos: vec![0; self.shape.len()],
    //        finished: self.shape.iter().any(|&d| d == 0),
    //    }
    //}

    pub fn iter(&self) -> TensorIter<'_, Self> {
        iter(self)
    }

    pub fn iter_with_index(&self) -> TensorIterWithIndex<'_, T, D> {
        TensorIterWithIndex {
            tensor: self,
            logical_pos: vec![0; self.shape.len()],
            finished: self.shape.iter().any(|&d| d == 0),
        }
    }

    /// Creates a zero-copy view (TensorRef) of a sub-tensor by selecting an index along a specified axis.
    ///
    /// This effectively reduces the dimensionality of the tensor by one. For example, selecting
    /// index `i` along axis `3` of a 4D tensor `[C, W, H, N]` will produce a 3D view
    /// representing the `i`-th sample, with shape `[C, W, H]`.
    /// You have to solely own data in Arc's Tensor (use make_unique if needed)
    ///
    /// # Arguments
    /// * `axis` - The dimension along which to slice.
    /// * `index` - The index to select on that axis.
    ///
    /// # Returns
    /// A `Result` containing a `TensorRef` on success, or a `LibError` if the
    /// axis or index are out of bounds.
    #[inline(always)]
    pub fn select<'a>(&'a self, axis: usize, index: usize) -> Result<TensorRef<'a, T>, LibError> {
        // Validation
        if axis >= D {
            return Err(LibError::InvalidDimensionality {
                operation: "select".to_string(),
                expected: axis + 1,
                actual: D,
            });
        }
        if index >= self.shape[axis] {
            return Err(LibError::OutOfBounds {
                shape: self.shape.to_vec(),
                index: {
                    let mut idx = vec![0; D];
                    idx[axis] = index;
                    idx
                },
            });
        }

        // New data pointer
        // Let's move pointer by index steps (times stride element by selected exis)
        let offset = self.strides[axis] * index as isize;
        let new_ptr = unsafe { self.data.as_ptr().offset(offset) };

        // Let's calculate new shape and strides
        // We can just remove selected positions from shape and strides vecs.
        let mut new_shape = self.shape.to_vec();
        new_shape.remove(axis);

        let mut new_strides = self.strides.to_vec();
        new_strides.remove(axis);

        // Edge case: if we select 1-dim Tensor, we must gracefully handle it
        // The output will be 1-dim Tensor
        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }

        Ok(unsafe { TensorRef::new(new_ptr, new_shape, new_strides) })
    }

    /// Creates a zero-copy mutable view (TensorRefMut) of a sub-tensor by selecting an index along a specified axis.
    ///
    /// This effectively reduces the dimensionality of the tensor by one. For example, selecting
    /// index `i` along axis `3` of a 4D tensor `[C, H, W, N]` will produce a 3D view
    /// representing the `i`-th sample, with shape `[C, H, W]`.
    /// You have to solely own data in Arc's Tensor (use make_unique if needed)
    ///
    /// # Arguments
    /// * `axis` - The dimension along which to slice.
    /// * `index` - The index to select on that axis.
    ///
    /// # Returns
    /// A `Result` containing a `TensorRefMut` on success, or a `LibError` if the
    /// axis or index are out of bounds.
    #[inline(always)]
    pub fn select_mut<'a>(
        &'a mut self,
        axis: usize,
        index: usize,
    ) -> Result<TensorRefMut<'a, T>, LibError> {
        // Safety check - we habe to be sure we are solely Tensor owners
        if Arc::strong_count(&self.data) > 1 {
            return Err(LibError::SharedTensorMutability);
        }

        // Validation
        if axis >= self.shape.len() {
            return Err(LibError::InvalidDimensionality {
                operation: "select_mut".to_string(),
                expected: axis + 1,
                actual: self.shape.len(),
            });
        }
        if index >= self.shape[axis] {
            return Err(LibError::OutOfBounds {
                shape: self.shape.to_vec(),
                index: {
                    let mut idx = vec![0; self.shape.len()];
                    idx[axis] = index;
                    idx
                },
            });
        }

        // Lets calculate new mut data pointer
        // It's safe to get mut pointer, as we are solely owner of Arc data
        let base_ptr = Arc::get_mut(&mut self.data).unwrap().as_mut_ptr();
        let offset = self.strides[axis] * index as isize;
        let new_ptr = unsafe { base_ptr.offset(offset) };

        let mut new_shape = self.shape.to_vec();
        new_shape.remove(axis);

        let mut new_strides = self.strides.to_vec();
        new_strides.remove(axis);

        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }

        Ok(unsafe { TensorRefMut::new(new_ptr, new_shape, new_strides) })
    }

    /// Permutes the dimensions of the tensor. This is a zero-copy operation.
    ///
    /// The `axes` slice specifies the new order of dimensions. For example, for a 3D tensor,
    /// `permute(&[2, 0, 1])` would move the original 2nd dimension to the 0th, the 0th to the 1st,
    /// and the 1st to the 2nd.
    ///
    /// # Panics
    /// Panics if the length of `axes` is not equal to the number of dimensions of the tensor,
    /// or if `axes` contains duplicate dimension indices.
    #[inline(always)]
    pub fn permute(&self, axes: [usize; D]) -> Self {
        let mut new_shape = [0usize; D];
        let mut new_strides = [0isize; D];
        let mut seen = [false; D];

        for (i, &axis) in axes.iter().enumerate() {
            if axis >= D {
                panic!(
                    "Permutation error: axis {} is out of bounds for tensor with {} dimensions",
                    axis, D
                );
            }
            if seen[axis] {
                panic!("Permutation error: axis {} is duplicated", axis);
            }
            seen[axis] = true;
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }

        unsafe {
            Tensor::from_raw_parts(
                self.id,
                new_shape,
                self.data.clone(),
                new_strides,
                self.grad.clone(),
            )
        }
    }

    /// Flips the tensor along the specified axes. This is a zero-copy operation.
    ///
    /// This method creates a new view of the tensor with the order of elements
    /// along the given axes reversed. It does so by adjusting the strides and
    /// the internal data pointer, without copying any data.
    ///
    /// # Panics
    /// Panics if any axis in `axes` is out of bounds.
    #[inline(always)]
    pub fn flip(&self, axes: &[usize]) -> Self {
        let mut new_strides = self.strides.clone();
        let mut new_ptr = self.data.as_ptr();

        for &axis in axes {
            if axis >= self.shape.len() {
                panic!(
                    "Flip error: axis {} is out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape.len()
                );
            }

            new_strides[axis] = -new_strides[axis];

            let offset = (self.shape[axis] - 1) as isize * self.strides[axis];
            new_ptr = unsafe { new_ptr.offset(offset) };
        }

        let reversed_view =
            unsafe { TensorRef::new(new_ptr, self.shape.to_vec(), new_strides.to_vec()) };
        reversed_view.to_owned().unwrap()
    }

    #[inline(always)]
    pub fn contiguous(&self) -> bool {
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    #[inline(always)]
    pub fn make_contiguous(&self) -> Result<Self, LibError> {
        if self.contiguous() {
            Ok(self.clone())
        } else {
            let new_data: Vec<T> = self.iter().collect();
            Tensor::new(self.shape.clone(), new_data)
        }
    }

    pub fn iter_indices(&self) -> impl Iterator<Item = Vec<usize>> {
        let shape = self.shape.clone();
        let len: usize = shape.iter().product();
        (0..len).map(move |n| {
            let mut indices = vec![0; shape.len()];
            let mut temp_n = n;
            for i in 0..shape.len() {
                indices[i] = temp_n % shape[i];
                temp_n /= shape[i];
            }
            indices
        })
    }

    /// Returns a new view of the tensor with singleton dimensions expanded
    /// to a larger size.
    ///
    /// This is a zero-copy operation. It uses `stride=0` to repeat values
    /// along a dimension without allocating new memory.
    ///
    /// # Arguments
    /// * `shape` - The desired output shape.
    ///
    /// # Errors
    /// Returns `LibError::ShapeMismatch` if the tensor cannot be broadcast
    /// to the target shape.
    ///
    // WARN: REFACTOR: ADDED RETURN WITH EXISTING ID
    #[inline(always)]
    pub fn expand<const N: usize>(&self, shape: [usize; N]) -> Result<Tensor<T, N>, LibError> {
        if shape.len() < self.shape.len() {
            return Err(LibError::ShapeMismatch {
                operation: "expand (target dims < source dims)".to_string(),
                expected: shape.to_vec(),
                actual: self.shape.to_vec(),
            });
        }

        let mut new_strides = [0isize; N];
        let offset = shape.len() - self.shape.len();

        for i in 0..self.shape.len() {
            let source_dim = self.shape[i];
            let source_stride = self.strides[i];

            let target_dim_idx = i + offset;
            let target_dim = shape[target_dim_idx];

            if source_dim == target_dim {
                new_strides[target_dim_idx] = source_stride;
            } else if source_dim == 1 {
                new_strides[target_dim_idx] = 0;
            } else {
                return Err(LibError::ShapeMismatch {
                    operation: "expand (incompatible dimension)".to_string(),
                    expected: shape.to_vec(),
                    actual: self.shape.to_vec(),
                });
            }
        }

        match self.grad {
            Some(ref g) => {
                let grad = g.expand(shape)?;
                unsafe {
                    Ok(Tensor::from_raw_parts(
                        self.id,
                        shape,
                        self.data.clone(),
                        new_strides,
                        Some(Box::new(grad)),
                    ))
                }
            }
            None => unsafe {
                Ok(Tensor::from_raw_parts(
                    self.id,
                    shape,
                    self.data.clone(),
                    new_strides,
                    None,
                ))
            },
        }
    }
}

// WARN: REFACTOR: ADDED RETURN WITH EXISTING ID
impl<T: TensorFloat> Tensor<T, 2> {
    #[inline(always)]
    /// Returns a new tensor that is the transpose of the original tensor. This is a zero-copy operation.
    pub fn t(&self) -> Tensor<T, 2> {
        let grad = self.grad.clone();
        unsafe {
            Tensor::from_raw_parts(
                self.id,
                [self.shape[1], self.shape[0]],
                self.data.clone(),
                [self.strides[1], self.strides[0]],
                grad,
            )
        }
    }
}

impl<T: TensorFloat, const D: usize> Display for Tensor<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shape.is_empty() {
            return write!(f, "{}", self.data[0]);
        }

        writeln!(
            f,
            "Tensor (shape: {:?}, strides: {:?})",
            self.shape, self.strides
        )?;

        let mut indices = vec![0; self.shape.len()];
        fmt_recursive(self, f, self.shape.len() - 1, &mut indices)
    }
}

fn fmt_recursive<T: TensorFloat, const D: usize>(
    tensor: &Tensor<T, D>,
    f: &mut std::fmt::Formatter<'_>,
    dim: usize,
    indices: &mut [usize],
) -> std::fmt::Result {
    if dim == 1 {
        // We are at the level of a 2D matrix, so we print it.
        if tensor.shape.len() > 2 {
            let mut s = String::from("[:, :, ");
            for i in 2..tensor.shape.len() {
                if i > 2 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}", indices[i]));
            }
            s.push_str("] =");
            writeln!(f, "{}", s)?;
        }

        for i in 0..tensor.shape[0] {
            indices[0] = i;
            for j in 0..tensor.shape[1] {
                indices[1] = j;
                write!(f, "{:8.4}", tensor[indices])?;
            }
            writeln!(f)?;
        }
        if tensor.shape.len() > 2 {
            writeln!(f)?;
        }
    } else {
        // We are at a higher dimension, so we recursively call the function.
        for i in 0..tensor.shape[dim] {
            indices[dim] = i;
            fmt_recursive(tensor, f, dim - 1, indices)?;
        }
    }

    Ok(())
}

pub struct TensorIterWithIndex<'a, T: TensorFloat, const D: usize> {
    tensor: &'a Tensor<T, D>,
    logical_pos: Vec<usize>,
    finished: bool,
}

impl<'a, T: TensorFloat, const D: usize> Iterator for TensorIterWithIndex<'a, T, D> {
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let logical_pos = self.logical_pos.clone();
        let item = self.tensor[logical_pos.as_slice()];

        // Advance logical_pos (column-major)
        let mut i = 0;
        loop {
            self.logical_pos[i] += 1;
            if self.logical_pos[i] < self.tensor.shape[i] {
                break;
            }
            self.logical_pos[i] = 0;
            if i == self.tensor.shape.len() - 1 {
                self.finished = true;
                break;
            }
            i += 1;
        }

        Some((logical_pos, item))
    }
}

impl<T: TensorFloat, const D: usize> Index<&[usize]> for Tensor<T, D> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &T {
        let index = self.physical_offset(&index) as usize;
        &self.data[index]
    }
}

/*
impl<T: TensorFloat> IndexMut<Vec<usize>> for Tensor<T> {
    fn index_mut(&mut self, index: Vec<usize>) -> &mut T {
        let index = self.physical_offset(&index) as usize;
        match Arc::get_mut(&mut self.data) {
            Some(data) => &mut data[index],
            None => panic!("Cannot get mutable reference to data, it is shared"),
        }
    }
}
*/

impl<T: TensorFloat, const D: usize> IndexMut<&[usize]> for Tensor<T, D> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        let index = self.physical_offset(index) as usize;
        match Arc::get_mut(&mut self.data) {
            Some(data) => &mut data[index],
            None => panic!("Cannot get mutable reference to data, it is shared"),
        }
    }
}

impl<'a, T: TensorFloat, const D: usize> Tensor<T, D> {
    pub fn as_ref(&'a self) -> TensorRef<'a, T> {
        let data = self.data.as_ptr();
        TensorRef {
            data,
            shape: self.shape.to_vec(),
            strides: self.strides.to_vec(),
            _marker: PhantomData,
        }
    }
}

impl<T, const D: usize> MatMul<Tensor<T, D>> for Tensor<T, 2>
where
    T: TensorFloat,
{
    type Output = Result<Tensor<T, 2>, LibError>;

    fn matmul(&self, rhs: &Tensor<T, D>) -> Self::Output {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(LibError::InvalidDimensionalityTwoTensors {
                operation: "matmul".to_string(),
                expected_a: 2,
                actual_a: self.shape.len(),
                expected_b: 2,
                actual_b: rhs.shape.len(),
            });
        }
        if self.shape[1] != rhs.shape[0] {
            return Err(LibError::ShapeMismatch {
                operation: "matmul".to_string(),
                expected: vec![self.shape[1]],
                actual: vec![rhs.shape[0]],
            });
        }

        let nrows = self.shape[0];
        let ncols = rhs.shape[1];

        let mut result = unsafe { Tensor::uninitialized([nrows, ncols])? };

        let a_faer = self.as_faer_ref_unsafe()?;
        let b_faer = rhs.as_faer_ref_unsafe()?;
        let mut c_faer = result.as_faer_mut_unsafe()?;

        faer::linalg::matmul::matmul(
            &mut c_faer,
            faer::Accum::Replace,
            a_faer,
            b_faer,
            T::one(),
            Par::Rayon(
                NonZero::new(num_cpus::get_physical().max(1))
                    .expect("this should never happen - library couldn't detect any CPU cores"),
            ),
        );

        //println!("Result data: {:?}", result.data.as_slice());

        Ok(result)
    }
}

impl<T, const D: usize> TensorAdd<Tensor<T, D>> for Tensor<T, D>
where
    T: TensorFloat,
{
    type Output = Result<Tensor<T, D>, LibError>;

    fn tensoradd(&self, rhs: &Tensor<T, D>) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(LibError::ShapeMismatch {
                operation: "tensoradd".to_string(),
                expected: self.shape.to_vec(),
                actual: rhs.shape.to_vec(),
            });
        }

        // /*
        if self.strides == rhs.strides {
            // fast path
            // we have the same memory layout for two Tensors, we can operate
            // on raw vector data
            let new_data: Vec<T> = self
                .data
                .par_iter()
                .zip(rhs.data.par_iter())
                .map(|(a, b)| *a + *b)
                .collect();
            return Tensor::new(*self.shape(), new_data);
        }
        // */
        // fallback, slow path
        let new_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect();
        Ok(Tensor::new(self.shape.clone(), new_data).unwrap())
    }
}

impl<T: TensorFloat, const D: usize> Tensor<T, D> {
    pub fn sub(&self, rhs: &Tensor<T, D>) -> Result<Tensor<T, D>, LibError> {
        if self.shape != rhs.shape {
            return Err(LibError::ShapeMismatch {
                operation: "sub".to_string(),
                expected: self.shape.to_vec(),
                actual: rhs.shape.to_vec(),
            });
        }
        if self.strides == rhs.strides {
            // fast path
            // we have the same memory layout for two Tensors, we can operate
            // on raw vector data
            let new_data: Vec<T> = self
                .data
                .par_iter()
                .zip(rhs.data.par_iter())
                .map(|(a, b)| *a - *b)
                .collect();
            return Tensor::new(*self.shape(), new_data);
        }
        let new_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect();
        Ok(Tensor::new(self.shape.clone(), new_data).unwrap())
    }
}

impl<'a, T: TensorFloat, const D: usize> TensorView<'a> for Tensor<T, D> {
    type Dtype = T;

    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[isize] {
        &self.strides
    }
    fn data_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<'a, T: TensorFloat, const D: usize> TensorView<'a> for &Tensor<T, D> {
    type Dtype = T;

    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn strides(&self) -> &[isize] {
        &self.strides
    }
    fn data_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

/*
impl<'a, T: TensorFloat> AsRef<TensorRef<'a, T>> for Tensor<T> {
    fn as_ref(&self) -> TensorRef<'a, T> {
        self.as_ref()
    }
}
*/

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn creating_new_tensor() {
        let tensor = Tensor::new(
            [2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        tensor.unwrap();
    }

    #[test]
    fn tensor_to_tensorref() {
        let tensor = Tensor::new(
            [2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let tensor = tensor.unwrap();
        let tensorref = tensor.as_ref();
        assert_eq!(tensorref.shape, &[2, 3, 2]);
    }

    #[test]
    fn tensor_shape_and_flatten() {
        let tensor = Tensor::new(
            [2, 3, 3],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0,
            ],
        )
        .unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 3]);
        let flat = tensor.flatten();
        assert_eq!(flat.shape, [18usize]);
        //let data = tensor.data;
        //let flat_slice = unsafe { std::slice::from_raw_parts(flat.data, flat.shape[0]) };
        //assert_eq!(flat_slice, data.as_slice());
    }

    #[test]
    fn tensor_get_and_set() {
        let mut tensor = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.get(&[0, 1]), Some(3.0));
        tensor.set(&[0, 1], 5.0).unwrap();
        assert_eq!(tensor.get(&[0, 1]), Some(5.0));
        assert!(tensor.set(&[2, 0], 1.0).is_err());
        assert_eq!(tensor.get(&[2, 0]), None);
    }

    #[test]
    fn tensor_from_slice() {
        let tensor = Tensor::from_slice([2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(tensor.is_ok());
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data.as_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn tensor_reshape() {
        let tensor = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensorref = tensor.reshape([4]);
        assert!(tensorref.is_ok());
        assert_eq!(tensorref.unwrap().shape, [4]);
        assert!(tensor.reshape([3]).is_err());
    }

    #[test]
    fn matmul_test() {
        let a = Tensor::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data.as_slice(), vec![76.0, 100.0, 103.0, 136.0]);
    }

    #[test]
    fn matmul_incompatible_shapes() {
        let a = Tensor::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([4, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();
        let c = a.matmul(&b);
        assert!(c.is_err());
    }

    #[test]
    fn matadd_test() {
        let a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.tensoradd(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data.as_slice(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn matadd_incompatible_shapes() {
        let a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new([2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        let c = a.tensoradd(&b);
        assert!(c.is_err());
    }

    #[test]
    fn matadd_different_strides() {
        let a = unsafe {
            Tensor::from_raw_parts(0, [2, 2], Arc::new(vec![1.0, 2.0, 3.0, 4.0]), [1, 2], None)
        };
        let b = unsafe {
            Tensor::from_raw_parts(
                1,
                [2, 2],
                Arc::new(vec![10.0, 100.0, 1000.0, 10000.0]),
                [2, 1],
                None,
            )
        };
        // strides
        let c = a.tensoradd(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data.as_slice(), vec![11.0, 1002.0, 103.0, 10004.0]);
    }

    #[test]
    fn tensor_make_unique() {
        let tensor = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor_clone = tensor.clone();
        assert_eq!(Arc::strong_count(&tensor.data), 2);
        let unique_tensor = tensor_clone.make_unique();
        assert_eq!(Arc::strong_count(&tensor.data), 1);
        assert_eq!(Arc::strong_count(&unique_tensor.data), 1);
        assert_eq!(tensor.data.as_slice(), unique_tensor.data.as_slice());
        assert_eq!(tensor.shape(), unique_tensor.shape());
        assert_eq!(tensor.strides, unique_tensor.strides);
    }

    #[test]
    fn logical_physical_index() {
        let tensor = Tensor::new([3, 4, 2], (0..24).map(|x| x as f64).collect()).unwrap();
        for i in 0..24 {
            let logical_idx = tensor.logical_index(i).unwrap();
            let physical_idx = tensor.physical_offset(&logical_idx);
            assert_eq!(physical_idx, i as isize);
        }
    }

    #[test]
    fn physical_index_out_of_bounds() {
        let tensor = Tensor::new([3, 4, 2], (0..24).map(|x| x as f64).collect()).unwrap();
        assert!(tensor.logical_index(24).is_none());
        assert!(tensor.logical_index(100).is_none());
    }

    #[test]
    fn sum_test() {
        let tensor = Tensor::new([2, 3, 4], (1..=24).map(|x| x as f64).collect()).unwrap();
        let summed_dim0 = tensor.sum_keepdim(0);
        assert_eq!(summed_dim0.shape(), &[1, 3, 4]);
        assert_eq!(
            summed_dim0.data.as_slice(),
            vec![
                3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0, 35.0, 39.0, 43.0, 47.0
            ]
        );

        let summed_dim1 = tensor.sum_keepdim(1);
        assert_eq!(summed_dim1.shape(), &[2, 1, 4]);
        assert_eq!(
            summed_dim1.data.as_slice(),
            vec![9.0, 12.0, 27.0, 30.0, 45.0, 48.0, 63.0, 66.0]
        );

        let summed_dim2 = tensor.sum_keepdim(2);
        assert_eq!(summed_dim2.shape(), &[2, 3, 1]);
        assert_eq!(
            summed_dim2.data.as_slice(),
            vec![40.0, 44.0, 48.0, 52.0, 56.0, 60.0]
        );
    }

    #[test]
    fn matmul_transposed_one_mat() {
        let a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new([2, 2], vec![5.0, 6.0, 7.0, 8.0]);

        let b_t = b.unwrap().t();

        let c = &a.matmul(&b_t).unwrap();
        println!("{:?}", c);

        assert_eq!(&vec![26.0, 38.0, 30.0, 44.0], c.data.as_ref());
    }

    #[test]
    fn matadd_different_strides_flipped() {
        // `a` is a standard contiguous tensor
        let a = Tensor::new([2, 2], vec![1.0, 2.0, 30.0, 40.0]).unwrap();

        // `b` has a non-contiguous (transposed) memory layout
        let b = unsafe {
            Tensor::from_raw_parts(
                1,
                [2, 2],
                Arc::new(vec![5.0, 60.0, 7.0, 80.0]),
                [2, 1],
                None,
            )
        };

        // The correct result of b + a, element-by-element, stored contiguously
        // b is logically [[5, 60], [7, 80]] and a is [[1, 30], [2, 40]]
        // Sum is [[6, 90], [9, 120]]. Column-major data is [6, 9, 90, 120]
        let expected_data = vec![6.0, 9.0, 90.0, 120.0];

        // We call b.tensoradd(&a), with the non-contiguous tensor first.
        let c = b.tensoradd(&a).unwrap();

        assert_eq!(c.get_data(), &expected_data);
    }

    #[test]
    fn test_iterators() {
        // Contiguous tensor
        let a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut a_iter = a.iter();
        assert_eq!(a_iter.next(), Some(1.0));
        assert_eq!(a_iter.next(), Some(2.0));
        assert_eq!(a_iter.next(), Some(3.0));
        assert_eq!(a_iter.next(), Some(4.0));
        assert_eq!(a_iter.next(), None);

        // Non-contiguous tensor (transposed)
        let a_t = a.t();
        let mut at_iter = a_t.iter();
        assert_eq!(at_iter.next(), Some(1.0)); // logical [0,0]
        assert_eq!(at_iter.next(), Some(3.0)); // logical [0,1]
        assert_eq!(at_iter.next(), Some(2.0)); // logical [1,0]
        assert_eq!(at_iter.next(), Some(4.0)); // logical [1,1]
        assert_eq!(at_iter.next(), None);

        // Iter with index
        let mut a_iter_idx = a.iter_with_index();
        assert_eq!(a_iter_idx.next(), Some((vec![0, 0], 1.0)));
        assert_eq!(a_iter_idx.next(), Some((vec![1, 0], 2.0)));
        assert_eq!(a_iter_idx.next(), Some((vec![0, 1], 3.0)));
        assert_eq!(a_iter_idx.next(), Some((vec![1, 1], 4.0)));
        assert_eq!(a_iter_idx.next(), None);

        println!("Iterating over tensor a:");
        for i in a.iter() {
            println!("{:?}", i);
        }
    }

    #[test]
    fn test_map_and_mul_elem() {
        let a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_t = a.t(); // non-contiguous

        // Test map
        let mapped_a = a.map(|x| x * 2.0);
        assert_eq!(mapped_a.get_data(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(mapped_a.strides, [1, 2]); // Should be contiguous

        let mapped_at = a_t.map(|x| x * 2.0);
        assert_eq!(mapped_at.get_data(), &[2.0, 6.0, 4.0, 8.0]);
        assert_eq!(mapped_at.strides, [1, 2]); // Should be contiguous

        // Test mul_elem
        let b = Tensor::new([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let mul_ab = a.mul_elem(&b);
        assert_eq!(mul_ab.get_data(), &[5.0, 12.0, 21.0, 32.0]);

        // Test with non-contiguous
        let mul_atb = a_t.mul_elem(&b);
        assert_eq!(mul_atb.get_data(), &[5.0, 18.0, 14.0, 32.0]);
    }

    #[test]
    fn test_map_inplace() {
        let mut a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        a.map_inplace(|x| x + 1.0);
        assert_eq!(a.get_data(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn test_map_inplace_panic_on_shared() {
        let mut a = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let _b = a.clone(); // Create a view, sharing the data
        a.map_inplace(|x| x + 1.0); // This should panic
    }

    #[test]
    fn test_display_impl() {
        let tensor = Tensor::new(
            [2, 2, 2, 2],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        )
        .unwrap();
        println!("{}", tensor);
        assert!(true);
    }

    //#[ignore = "Performance bench"]
    #[test]
    fn benchmark_matmul_transposed() {
        const DIM: usize = 1024;
        const TEST_COUNT: u32 = 10;

        // Stwórz dwie duże, losowe macierze. Domyślnie są ciągłe (column-major).
        let a: Tensor<f32, 2> = Tensor::random([DIM, DIM]);
        let b: Tensor<f32, 2> = Tensor::random([DIM, DIM]);

        let b_t = b.t();

        println!("\n MATRIX MATMUL BENCH");
        println!("Matrix size: {}x{}", DIM, DIM);

        let mut sum = Duration::from_secs(0);
        for _i in 0..TEST_COUNT {
            let time_a_b = std::time::Instant::now();
            let _result_a_b = a.matmul(&b).unwrap();
            let duration = time_a_b.elapsed();
            sum += duration;
        }
        println!("Option 1 (A   * B  ): {:?}", sum / TEST_COUNT);

        let mut sum = Duration::from_secs(0);
        for _i in 0..TEST_COUNT {
            let time_a_b_t = std::time::Instant::now();
            let _result_a_b_t = a.matmul(&b_t).unwrap();
            let duration = time_a_b_t.elapsed();
            sum += duration;
        }
        println!("Option 2 (A   * B^t): {:?}", sum / TEST_COUNT);

        let a_t = a.t();

        let mut sum = Duration::from_secs(0);
        for _i in 0..TEST_COUNT {
            let time_a_t_b = std::time::Instant::now();
            let _result_a_t_b = a_t.matmul(&b).unwrap();
            let duration = time_a_t_b.elapsed();
            sum += duration;
        }
        println!("Option 3 (A^t * B  ): {:?}", sum / TEST_COUNT);

        let mut sum = Duration::from_secs(0);
        for _i in 0..TEST_COUNT {
            let time_a_t_b_t = std::time::Instant::now();
            let _result_a_t_b_t = a_t.matmul(&b_t).unwrap();
            let duration = time_a_t_b_t.elapsed();
            sum += duration;
        }
        println!("Option 4 (A^t * B^t): {:?}", sum / TEST_COUNT);

        println!("MATRIX MATMUL BENCHMARK END\n");
    }

    #[test]
    fn test_select() {
        // LEt's create a 2x3x4 tensor with values from 0 to 23
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::new([2, 3, 4], data).unwrap();

        // And let's take a slice along axis 2 (the last axis) at index 1
        // we should get a 2x3 tensor
        let slice = tensor.select(2, 1).unwrap();
        assert_eq!(slice.shape(), &[2, 3]);

        // And check the data consistency. With our column-major layout, the data for `slice`
        // should be equal to the second "column" in the last dimension.
        // Original data for [:,:,1] is [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        let expected_data = vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let slice_data: Vec<f32> = slice.iter().collect();
        assert_eq!(slice_data, expected_data);

        // And let's check slice dim along axis 0 at index 1
        let slice_axis0 = tensor.select(0, 1).unwrap();
        assert_eq!(slice_axis0.shape(), &[3, 4]);

        // For [1, :, :] the data is [1,3,5,7,9,11,13,15,17,19,21,23]
        //let expected_data_axis0 = vec![
        //    1.0, 9.0, 17.0, 3.0, 11.0, 19.0, 5.0, 13.0, 21.0, 7.0, 15.0, 23.0,
        //];
        //let slice_data_axis0: Vec<f32> = slice_axis0.iter().collect();

        assert_eq!(
            slice_axis0.get(&[0, 0]).unwrap(),
            tensor.get(&[1, 0, 0]).unwrap()
        );
        assert_eq!(
            slice_axis0.get(&[1, 0]).unwrap(),
            tensor.get(&[1, 1, 0]).unwrap()
        );
        assert_eq!(
            slice_axis0.get(&[0, 1]).unwrap(),
            tensor.get(&[1, 0, 1]).unwrap()
        );

        // err check
        assert!(tensor.select(3, 0).is_err()); // bad axis
        assert!(tensor.select(0, 2).is_err()); // bad index
    }
    #[test]
    fn test_select_mut_and_modify() {
        // Tworzymy tensor 2x2x2
        let mut tensor = Tensor::new([2, 2, 2], (0..8).map(|x| x as f32).collect()).unwrap();

        // tensor (col-major):
        // Slice z=0: [[0, 2], [1, 3]]
        // Slice z=1: [[4, 6], [5, 7]]

        // 1. Wybierz mutowalnie drugi plaster wzdłuż osi Z (axis=2, index=1)
        //    Oczekujemy widoku na macierz [[4, 6], [5, 7]]
        let mut slice_z1 = tensor.select_mut(2, 1).unwrap();
        assert_eq!(slice_z1.shape(), &[2, 2]);

        // 2. Modyfikuj element w tym widoku.
        //    Chcemy zmienić element (y=0, x=1), który ma wartość 6.0, na 99.0
        //    W widoku 2x2, to jest element [0, 1]
        *slice_z1.get_mut(&[0, 1]).unwrap() = 99.0;

        // 3. Sprawdź, czy oryginalny tensor został zmodyfikowany.
        //    Oryginalny indeks to [0, 1, 1]
        //assert_eq!(tensor.get(&[0, 1, 1]).unwrap(), 99.0);
        // 4. Test łańcuchowego wycinania
        //    Wybierz pierwszy wiersz (y=0) z naszego plastra `slice_z1`
        let row_y0 = slice_z1.select(0, 0).unwrap();
        assert_eq!(row_y0.shape(), &[2]); // Oczekujemy wektora [4, 99]

        // Sprawdźmy, czy get() na tym ostatecznym widoku działa
        assert_eq!(row_y0.get(&[0]).unwrap(), 4.0);
        assert_eq!(row_y0.get(&[1]).unwrap(), 99.0);
    }

    /*
    #[test]
    fn test_permute_2d_transpose() {
        // Tensor 2x3
        let tensor = Tensor::new([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        println!("Original tensor:\n{}", tensor);

        // Transpozycja (zamiana osi 0 z 1)
        let permuted = tensor.permute(&[1, 0]);
        println!("Permuted tensor:\n{}", permuted);

        // Sprawdzenie nowego kształtu i strides
        assert_eq!(permuted.shape(), &[3, 2]);
        assert_eq!(permuted.strides(), &[tensor.strides[1], tensor.strides[0]]);

        // Weryfikacja, czy elementy są na właściwych pozycjach
        assert_eq!(permuted[&[0, 0]], 0.0); // Oryginalnie [0, 0]
        assert_eq!(permuted[&[1, 0]], 2.0); // Oryginalnie [0, 1]
        assert_eq!(permuted[&[0, 1]], 1.0); // Oryginalnie [1, 0]
        assert_eq!(permuted[&[2, 1]], 5.0); // Oryginalnie [1, 2]
    }

    #[test]
    fn test_permute_3d() {
        // Tensor 2x3x4
        let tensor = Tensor::new([2, 3, 4], (0..24).map(|x| x as f32).collect()).unwrap();
        println!("Original tensor:\n{}", tensor);

        // Permutacja z [d0, d1, d2] na [d2, d0, d1]
        let permuted = tensor.permute(&[2, 0, 1]);
        println!("Permuted tensor:\n{}", permuted);

        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(
            permuted.strides(),
            &[tensor.strides[2], tensor.strides[0], tensor.strides[1]]
        );

        // Sprawdźmy kilka kluczowych elementów
        // Oryginalny element pod indeksem [d0=1, d1=2, d2=3]
        let original_val = tensor[&[1, 2, 3]];
        // Nowy indeks tego elementu to [d2=3, d0=1, d1=2]
        assert_eq!(permuted[&[3, 1, 2]], original_val);
    }

    #[test]
    fn test_permute_4d_2last_axes() {
        let tensor = Tensor::new([2, 2, 3, 4], (0..48).map(|x| x as f32).collect()).unwrap();
        println!("Original tensor:\n{}", tensor);

        let permuted = tensor.permute(&[0, 1, 3, 2]);
        println!("Permuted tensor:\n{}", permuted);

        assert_eq!(permuted.shape(), &[2, 2, 4, 3]);
        assert_eq!(
            permuted.strides(),
            &[
                tensor.strides[0],
                tensor.strides[1],
                tensor.strides[3],
                tensor.strides[2]
            ]
        );

        // Sprawdźmy kilka kluczowych elementów
        // Oryginalny element pod indeksem [d0=1, d1=2, d2=3]
        //let original_val = tensor[&[1, 2, 3]];
        // Nowy indeks tego elementu to [d2=3, d0=1, d1=2]
        //assert_eq!(permuted[&[3, 1, 2]], original_val);
    }

    #[test]
    #[should_panic(
        expected = "Permutation error: axes length 2 does not match tensor dimensionality 3"
    )]
    fn test_permute_invalid_axes_length() {
        let tensor = Tensor::<f32, _>::zeros([2, 3, 4]).unwrap();
        tensor.permute(&[1, 0]); // Za mało osi
    }

    #[test]
    #[should_panic(expected = "Permutation error: axis 1 is duplicated")]
    fn test_permute_duplicate_axes() {
        let tensor = Tensor::<f32, _>::zeros([2, 3, 4]).unwrap();
        tensor.permute(&[0, 1, 1]); // Zduplikowana oś
    }

    #[test]
    #[should_panic(
        expected = "Permutation error: axis 3 is out of bounds for tensor with 3 dimensions"
    )]
    fn test_permute_axis_out_of_bounds() {
        let tensor = Tensor::<f32, _>::zeros([2, 3, 4]).unwrap();
        tensor.permute(&[0, 1, 3]); // Oś 3 jest poza zakresem
    }
    */

    #[test]
    fn test_flip_4d_spatial_axes() {
        // Tworzymy tensor 4D o kształcie [N, C, H, W] = [1, 2, 3, 2]
        // N - batch size, C - kanały, H - wysokość, W - szerokość
        let tensor = Tensor::new([1, 2, 3, 2], (0..12).map(|x| x as f32).collect()).unwrap();
        println!("Original tensor:\n{}", tensor);

        // Odwracamy osie przestrzenne: wysokość (oś 2) i szerokość (oś 3)
        // To jest ekwiwalent obrotu o 180 stopni dla każdego filtra 2D.
        let flipped = tensor.flip(&[0, 1]);
        println!("Flipped tensor:\n{}", flipped);

        // Kształt i strides powinny pozostać bez zmian, bo `flip` zwraca nowy, ciągły tensor
        assert_eq!(flipped.shape(), tensor.shape());

        // --- Weryfikacja kluczowych elementów ---

        assert_eq!(flipped[&[0, 1, 2, 1]], tensor[&[0, 0, 2, 1]]);

        assert_eq!(flipped[&[0, 0, 2, 1]], tensor[&[0, 1, 2, 1]]);
    }
    #[test]
    fn test_expand_broadcast_bias() {
        // Scenariusz: Mamy wektor biasu [3, 1] i chcemy go dodać do batcha [3, 4].
        // Wartości: [10, 20, 30]
        let bias = Tensor::new([3, 1], vec![10.0, 20.0, 30.0]).unwrap();

        // Rozszerzamy do [3, 4] (3 cechy, 4 przykłady w batchu)
        let expanded = bias.expand([3, 4]).unwrap();

        assert_eq!(expanded.shape(), &[3, 4]);

        // Sprawdzamy strides:
        // Wymiar 0 (wiersze) ma rozmiar 3 (bez zmian) -> stride 1
        // Wymiar 1 (kolumny) zmienił się z 1 na 4 -> stride powinien być 0!
        assert_eq!(expanded.strides(), &[1, 0]);

        // Sprawdzamy wartości w różnych kolumnach (powinny być takie same)
        // Wiersz 0 (wartość 10.0)
        assert_eq!(expanded.get(&[0, 0]), Some(10.0)); // Kolumna 0
        assert_eq!(expanded.get(&[0, 1]), Some(10.0)); // Kolumna 1
        assert_eq!(expanded.get(&[0, 3]), Some(10.0)); // Kolumna 3

        // Wiersz 2 (wartość 30.0)
        assert_eq!(expanded.get(&[2, 0]), Some(30.0));
        assert_eq!(expanded.get(&[2, 3]), Some(30.0));
    }

    #[test]
    fn test_expand_broadcast_scalar() {
        // Scenariusz: Mamy tensor 1x1 (skalar) i rozszerzamy go do 2x2.
        let scalar = Tensor::new([1, 1], vec![5.0]).unwrap();

        let expanded = scalar.expand([2, 2]).unwrap();

        assert_eq!(expanded.shape(), &[2, 2]);
        // Oba wymiary były 1, więc oba strides powinny być 0
        assert_eq!(expanded.strides(), &[0, 0]);

        // Każdy element macierzy powinien wynosić 5.0
        assert_eq!(expanded.get(&[0, 0]), Some(5.0));
        assert_eq!(expanded.get(&[1, 1]), Some(5.0));
    }

    #[test]
    fn test_expand_add_new_dim_left() {
        // Scenariusz: Mamy wektor [2] i chcemy go traktować jako [2, 2]
        // UWAGA: expand wyrównuje do prawej.
        // Źródło: [2]
        // Cel:    [2, 2]
        // Wyrównanie:
        // Cel:    dim0=2, dim1=2
        // Źródło:         dim0=2
        // Zatem dim0 celu (nowy wymiar) dostanie stride 0 (powielanie całego wektora)

        let vec = Tensor::new([2], vec![1.0, 2.0]).unwrap();
        let expanded = vec.expand([2, 2]).unwrap();

        assert_eq!(*expanded.shape(), [2, 2]);

        // Logika expand (wyrównanie do prawej):
        // Target dim 1 (rozmiar 2) odpowiada Source dim 0 (rozmiar 2) -> stride 1
        // Target dim 0 (rozmiar 2) jest nowy -> stride 0
        assert_eq!(*expanded.strides(), [0, 1]);

        // Sprawdźmy dane:
        // expanded[0, :] powinno być [1.0, 2.0]
        assert_eq!(expanded.get(&[0, 0]), Some(1.0));
        assert_eq!(expanded.get(&[0, 1]), Some(2.0));

        // expanded[1, :] powinno być TEŻ [1.0, 2.0] (powielenie)
        assert_eq!(expanded.get(&[1, 0]), Some(1.0));
        assert_eq!(expanded.get(&[1, 1]), Some(2.0));
    }

    #[test]
    fn test_expand_invalid_shapes() {
        let t = Tensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Błąd 1: Próba zmiany wymiaru, który nie jest 1 (2 -> 3)
        assert!(t.expand([2, 3]).is_err());

        // Błąd 2: Próba zmniejszenia liczby wymiarów (expand nie robi shrink/reduce)
        assert!(t.expand([2]).is_err());
    }

    #[test]
    fn test_expand_zero_copy_verification() {
        // Weryfikacja, czy faktycznie nie kopiujemy danych (czy wskaźniki są te same)
        let t = Tensor::new([1], vec![123.0]).unwrap();
        let expanded = t.expand([100, 100]).unwrap();

        // Oba tensory powinny wskazywać na ten sam obszar pamięci
        assert_eq!(t.data_ptr(), expanded.data_ptr());

        // Oryginalny tensor ma 1 element
        assert_eq!(t.data.len(), 1);
        // "Rozszerzony" tensor nadal korzysta z wektora o długości 1, mimo że logicznie ma 10000 elementów
        assert_eq!(expanded.data.len(), 1);
    }
}
