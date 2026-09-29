use std::hint::black_box;

use crate::error::LibError;

#[derive(Debug, Clone, Copy)]
pub struct Layout<const D: usize> {
    offset: usize,
    shape: [usize; D],
    strides: [isize; D],
}

impl<const D: usize> Layout<D> {
    /// Creates layout for column-major data (fortran-style). It means that
    /// the first dimension is contiguous in memory, and the other dimensions
    /// are strided. For example, let's look at 2D tensor with shape [3, 4] (3 rows, 4 columns):
    /// ```text
    /// 0  3  6  9
    /// 1  4  7 10
    /// 2  5  8 11
    /// ```
    /// In the memory, we have the following layout: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].
    /// The strides for this layout are [1, 3], which means that to move to the next element
    /// in the first dimension (rows), we need to move by 1 in memory, and to move to the next
    /// element in the second dimension (columns), we need to move by 3 in memory.
    pub fn new_column_major(shape: [usize; D]) -> Self {
        let (_, strides) = shape.into_iter().enumerate().fold(
            (1isize, [0isize; D]),
            |(acc, mut strides), (i, dim)| {
                strides[i] = acc;
                (acc * dim as isize, strides)
            },
        );

        Self {
            offset: 0,
            shape,
            strides,
        }
    }

    /// Creates layout for row-major data (C-style). It means that
    /// the last dimension is contiguous in memory, and the other dimensions
    /// are strided. For example, let's look at 2D tensor with shape [3, 4] (3 rows, 4 columns):
    /// ```text
    /// 0  3  6  9
    /// 1  4  7 10
    /// 2  5  8 11
    /// ```
    /// In the memory, we have the following layout: [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11].
    /// The strides for this layout are [4, 1], which means that to move to the next element
    /// in the first dimension (rows), we need to move by 4 in memory, and to move to the next
    /// element in the second dimension (columns), we need to move by 1 in memory.
    pub fn new_row_major(shape: [usize; D]) -> Self {
        let mut strides = [0isize; D];

        if D > 0 {
            strides[D - 1] = 1;
            for i in (0..D - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1] as isize;
            }
        }

        Self {
            offset: 0,
            shape,
            strides,
        }
    }

    pub fn new_custom(offset: usize, shape: [usize; D], strides: [isize; D]) -> Self {
        Self {
            offset,
            shape,
            strides,
        }
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }

    #[inline(always)]
    pub fn strides(&self) -> &[isize; D] {
        &self.strides
    }

    /// get physical offset from beginning of data for logical index.
    ///
    /// # Returns
    /// - Some(offset) if logical index is valid
    /// - None if out of bounds.
    pub fn physical_offset(&self, logical_index: &[usize; D]) -> Option<usize> {
        logical_index
            .iter()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
            .try_fold(self.offset as isize, |acc, ((&ind, &dim), &stride)| {
                (ind < dim).then(|| acc + ind as isize * stride as isize)
            })
            .and_then(|offset| usize::try_from(offset).ok())
    }

    /// get physical offset from beginning of data for logical index.
    ///
    /// # Safety
    /// This function does not check if the logical index is valid.
    /// This function does not dereference any data, but the output is used to do so.
    pub fn physical_offset_unchecked(&self, logical_index: &[usize; D]) -> usize {
        logical_index
            .iter()
            .zip(self.strides.iter())
            .fold(self.offset, |acc, (&ind, &stride)| {
                (acc as isize + ind as isize * stride) as usize
            })
    }

    /// Returns a new layout with dimensions with size equal 1 expanded
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
    #[inline(always)]
    pub fn expand(&self, shape: [usize; D]) -> Result<Self, LibError> {
        let strides = std::array::try_from_fn(|i| match (self.shape[i], shape[i]) {
            (src, tgt) if src == tgt => Ok(self.strides[i]),
            (1, _tgt) => Ok(0),
            _ => Err(LibError::ShapeMismatch {
                operation: "expand (incompatible dimension)".to_string(),
                expected: shape.to_vec(),
                actual: self.shape.to_vec(),
            }),
        })?;

        Ok(Self {
            offset: self.offset,
            shape,
            strides,
        })
    }
}
