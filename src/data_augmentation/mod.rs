use rand::Rng;

use crate::tensor::tensor::TensorFloat;

#[derive(Clone)]
pub struct AugmentationOptions {
    pub swap_vertical: bool,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl AugmentationOptions {
    pub fn new(swap_vertical: bool, c: usize, h: usize, w: usize) -> Self {
        AugmentationOptions {
            swap_vertical,
            c,
            h,
            w,
        }
    }
}

// src: surowe dane jednego obrazka z datasetu (np. [3*32*32])
// dst: wycinek pamięci w tensorze batcha przeznaczony na ten obrazek
pub fn copy_with_optional_augment<T>(src: &[T], dst: &mut [T], c: usize, h: usize, w: usize)
where
    T: TensorFloat,
{
    let mut rng = rand::rng();
    let do_flip = rng.random_bool(0.5);

    if !do_flip {
        dst.copy_from_slice(src);
    } else {
        for channel in 0..c {
            for row in 0..h {
                let offset = channel * h * w + row * w;

                let src_row = &src[offset..offset + w];
                let dst_row = &mut dst[offset..offset + w];

                for (d, s) in dst_row.iter_mut().zip(src_row.iter().rev()) {
                    *d = *s;
                }
            }
        }
    }
}
