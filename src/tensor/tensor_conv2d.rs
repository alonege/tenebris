use crate::error::LibError;
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::TensorFloat;
use crate::tensor::tensorops::MatMul;
use crate::tensor::tensorops::TensorConv2D;
use rayon::prelude::*;

// Abandon hope, all ye who enter here

impl<T> TensorConv2D<Tensor<T>> for Tensor<T>
where
    T: TensorFloat,
{
    fn conv2d(
        &self,
        images: Tensor<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor<T>, LibError> {
        // c - number of channels
        // w_in - width of input
        // h_in - height of input
        // n - number of samples (in batch)
        let (image_c, image_w_in, image_h_in, image_n) = (
            images.shape()[0],
            images.shape()[1],
            images.shape()[2],
            images.shape()[3],
        );
        let (stride_h, stride_w) = stride;
        let (padding_h, padding_w) = padding;
        let (filter_h, filter_w, filter_c_in, filter_c_out) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        if image_c != filter_c_in {
            panic!("Input channels do not match filter channels");
        }
        let h_out = (image_h_in - filter_h + 2 * padding_h) / stride_h + 1;
        let w_out = (image_w_in - filter_w + 2 * padding_w) / stride_w + 1;

        // We will be performing KER^T x IM^T = out^T
        // First, we have to create filter matrix
        // let flter_mat_time = std::time::Instant::now();

        //println!("");
        let filter_mat_data: Vec<T> = (0..filter_c_out)
            .flat_map(|c_out| {
                (0..filter_c_in).flat_map(move |c_in| {
                    (0..filter_h).flat_map(move |h_f| {
                        (0..filter_w).map(move |w_f| {
                            let offset = self.physical_offset(&[h_f, w_f, c_in, c_out]);
                            //println!(" ({offset}): {} ", self.get_data()[offset as usize]);
                            self.get_data()[offset as usize]
                        })
                    })
                })
            })
            .collect();

        //println!(
        //    "Filter Mat Time: {:?}",
        //    std::time::Instant::now().duration_since(flter_mat_time)
        //);
        let filter_mat = unsafe {
            Tensor::new_raw(
                vec![filter_c_out, filter_c_in * filter_h * filter_w],
                filter_mat_data,
                vec![(filter_c_in * filter_h * filter_w) as isize, 1],
            )
        };
        /*
                println!(
                    "Filter Mat Time (with creation): {:?}",
                    std::time::Instant::now().duration_since(flter_mat_time)
                );
                println!("Filter Mat: {}", filter_mat);
                println!(
                    "Filter Mat Shape: {:?}",
                    vec![filter_c_out, filter_c_in * filter_h * filter_w]
                );
        */

        let image_mat_time = std::time::Instant::now();

        // OLD VALUES FOR ONE BIG IM2COL
        //let patch_size = image_c * filter_h * filter_w;
        //let num_patches = image_n * h_out * w_out;
        let patch_size = image_c * filter_h * filter_w;
        let num_patches = h_out * w_out;
        //let mut image_mat_data = vec![T::zero(); num_patches * patch_size];
        let mut image_mat_data: Vec<T> = Vec::with_capacity(num_patches * patch_size);
        unsafe {
            image_mat_data.set_len(num_patches * patch_size);
        }

        if image_c != filter_c_in {
            panic!(
                "Niezgodność kanałów: obraz ma {}, a filtr oczekuje {}",
                image_c, filter_c_in
            );
        }

        #[cfg(debug_assertions)]
        println!("Image out size: ({}, {})", h_out, w_out);

        let patch_size = image_c * filter_h * filter_w;
        let num_patches = h_out * w_out;
        let mut image_mat_data = vec![T::zero(); num_patches * patch_size];

        image_mat_data
            .par_chunks_mut(patch_size)
            .enumerate()
            .for_each(|(patch_idx, patch_slice)| {
                let i_out_h = (patch_idx / w_out) % h_out;
                let j_out_w = patch_idx % w_out;

                let mut current_pos = 0;
                for c in 0..image_c {
                    for h_f in 0..filter_h {
                        for w_f in 0..filter_w {
                            let h =
                                (i_out_h * stride_h) as isize + h_f as isize - padding_h as isize;
                            let w =
                                (j_out_w * stride_w) as isize + w_f as isize - padding_w as isize;

                            if h >= 0
                                && h < image_h_in as isize
                                && w >= 0
                                && w < image_w_in as isize
                            {
                                let offset =
                                    images.physical_offset(&[c, w as usize, h as usize, 0]);
                                patch_slice[current_pos] = images.get_data()[offset as usize];
                            }
                            // TODO: Zróbmy to tańsze -- zmieńmy inicjalizację zerami na uninit i
                            // wrzucajmy zero gdy trzeba dla paddingu, mniej cykli na niepotrzebne
                            // wypełnianie zerami -- dokładniej, tu byłby 'else { patch_slice[current_pos] = T::zero() }'

                            current_pos += 1;
                        }
                    }
                }
            });

        #[cfg(debug_assertions)]
        println!(
            "Image Mat Time: {:?}",
            std::time::Instant::now().duration_since(image_mat_time)
        );
        let image_shape = vec![image_c * filter_h * filter_w, h_out * w_out * image_n];

        #[cfg(debug_assertions)]
        println!("h_out: {}, w_out: {}", h_out, w_out);
        #[cfg(debug_assertions)]
        println!("Shape: {:?}", image_shape);
        let image_mat = Tensor::new(image_shape, image_mat_data).unwrap();

        #[cfg(debug_assertions)]
        println!("Image Mat: {}", image_mat);

        let matmul_time = std::time::Instant::now();
        let output_image = filter_mat.matmul(&image_mat).unwrap();
        #[cfg(debug_assertions)]
        println!(
            "MatMul Time: {:?}",
            std::time::Instant::now().duration_since(matmul_time)
        );
        #[cfg(debug_assertions)]
        println!("Output Image Before Reshape: {}", output_image);
        let output_image = output_image
            .reshape(vec![filter_c_out, w_out, h_out, image_n])
            .unwrap();
        #[cfg(debug_assertions)]
        println!("Output Image Mat: {}", output_image);
        #[cfg(debug_assertions)]
        println!(
            "Total Conv2D matmul Time: {:?}",
            std::time::Instant::now().duration_since(matmul_time)
        );
        Ok(output_image)
    }

    fn im2col(
        images: Tensor<T>,
        w_kernel: usize,
        h_kernel: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor<T>, LibError> {
        let (c_image_in, w_image_in, h_image_in, n) = (
            images.shape()[0],
            images.shape()[1],
            images.shape()[2],
            images.shape()[3],
        );
        let (stride_h, stride_w) = stride;
        let (padding_h, padding_w) = padding;
        let h_out = (h_image_in - h_kernel + 2 * padding_h) / stride_h + 1;
        let w_out = (w_image_in - w_kernel + 2 * padding_w) / stride_w + 1;

        let patch_size = c_image_in * h_kernel * w_kernel;
        let num_patches_per_image = h_out * w_out;
        let num_patches = num_patches_per_image * n;
        let mut image_mat_data = vec![T::zero(); num_patches * patch_size];

        image_mat_data
            .par_chunks_mut(patch_size)
            .enumerate()
            .for_each(|(patch_idx, patch_slice)| {
                let n_idx = patch_idx / num_patches_per_image;
                let i_out_h = (patch_idx / w_out) % h_out;
                let j_out_w = patch_idx % w_out;

                let mut current_pos = 0;
                for c in 0..c_image_in {
                    for h_f in 0..h_kernel {
                        for w_f in 0..w_kernel {
                            let h =
                                (i_out_h * stride_h) as isize + h_f as isize - padding_h as isize;
                            let w =
                                (j_out_w * stride_w) as isize + w_f as isize - padding_w as isize;

                            if h >= 0
                                && h < h_image_in as isize
                                && w >= 0
                                && w < w_image_in as isize
                            {
                                let offset =
                                    images.physical_offset(&[c, w as usize, h as usize, n_idx]);
                                patch_slice[current_pos] = images.get_data()[offset as usize];
                            }

                            current_pos += 1;
                        }
                    }
                }
            });

        let image_shape = vec![c_image_in * h_kernel * w_kernel, h_out * w_out * n];
        let x_col = Tensor::new(image_shape, image_mat_data)?;

        Ok(x_col)
    }

    /// # kn2k_mat
    /// function takes kernels for convolution and changes them into kn2k_mat matrix
    /// for usage in conv2d-im2col approach.
    /// ## Arguments
    /// filters: Kernel tensor of shape (C_in, W_kernel, H_kernel, C_out)
    /// filters dont't have to be continous, but tensor have to use shape convention.
    /// ## Returns
    /// matrix of shape (C_out, C_in * H_kernel * W_kernel)
    /// ## error
    /// for now, we only broadcast errors
    fn kn2k_mat(filters: Tensor<T>) -> Result<Tensor<T>, LibError> {
        let (c_in, w_kernel, h_kernel, c_out) = (
            filters.shape()[0],
            filters.shape()[1],
            filters.shape()[2],
            filters.shape()[3],
        );

        // we cannot use this approach, as we future-profed kernel convention for kn2row
        /*
        let k_mat = filters
            .make_contiguous()?
            .reshape(vec![c_out, c_in * h_kernel * w_kernel])?;
        // */
        // /*
        let filt = &filters;
        // /*
        let filter_mat_data: Vec<T> = (0..c_out)
            .flat_map(|c_out| {
                (0..c_in).flat_map(move |c_in| {
                    (0..h_kernel).flat_map(move |h_f| {
                        (0..w_kernel).map(move |w_f| {
                            let offset = filt.physical_offset(&[c_in, w_f, h_f, c_out]);
                            #[cfg(debug_assertions)]
                            println!(
                                "[{c_in}, {w_f}, {h_f}, {c_out}] ({offset}): {} ",
                                filt.get_data()[offset as usize]
                            );

                            filt.get_data()[offset as usize]
                        })
                    })
                })
            })
            .collect();
        let k_mat = unsafe {
            Tensor::new_raw(
                vec![c_out, c_in * h_kernel * w_kernel],
                filter_mat_data,
                vec![(c_in * h_kernel * w_kernel) as isize, 1],
            )
        };
        // */
        #[cfg(debug_assertions)]
        println!("K Mat: {}", k_mat);
        Ok(k_mat)
    }

    /// Input shape = [w_kernel * h_kernel * c_in, w_out * h_out * n]
    /// Output shape = [c, w, h, n]
    fn col2im(
        x_col: Tensor<T>,
        image_shape: (usize, usize, usize, usize), // (C_in, W_in, H_in, N)
        w_kernel: usize,
        h_kernel: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor<T>, LibError> {
        let (c, w, h, n) = image_shape;
        let (stride_h, stride_w) = stride;
        let (padding_h, padding_w) = padding;

        let h_out = (h - h_kernel + 2 * padding_h) / stride_h + 1;
        let w_out = (w - w_kernel + 2 * padding_w) / stride_w + 1;

        let patch_size = c * h_kernel * w_kernel;
        let num_patches = h_out * w_out * n;

        assert_eq!(
            x_col.shape(),
            &[patch_size, num_patches],
            "col2im: shape mismatch"
        );

        let mut image_grad_data = vec![T::zero(); c * w * h * n];

        x_col
            .get_data()
            .chunks(patch_size)
            .enumerate()
            .for_each(|(patch_idx, patch_slice)| {
                // te same współrzędne co w im2col()
                let i_out_h = (patch_idx / w_out) % h_out;
                let j_out_w = patch_idx % w_out;
                // batch index (jeśli batch > 1)
                let n_idx = patch_idx / (h_out * w_out);

                let mut current_pos = 0;
                for c_ker in 0..c {
                    for h_f in 0..h_kernel {
                        for w_f in 0..w_kernel {
                            let h_loc =
                                (i_out_h * stride_h) as isize + h_f as isize - padding_h as isize;
                            let w_loc =
                                (j_out_w * stride_w) as isize + w_f as isize - padding_w as isize;

                            if h_loc >= 0 && h_loc < h as isize && w_loc >= 0 && w_loc < w as isize
                            {
                                //                                let offset = ((c * w * h * n)
                                //                                    + (w_loc as usize * h * n)
                                //                                    + (h_loc as usize * n)
                                //                                    + n_idx) as usize;
                                let offset = c_ker
                                    + (w_loc as usize) * c
                                    + (h_loc as usize) * c * w
                                    + n_idx * c * w * h;

                                // UWAGA: piksel może być odwiedzony wielokrotnie,
                                // więc sumujemy wkłady (np. z różnych patchy)
                                image_grad_data[offset] =
                                    image_grad_data[offset] + patch_slice[current_pos];
                            }

                            current_pos += 1;
                        }
                    }
                }
            });

        let result = Tensor::new(vec![c, w, h, n], image_grad_data)?;
        Ok(result)
    }
}

#[allow(unused_imports)]
#[allow(deprecated)]
pub mod tests {
    use crate::tensor::tensor::Tensor;
    use crate::tensor::tensor::TensorFloat;
    use crate::tensor::tensorops::TensorConv2D;

    #[test]
    fn test_conv2d() {
        //let image = Tensor::from_fn(vec![5, 5, 1, 1], |_| 1.0).unwrap();
        let image = Tensor::new(
            vec![1, 5, 5, 1],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);
        let kernel = Tensor::new_row(
            vec![3, 3, 1, 1],
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        println!("Kernel: {}", kernel);
        let out = kernel.conv2d(image, (1, 1), (0, 0)).unwrap();
        let out_expected = Tensor::new(
            vec![1, 3, 3, 1],
            vec![7.0, 9.0, 11.0, 17.0, 19.0, 21.0, 27.0, 29.0, 31.0],
        )
        .unwrap();
        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
        assert_eq!(out.strides(), out_expected.strides());
    }

    #[test]
    fn test_conv2d_single_image_2c_in() {
        // Obraz wejściowy o kształcie [C, W, H], czyli [1, 5, 5]
        let image = Tensor::new(
            vec![2, 5, 5, 1],
            vec![
                0.0, 25.0, 1.0, 26.0, 2.0, 27.0, 3.0, 28.0, 4.0, 29.0, 5.0, 30.0, 6.0, 31.0, 7.0,
                32.0, 8.0, 33.0, 9.0, 34.0, 10.0, 35.0, 11.0, 36.0, 12.0, 37.0, 13.0, 38.0, 14.0,
                39.0, 15.0, 40.0, 16.0, 41.0, 17.0, 42.0, 18.0, 43.0, 19.0, 44.0, 20.0, 45.0, 21.0,
                46.0, 22.0, 47.0, 23.0, 48.0, 24.0, 49.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);

        // Jądro konwolucji (filtr)
        // /*
        let kernel = Tensor::new_row(
            vec![3, 3, 2, 2], // [H_f, W_f, C_in, C_out]
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        println!("Kernel: {}", kernel);

        // Wywołujemy funkcję, przekazując plaster z jednym obrazem
        let out = kernel.conv2d(image, (1, 1), (0, 0)).unwrap();

        println!("Output: {}", out);
        // Oczekiwany wynik
        let out_expected = Tensor::new(
            vec![2, 3, 3, 1], // [C_out, W_out, H_out]
            vec![
                34.0, 31.0, 37.0, 32.0, 40.0, 33.0, 49.0, 36.0, 52.0, 37.0, 55.0, 38.0, 64.0, 41.0,
                67.0, 42.0, 70.0, 43.0,
            ],
        )
        .unwrap();

        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
    }

    #[test]
    fn test_conv2d_padding() {
        let image = Tensor::new(
            vec![1, 5, 5, 1],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);

        // Jądro konwolucji (filtr)
        let kernel = Tensor::new_row(
            vec![3, 3, 1, 1], // [H_f, W_f, C_in, C_out]
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        println!("Kernel: {}", kernel);

        // Wywołujemy funkcję, przekazując plaster z jednym obrazem
        let out = kernel.conv2d(image, (1, 1), (1, 1)).unwrap();

        println!("Output: {}", out);
        println!("Output data: {:?}", out.get_data());
        // Oczekiwany wynik
        let out_expected = Tensor::new(
            vec![1, 5, 5, 1], // [C_out, W_out, H_out]
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
                25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0,
            ],
        )
        .unwrap();

        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
    }

    #[test]
    fn test_conv2d_padding_strides() {
        let image = Tensor::new(
            vec![1, 5, 5, 1],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);

        // Jądro konwolucji (filtr)
        let kernel = Tensor::new_row(
            vec![3, 3, 1, 1], // [H_f, W_f, C_in, C_out]
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        println!("Kernel: {}", kernel);

        // Wywołujemy funkcję, przekazując plaster z jednym obrazem
        let out = kernel.conv2d(image, (2, 2), (1, 1)).unwrap();

        println!("Output: {}", out);
        println!("Output data: {:?}", out.get_data());
        // Oczekiwany wynik
        let out_expected = Tensor::new(
            vec![1, 3, 3, 1], // [C_out, W_out, H_out]
            vec![0.0, 2.0, 4.0, 15.0, 19.0, 23.0, 35.0, 39.0, 43.0],
        )
        .unwrap();

        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
    }

    #[test]
    fn test_conv2d_padding_uneven_strides() {
        let image = Tensor::new(
            vec![1, 5, 5, 1],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        )
        .unwrap();
        println!("Image: {}", image);

        // Jądro konwolucji (filtr)
        let kernel = Tensor::new_row(
            vec![3, 3, 1, 1], // [H_f, W_f, C_in, C_out]
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        println!("Kernel: {}", kernel);

        // Wywołujemy funkcję, przekazując plaster z jednym obrazem
        let out = kernel.conv2d(image, (1, 2), (1, 1)).unwrap();

        println!("Output: {}", out);
        println!("Output data: {:?}", out.get_data());
        // Oczekiwany wynik
        let out_expected = Tensor::new(
            vec![1, 3, 5, 1], // [C_out, W_out, H_out]
            vec![
                0.0, 2.0, 4.0, 5.0, 9.0, 13.0, 15.0, 19.0, 23.0, 25.0, 29.0, 33.0, 35.0, 39.0, 43.0,
            ],
        )
        .unwrap();

        assert_eq!(out.get_data(), out_expected.get_data());
        assert_eq!(out.shape(), out_expected.shape());
    }

    #[ignore = "For now, batch isn't supported"]
    #[test]
    fn test_conv2d_big_1() {
        let image: Tensor<f64> = Tensor::random([3, 1024, 1024, 5]);
        let kernel: Tensor<f64> = Tensor::random([5, 5, 3, 16]);
        let time_start = std::time::Instant::now();
        let out = kernel.conv2d(image, (1, 1), (0, 0)).unwrap();
        println!(
            "Conv2D big 1 Time: {:?}",
            std::time::Instant::now().duration_since(time_start)
        );
        assert_eq!(out.shape(), vec![16, 1020, 1020, 5]);
    }

    #[ignore = "For now, batch isn't supported"]
    #[test]
    fn test_conv2d_big_2() {
        let image: Tensor<f64> = Tensor::random([3, 1024, 1024, 5]);
        let kernel: Tensor<f64> = Tensor::random([7, 7, 3, 16]);
        let time_start = std::time::Instant::now();
        let out = kernel.conv2d(image, (1, 1), (0, 0)).unwrap();
        println!(
            "Conv2D big 2 Time: {:?}",
            std::time::Instant::now().duration_since(time_start)
        );
        assert_eq!(out.shape(), vec![16, 1018, 1018, 5]);
    }
}
