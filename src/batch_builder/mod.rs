use rand::{rng, seq::SliceRandom};

use crate::{
    data_augmentation::AugmentationOptions,
    error::LibError,
    tensor::tensor::{Tensor, TensorFloat},
};

pub struct BatchBuilder;

impl BatchBuilder {
    pub fn new() -> Self {
        BatchBuilder
    }

    pub fn build_batches<T: TensorFloat>(
        data: &Vec<Tensor<T>>,
        shuffle: bool,
        batch_size: usize,
    ) -> Result<Vec<Tensor<T>>, LibError> {
        let mut batches = Vec::new();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = rng();

        if shuffle {
            indices.shuffle(&mut rng);
        }

        let mut current_batch_content_counter = 0;
        let mut current_batch_data: Vec<T> =
            Vec::with_capacity(batch_size * data[0].get_data().len());
        for i in 0..data.len() {
            let tensor_data = data[indices[i]].get_data();
            current_batch_data.extend_from_slice(tensor_data);
            current_batch_content_counter += 1;
            if current_batch_content_counter == batch_size || i == data.len() - 1 {
                let mut shape = data[0]
                    .shape()
                    .iter()
                    .map(|&dim| dim as usize)
                    .collect::<Vec<usize>>();
                let shape_len = shape.len();
                shape[shape_len - 1] = current_batch_content_counter;
                let batch_tensor = Tensor::new(shape, current_batch_data.clone())?;
                batches.push(batch_tensor);
                current_batch_data.clear();
                current_batch_content_counter = 0;
            }
        }

        Ok(batches)
    }

    pub fn build_batches_with_labels<T: TensorFloat>(
        data: &Vec<Tensor<T>>,
        labels: &Vec<Tensor<T>>,
        shuffle: bool,
        batch_size: usize,
    ) -> Result<(Vec<Tensor<T>>, Vec<Tensor<T>>), LibError> {
        let mut batches = Vec::new();
        let mut batches_labels = Vec::new();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = rng();

        if shuffle {
            indices.shuffle(&mut rng);
        }

        let mut current_batch_content_counter = 0;
        let mut current_batch_data: Vec<T> =
            Vec::with_capacity(batch_size * data[0].get_data().len());
        let mut current_batch_labels: Vec<T> =
            Vec::with_capacity(batch_size * labels[0].get_data().len());
        for i in 0..data.len() {
            let tensor_data = data[indices[i]].get_data();
            current_batch_data.extend_from_slice(tensor_data);

            let label_data = labels[indices[i]].get_data();
            current_batch_labels.extend_from_slice(label_data);

            current_batch_content_counter += 1;

            if current_batch_content_counter == batch_size || i == data.len() - 1 {
                let mut shape = data[0]
                    .shape()
                    .iter()
                    .map(|&dim| dim as usize)
                    .collect::<Vec<usize>>();
                let shape_len = shape.len();
                shape[shape_len - 1] = current_batch_content_counter;
                let batch_tensor = Tensor::new(shape, current_batch_data.clone())?;
                batches.push(batch_tensor);

                let mut shape_labels = labels[0]
                    .shape()
                    .iter()
                    .map(|&dim| dim as usize)
                    .collect::<Vec<usize>>();
                let shape_labels_len = shape_labels.len();
                shape_labels[shape_labels_len - 1] = current_batch_content_counter;
                let batch_labels_tensor = Tensor::new(shape_labels, current_batch_labels.clone())?;
                batches_labels.push(batch_labels_tensor);

                current_batch_data.clear();
                current_batch_labels.clear();
                current_batch_content_counter = 0;
            }
        }

        Ok((batches, batches_labels))
    }

    pub fn build_batches_with_labels2<T: TensorFloat>(
        data: &Vec<Tensor<T>>,
        labels: &Vec<Tensor<T>>,
        shuffle: bool,
        batch_size: usize,
        augmentation: AugmentationOptions,
    ) -> Result<Vec<(Tensor<T>, Tensor<T>)>, LibError> {
        let mut batches = Vec::new();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = rng();

        if shuffle {
            indices.shuffle(&mut rng);
        }

        let mut current_batch_content_counter = 0;
        let mut current_batch_data: Vec<T> =
            Vec::with_capacity(batch_size * data[0].get_data().len());
        let mut current_batch_labels: Vec<T> =
            Vec::with_capacity(batch_size * labels[0].get_data().len());
        for i in 0..data.len() {
            let tensor_data = data[indices[i]].get_data();
            let mut tensor_data_new = vec![T::from(0.0).unwrap(); tensor_data.len()];
            crate::data_augmentation::copy_with_optional_augment(
                tensor_data,
                &mut tensor_data_new,
                augmentation.c,
                augmentation.h,
                augmentation.w,
            );
            current_batch_data.extend_from_slice(tensor_data);

            let label_data = labels[indices[i]].get_data();
            current_batch_labels.extend_from_slice(label_data);

            current_batch_content_counter += 1;

            if current_batch_content_counter == batch_size || i == data.len() - 1 {
                let mut shape = data[0]
                    .shape()
                    .iter()
                    .map(|&dim| dim as usize)
                    .collect::<Vec<usize>>();
                let shape_len = shape.len();
                shape[shape_len - 1] = current_batch_content_counter;
                let batch_tensor = Tensor::new(shape, current_batch_data.clone())?;

                let mut shape_labels = labels[0]
                    .shape()
                    .iter()
                    .map(|&dim| dim as usize)
                    .collect::<Vec<usize>>();
                let shape_labels_len = shape_labels.len();
                shape_labels[shape_labels_len - 1] = current_batch_content_counter;
                let batch_labels_tensor = Tensor::new(shape_labels, current_batch_labels.clone())?;

                batches.push((batch_tensor, batch_labels_tensor));

                current_batch_data.clear();
                current_batch_labels.clear();
                current_batch_content_counter = 0;
            }
        }

        Ok(batches)
    }
}
