#![feature(generic_const_exprs)]
#![feature(array_try_from_fn)]
#![feature(f16)]
#![allow(incomplete_features)]

pub mod backend;
pub mod backend_implementation;
pub mod batch_builder;
pub mod data_augmentation;
pub mod error;
pub mod errorfn;
pub mod initialization;
pub mod layer;
pub mod optimizer;
pub mod tensor;
pub mod tensor2;
pub mod tensor_number;
pub mod tensorid;
pub mod utils;

/// system powstał na Wydziale Informatyki Politechniki Białostockiej
/// ramach pracy inżynierskiej napisanej przez Kacpra Hącia.

// Abandon hope, all ye who enter here

#[cfg(test)]
mod tests {
    //use super::*;
    //use crate::layer::model::Model;
}
