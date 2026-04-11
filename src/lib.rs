#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod backend;
pub mod batch_builder;
pub mod data_augmentation;
pub mod error;
pub mod errorfn;
pub mod initialization;
pub mod layer;
pub mod optimizer;
pub mod tensor;
pub mod utils;

/// system powstał na Wydziale Informatyki Politechniki Białostockiej
/// ramach pracy inżynierskiej napisanej przez Kacpra Hącia.

// Abandon hope, all ye who enter here

#[cfg(test)]
mod tests {
    //use super::*;
    //use crate::layer::model::Model;
}
