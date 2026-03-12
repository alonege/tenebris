# tenebris

A from-scratch machine learning library written in pure Rust. 

Currently, `tenebris` serves as an playground for understanding deep learning internals, backpropagation, and basic tensor operations without relying on heavy external ML frameworks. 

## Current State
The current implementation is CPU-bound and focuses on the core mathematical and structural foundations of neural networks. It features:
* Custom Tensor API: Basic multi-dimensional array operations; it uses [`faer`](https://github.com/sarah-ek/faer-rs) for optimized CPU linear algebra.
* Core Layers: `Linear`, `Conv2d`, `Softmax`, `Dropout`, `Flatten`, `Chain` (Sequential).
* Activation Functions: Standard activations integrated within the network graph.
* Optimization: Stochastic Gradient Descent (`SGD`).
* Loss Functions: `MSE`, `SoftmaxWithCrossEntropy`.

## Quick Start
You can run the existing entry points to see the library in action. Ensure you have the Rust toolchain installed.

Clone the repository and run one of the development binaries:

```bash
# Run the Multi-Layer Perceptron (MLP) -- XOR example
cargo run --manifest-path dev/Cargo.toml --release

# Run the Convolutional Neural Network (CNN) example
cargo run --manifest-path devconv2/Cargo.toml --release
```

## Roadmap / Future Research

This project is an evolving playground. The upcoming architecture changes focus on deep type safety, zero-cost abstractions, and hardware acceleration:

1. Major Architectural Refactor
   * Introduction of a generic `Backend<...>` trait to decouple tensor logic from the underlying execution engine.
   * Upgrading the `Tensor` API to utilize const generics for compile-time rank and dimensionality checking (e.g., `Tensor<B: Backend, const D: usize>`), 
   using [`burn`'s](https://burn.dev/) and [`dfdx`'s](https://github.com/chelsea0x3b/dfdx) `Tensor` as a reference for design patterns and ergonomics.
   * Ongoing analysis of memory-efficient data layouts and views.

2. GPU Acceleration & Compute Shaders
   * Implementing hardware acceleration utilizing `CubeCL` or `wgpu`.
   * Exploring the development of native Rust compute shaders for highly parallelized, custom tensor operations.

3. Automatic Differentiation Engine
   * Tracking the stabilization of LLVM-based automatic differentiation (e.g., Enzyme) in stable Rust for zero-overhead gradient computation.
   * Evaluating the implementation of a standard, dynamic computational graph (Autograd) as a pragmatic interim solution to compute $\frac{\partial \mathcal{L}}{\partial W}$ for complex, non-sequential topologies.

4. Expanded Layer & Architecture API
   * Implementation of widely used building blocks required for modern topologies (e.g., Normalization layers, Recurrent structures, Attention mechanisms).

## Academic Background & Validation

This project originated as a Bachelor's thesis (B.Sc. in Engineering) at the Faculty of Computer Science, Bialystok University of Technology (successfully defended)
under the supervision of Prof. Khalid Saeed.

The library has been experimentally validated against standard deep learning problems:
* Accuracy: Achieved 77.50% accuracy on the CIFAR-10 test set using a custom deep convolutional network (implementing the `im2col` algorithm to reduce convolution to matrix multiplication).
* Performance: Preliminary internal benchmarks demonstrate that this CPU-bound Rust implementation achieves approximately 58% of the performance of PyTorch (CPU) for equivalent operations (tested on Ryzen 5 7600x CPU). 

*Note on Benchmarks: The performance metrics are based on multiple runs with internal time measuring mechanisms. The reference Python/PyTorch scripts used for these baseline measurements are included in the repository for transparency. A transition to formal benchmarking tools like `criterion` is planned*

## State of the Project & Contributing

This is an early-stage MVP developed by a MSc student balancing academic research with severe time constraints. Because of this, the current focus is strictly on the mathematical engine and core architecture, which means the documentation is currently bare-bones (limited to the two entry points in the `dev` and `devconv2` directories and some internal code comments). 

I am actively learning and iterating on this architecture. Feedback, architectural discussions, and pull requests are highly welcome. If you spot an issue with the tensor math, have ideas for the upcoming backend refactor, or just want to improve the documentation, feel free to open an issue or a PR.
