# Neuron

A Compiled, Differentiable, Tensor-Native Language for AI Development.

---

## Table of Contents

1. [Vision](#vision)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Usage Examples](#usage-examples)

   * [Defining a Model](#defining-a-model)
   * [Computing Gradients](#computing-gradients)
   * [Staged Execution](#staged-execution)
6. [Project Structure](#project-structure)
7. [Roadmap](#roadmap)
8. [Contributing](#contributing)
9. [License](#license)

---

## Vision

Neuron is a statically typed, mid-to-high level programming language purpose-built for modern AI development. It brings first-class differentiable programming, tensor-type safety, and stage-aware execution into a compiled language with real performance guarantees. Neuron targets developers and researchers building machine learning models, pipelines, and autonomous agents who demand safety, performance, and expressive power without sacrificing clarity or ergonomics.

## Key Features

* **Differentiable Functions**: `grad(fn)` produces symbolic gradients without external libraries.
* **Tensor-Aware Type System**: Static shape and data type checking for `Tensor[DType, Shape]`.
* **Staged Execution**: Separate `TRAIN` and `INFER` phases for optimized compilation.
* **Compiled via LLVM/MLIR**: Native code generation for CPU, with future GPU support.
* **Rust-Like Syntax**: Concise, expression-oriented, and familiar to systems programmers.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Mohammadreza-Farkhondeh/neuron.git
cd neuron

# Build the interpreter prototype
cargo build --release

# Run the example
./target/release/neuron examples/linear_model.nrn
```

## Installation

Neuron requires:

* Rust (1.70+)
* CMake
* LLVM (12+)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install LLVM on Ubuntu
sudo apt-get install llvm-12-dev cmake build-essential
```

## Usage Examples

### Defining a Model

```nrn
fn model(x: Tensor[Float32, [batch, 3]]) -> Tensor[Float32, [batch, 1]] {
  let w: Tensor[Float32, [3, 1]] = init_weights();
  return matmul(x, w);
}
```

### Computing Gradients

```nrn
fn loss(x: Tensor[Float32, [batch, 3]], y: Tensor[Float32, [batch, 1]]) -> Float32 {
  let pred = model(x);
  mse(pred, y)
}

let grad_loss = grad(loss);
```

### Staged Execution

```nrn
stage TRAIN {
  fn step(x, y) {
    let l = loss(x, y);
    optimizer.step(l);
  }
}

stage INFER {
  fn predict(x) {
    model(x)
  }
}
```

## Project Structure

```
neuron/
├── Cargo.toml       # Rust project file
├── src/             # Source code
├── examples/        # Example `.nrn` scripts
├── docs/            # RFCs and design docs
└── tests/           # Unit and integration tests
```

## Roadmap

* **v0.1**: Parser + AST, interpreter prototype with CPU-only autograd
* **v0.2**: Static type checker, shape inference, MLIR dialect
* **v0.3**: LLVM backend, native codegen
* **v0.4**: GPU kernel support (MLIR/Triton)
* **v1.0**: Stable release with package manager, documentation site

## Contributing

We welcome contributions! Please:

1. Fork the repo and create a new branch (`git checkout -b feature/foo`).
2. Commit your changes (`git commit -am 'Add foo feature'`).
3. Push to the branch (`git push origin feature/foo`).
4. Open a Pull Request.

Please adhere to the existing coding style and include tests for new functionality.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
