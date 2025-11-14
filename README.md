# CPU–GPU Matrix Multiplication Benchmark

This project benchmarks the performance of **matrix multiplication** across:

- **Single-core CPU**
- **Multi-core CPU (OpenMP)**
- **GPU (CUDA)**

It is designed to study how different architectures handle large matrix computations and how performance changes with different data types (**int**, **float**, **double**).

---

## 🚀 Features

- Templated matrix initialization for `int`, `float`, `double`
- Naive single-core matrix multiplication
- Cache-friendly loop ordering (i–k–j)
- Easy to extend with:
  - OpenMP multicore implementation
  - CUDA GPU kernel
  - Tensor Core acceleration
- Reproducible random matrix generation using fixed seed

---

## 📐 Matrix Dimensions

For multiplying two matrices:


The user inputs:
- `m`: rows of Matrix A
- `n`: columns of A (also rows of B)
- `p`: columns of Matrix B

---

## 🧮 Implementation Overview

### 1. **Matrix Initialization**
Uses a template function:
- `createMatrix<T>(rows, cols, seed)`
- Fills matrix with random values in range `[-5, 5]`
- Works for all numeric types

### 2. **Matrix Multiplication**
- Templated `matmul<T>()` function
- Uses linearized row-major storage (`vector<T>`)
- Loop ordering improves cache locality
- Supports all data types using safe accumulators

---

## 🏎️ Performance Benchmarking (Planned)

- [ ] Single-core CPU timing using `std::chrono`
- [ ] Multi-core CPU (OpenMP) version
- [ ] NVIDIA GPU CUDA implementation
- [ ] Tensor Core acceleration
- [ ] Compare speedups across data types

---

## 📂 Project Structure


---

## 🛠️ Build Instructions (G++)

```bash
g++ -O3 -std=c++17 main.cpp -o benchmark
./benchmark
