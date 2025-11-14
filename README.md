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

