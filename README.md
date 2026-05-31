# High-Performance Matrix Multiplication Benchmarking Suite (CPU & GPU)

## Overview
This repository provides a comprehensive benchmarking suite for evaluating the performance characteristics of Matrix Multiplication ($C = A \times B$) across various hardware architectures, including Single-core CPUs, Multi-core CPUs (OpenMP), and NVIDIA GPUs (CUDA). 

It is designed to study the impact of memory hierarchies, parallelization strategies, and hardware accelerators on computational bottlenecks in High-Performance Computing (HPC) and Machine Learning (ML) workloads.

## 🚀 Key Features

*   **Algorithmic Breadth:** Implements and compares multiple matrix multiplication strategies:
    *   **CPU Naive & Cache-Optimized:** Demonstrates the impact of loop-ordering (i-k-j) on L1/L2 cache locality.
    *   **GPU Naive (Global Memory):** Establishes a baseline for GPU execution, demonstrating memory bandwidth limitations.
    *   **GPU Tiled (Shared Memory):** Exploits fast on-chip shared memory to reduce global memory transactions, overcoming the memory wall.
    *   **Tensor Core (WMMA):** Hardware-accelerated mixed-precision (FP16/FP32) matrix multiplication using Warp Matrix Multiply Accumulate (WMMA) APIs.
*   **Precision Flexibility:** Templated C++ implementations support `int`, `float`, and `double`, dynamically handling precision casting (e.g., FP32 $\to$ FP16) for specialized hardware execution.
*   **High-Resolution Profiling:** Utilizes `std::chrono` (CPU) and `cudaEvent_t` (GPU) for microsecond-level timing and GFLOPS (Giga Floating Point Operations Per Second) computation.

## 🧮 Hardware & Performance Evaluation

The core of this project investigates how different memory access patterns and specialized compute cores impact theoretical vs. achieved GFLOPS.

$$ \text{GFLOPS} = \frac{2 \times N^3}{\text{Time (seconds)} \times 10^9} $$

### GPU Architecture Analysis
The GPU phase reveals critical insights into CUDA optimization, mapping directly to modern AI infrastructure challenges:
1.  **Memory Bound (Naive):** Redundant global memory accesses bottleneck the computation, keeping CUDA cores starved for data.
2.  **Compute Bound (Tiled):** By cooperatively loading data into Shared Memory (L1 Cache) in block tiles (e.g., $32 \times 32$), the algorithm avoids the memory wall and approaches the theoretical compute limit of standard FP32 cores.
3.  **Speed of Light (Tensor Cores):** By dispatching $16 \times 16 \times 16$ fragments directly to Tensor Cores, the throughput scales massively, reflecting the hardware paradigm driving the training and inference of modern Large Language Models (LLMs).

## 📂 Project Structure

- `phase1-matmul-singlecore/`: Baseline sequential CPU implementations and loop-ordering optimizations.
- `phase2-matmul-multicore/`: OpenMP extensions for parallel CPU execution.
- `phase3-matmul-gpu/`: CUDA implementations (Naive, Shared Memory, Tensor Cores), Jupyter notebooks, and extensive architectural documentation.

## 🛠️ Build & Run Instructions

### CPU (Single-Core) Compilation
```bash
cd phase1-matmul-singlecore
g++ -O3 -std=c++17 matmul-singlecore.cpp -o benchmark_singlecore
./benchmark_singlecore
```

### CPU (Multi-Core OpenMP) Compilation
```bash
cd phase2-matmul-multicore
g++ -O3 -fopenmp -std=c++17 matmul_multicore.cpp -o benchmark_multicore
./benchmark_multicore
```

### GPU (CUDA) Compilation
*Requires an NVIDIA GPU with Compute Capability >= 7.0 (Volta, Turing, Ampere, Ada, Hopper) for Tensor Core support.*
```bash
cd phase3-matmul-gpu
nvcc -arch=sm_75 matmul_gpu.cu -o benchmark_gpu
./benchmark_gpu
```
