# CUDA Matrix Multiplication Benchmark Suite

## 1. Overview
This codebase provides a high-performance benchmarking suite for Matrix Multiplication ($C = A \times B$) on NVIDIA GPUs. It is designed to evaluate and compare the performance characteristics of three distinct parallel algorithms:
1.  **Naive Implementation:** Baseline global memory access.
2.  **Tiled Implementation:** Optimized shared memory access to reduce memory bandwidth bottlenecks.
3.  **Tensor Core Implementation:** Hardware-accelerated matrix multiplication using the WMMA (Warp Matrix Multiply Accumulate) API.

The system is templated to support multiple data types (`int`, `float`, `double`) while automatically handling the necessary type conversions (FP32 $\to$ FP16) required for Tensor Core reference benchmarks.

---

## 2. Configuration & Tuning
The system is configurable via pre-processor macros, allowing for rapid A/B testing of different matrix sizes and data types without altering core logic.

| Macro | Default | Description |
| :--- | :--- | :--- |
| `N` | `8192` | The dimension of the square matrices ($N \times N$). **Constraint:** Must be a multiple of 16 to satisfy Tensor Core alignment requirements. |
| `BLOCK_SIZE` | `32` | Defines the thread block dimensions ($32 \times 32$). This directly controls the size of Shared Memory tiles. |
| `DATA_TYPE` | `float` | The primary data type for the Naive and Tiled benchmarks. Supported types: `int`, `float`, `double`. |

---

## 3. Algorithmic Implementations

### 3.1 Algorithm 1: Naive (Global Memory)
* **Function:** `matmul_naive`
* **Mechanism:** Direct implementation where each thread computes one element of the output matrix $C$.
* **Memory Pattern:**
    * Reads row $i$ of Matrix $A$ and column $j$ of Matrix $B$ directly from Global Memory.
* **Bottleneck:** High latency and redundant global memory accesses. This kernel is **Memory Bound**.

### 3.2 Algorithm 2: Tiled (Shared Memory)
* **Function:** `matmul_tiled`
* **Mechanism:** Implements block-tiling to exploit data locality.
* **Optimization:**
    * The matrix is divided into sub-tiles of size `BLOCK_SIZE x BLOCK_SIZE`.
    * Threads cooperatively load these tiles from high-latency Global Memory into low-latency **Shared Memory** (L1 Cache).
    * Computation is performed using the data residing in Shared Memory.
* **Synchronization:** Uses `__syncthreads()` barriers to prevent race conditions during the load-compute-store cycle.

### 3.3 Algorithm 3: Tensor Cores (WMMA)
* **Function:** `matmul_tensor_core`
* **Mechanism:** Utilizes NVIDIA's specialized Tensor Cores via the `nvcuda::wmma` API.
* **Data Flow:**
    1.  **Load:** Warps load $16 \times 16$ fragments of data from Global Memory.
    2.  **Compute:** Performs a full matrix multiply-accumulate step in a single hardware instruction cycle.
    3.  **Store:** Accumulates the result into a 32-bit float fragment and stores it back to memory.
* **Note:** This kernel strictly operates on Mixed Precision (Input: `half` / FP16, Output: `float` / FP32).

---

## 4. Code Structure & Data Flow

### Host-Side Logic (`main` & `run_complete_benchmark`)
1.  **Initialization:**
    * Allocates Host memory (`std::vector`).
    * Generates cryptographically seeded random data (range -5.0 to 5.0) using the `createMatrix` template.
2.  **Device Allocation:**
    * Allocates Device memory (`cudaMalloc`) for input matrices A, B and output C.
    * Transfers data from Host $\to$ Device (`cudaMemcpy`).
3.  **Data Preparation (Tensor Cores):**
    * Since Tensor Cores require `half` (FP16) precision, a helper function `convertToHalf` casts the input data type (`float`/`int`/`double`) into `half` precision buffers.
4.  **Execution:**
    * Kernel launches are timed using `cudaEvent_t` for high-precision microsecond resolution.
    * Grid and Block dimensions are calculated dynamically based on matrix size `N` and `BLOCK_SIZE`.

### Device-Side Kernels
* **Templating:** Kernels 1 and 2 uses C++ templates (`template <typename T>`) to allow generic execution across Integer and Floating Point types without code duplication.

---

## 5. Usage & Compilation

### Requirements
* **Compiler:** `nvcc` (NVIDIA CUDA Compiler)
* **Hardware:** Compute Capability 7.0 (Volta) or higher (Turing/Ampere recommended for Tensor Cores).

### Compilation
The code requires the `mma.h` header and linkage against the CUDA runtime.

```bash
# Compile for Turing (T4) or newer architectures
nvcc -arch=sm_75 matmul_gpu.cu -o matmul_gpu
```

## 6. Performance Interpretation
The output reports performance in **GFLOPS** (Giga Floating Point Operations Per Second).

$$
\text{GFLOPS} = \frac{2 \times N^3}{\text{Time (seconds)} \times 10^9}
$$

* **Naive:** Represents the baseline performance limit of unoptimized code (Memory Bandwidth limit).
* **Tiled:** Represents "Good" code performance (Compute limit of standard CUDA Cores).
* **Tensor Core:** Represents the "Speed of Light" – the maximum theoretical throughput of the hardware.

## 7. Source Code
The complete implementation, including all kernels and the benchmarking harness, is available in the file:

`matmul_gpu.cu`
