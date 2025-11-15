# File: matmul_single_core.cpp
Author / Maintainer: anzinmhd  
Purpose: Documentation for the single-core matrix multiplication implementations contained in matmul_single_core.cpp.
Language: C++17 (templated, header-only style functions inside single translation unit)
Compile (example):  

**if file is named matmul_single_core.cpp**  
g++ -O3 -march=native -funroll-loops -std=c++17 matmul_single_core.cpp -o matmul_single_core

## 1. Overview

This program implements multiple variants of dense matrix multiplication (single-threaded) and compares their correctness and performance:

- matmul_naive — standard triple-loop (i, j, k) implementation.
- matmul_interchanged — loop-interchanged variant (i, k, j) to improve locality for writes to C.
- transposeB + multiply (matmul_transpose) — transpose B first to turn column accesses into sequential accesses.
- matmul_blocked_naive — blocked (tiling) implementation with small j-loop unrolling.
- matmul_blocked_interchange — blocked + loop-interchanged inner kernel with unrolling.
- Small autotuners try several block sizes and pick the best for matmul_blocked_naive and matmul_blocked_interchange.

The main program builds random matrices, runs each variant, times them, compares results (with tolerances), and prints GFLOPS.

## 2. File / Symbol Map

- createMatrix<T>(int rows, int cols, unsigned int seed)
- matmul_naive<T>(const vector<T>& A, const vector<T>& B, int m, int n, int p)
- matmul_interchanged<T>(...)
- transposeB<T>(const vector<T>& B, int n, int p)
- matmul_transpose<T>(const vector<T>& A, const vector<T>& B, int m, int n, int p)
- matmul_blocked_naive<T>(const vector<T>& A, const vector<T>& B, int m, int n, int p, int bs = 128)
- matmul_blocked_interchange<T>(const vector<T>& A, const vector<T>& B, int m, int n, int p, int bs)
- almost_equal<T>(const vector<T>& X, const vector<T>& Y, double reltol = 1e-8, double abseps = 1e-12)
- sec_between(...) — timing helper
- main() — benchmark driver & autotuner

## 3. Data layout & conventions

- Row-major storage for all matrices.  
Element A[i][j] stored at A[i * cols + j].
- Template type T — typically float or double. Provided main() uses using Type = double;.
- Matrix dimensions (m, n, p):
  - A is m × n
  - B is n × p
  - C is m × p

## 4. Function reference (detailed)
### template<typename T> vector<T> createMatrix(int rows, int cols, unsigned int seed)  
Purpose: Create a rows × cols matrix filled with pseudo-random values in [-5.0, 5.0].  
Parameters  
- rows, cols — dimensions
- seed — RNG seed for reproducibility

Returns  
- vector<T> length rows * cols.
- Complexity: Time O(rows * cols).

### template<typename T> vector<T> matmul_naive(const vector<T>& A, const vector<T>& B, int m, int n, int p)
Purpose: Compute C = A × B using triple-loop (i, j, k).  
Algorithm:  

