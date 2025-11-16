# File: matmul_single_core.cpp
## Matrix Multiplication — Single Core (C++17)

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
```cpp
for (i = 0; i < m; i++) {
  for (j = 0; j < p; j++) {
    sum = 0;
    for (k = 0; k < n; k++) {
      sum += A[i*n + k] * B[k*p + j];
    }
    C[i*p + j] = sum;
  }
}
```
Complexity: Time O(m * n * p).  
Notes: Uses arow pointer per i for slightly better locality.

### template<typename T> vector<T> matmul_interchanged(const vector<T>& A, const vector<T>& B, int m, int n, int p)
Purpose: Loop order (i, k, j) to load A[i,k] once and update C[i,*] sequentially.  
Algorithm:  
```cpp
for i in 0..m-1
  for k in 0..n-1
    a = A[i*n + k]
    for j in 0..p-1
      C[i*p + j] += a * B[k*p + j]
```
Benefits: Better write locality for C; good access pattern for row-major B.

### template<typename T> vector<T> transposeB(const vector<T>& B, int n, int p)
Purpose: Compute transpose Bt of B with shape p × n. Bt[j*n + k] = B[k*p + j].  
Complexity: Time O(n * p).  
Use-case: Converting column access of B into sequential reads.  

### template<typename T> vector<T> matmul_transpose(const vector<T>& A, const vector<T>& B, int m, int n, int p)
Purpose: Transpose B and compute multiply using Bt, measuring transpose + multiply.  
Notes: The code separately times transpose and multiply; GFLOPS for multiply-only is reported separately if desired.  

### template<typename T> vector<T> matmul_blocked_naive(const vector<T>& A, const vector<T>& B, int m, int n, int p, int bs = 128)
Purpose: Blocked (tiled) matrix multiply. Iterates tiles of size bs × bs and performs inner tile updates with unrolled j loop by 4.  
Algorithm (tile loops):  
```cpp
for ii in 0..m-1 step bs
  for kk in 0..n-1 step bs
    for jj in 0..p-1 step bs
      process tile C[ii:i_max, jj:j_max] += A[ii:i_max, kk:k_max] * B[kk:k_max, jj:j_max]
```
Notes:  
- Handles edge partial tiles (via min).  
- Inner j loop unrolled by 4 (with scalar remainder).  
- bs tuned via autotuner in main().

### template<typename T> vector<T> matmul_blocked_interchange(const vector<T>& A, const vector<T>& B, int m, int n, int p, int bs)
Purpose: Blocked implementation with loop interchange inside tiles and unrolled j loop. Uses raw pointers Ap, Bp, Cp for lower indexing overhead in hot loops.  
Notes: Similar to blocked_naive but structured to favor (i, k, j) within each tile.  

### template<typename T> bool almost_equal(const vector<T>& X, const vector<T>& Y, double reltol = 1e-8, double abseps = 1e-12)
Purpose: Numerically compare two vectors with relative + absolute tolerance. Print diagnostic on first failing index.  
Behavior:  
- Treats pairs of NaN as equal.  
- Fails if |a - b| > max(abseps, reltol * max(|a|, |b|)).  
Use: Validate optimized versions against naive reference.  

### double sec_between(HRClock::time_point a, HRClock::time_point b)
Purpose: Return elapsed time in seconds between two chrono time points.  

## 5. main() — behavior & flow
1. using Type = double; — changeable to float for speed.  
2. Dimensions in the file: m = 2048, n = 2048, p = 2048 (edit for your tests).  
3. Create matrices A (seed=1) and B (seed=2) via createMatrix.  
4. Compute reference C_naive.  
5. Run transposeB + multiply, matmul_interchanged, blocked variants.  
6. Autotune block sizes for blocked kernels using candidate list {16,32,48,64,96,128}.  
7. Validate each optimized result with almost_equal(C_naive, C_variant); exit on mismatch.  
8. Print elapsed time and GFLOPS for each variant.

## 6. How to run / reproduce
Compile:  
```bash
g++ -O3 -march=native -funroll-loops -std=c++17 matmul_single_core.cpp -o matmul_single_core
```
Run:  
```bash
./matmul_single_core
```
Notes:  
- Program currently uses hard-coded sizes and autotune flags. For flexible use, add CLI parsing (see Suggested Improvements).  
- Large sizes (2048³) will be time-consuming and memory-heavy.

## 7. Correctness & verification
- The program uses almost_equal to compare each optimized result to the naive reference.  
- If a mismatch occurs, the program prints an error and exits with non-zero status.  
- Use double for more conservative verification. For float, set reltol to 1e-6 or 1e-5.

## 8. Performance notes & recommendations
- Blocking is critical for cache reuse; bs must be tuned for your CPU cache sizes (L1/L2/L3).  
- Loop ordering affects cache and write locality. (i, k, j) or blocked (i,k,j) are often faster than (i,j,k).  
- Transpose B reduces strided accesses when B is accessed by columns.  
- Use compiler flags: -O3 -march=native -funroll-loops for best autovectorization/unrolling.  
- Consider explicit SIMD (AVX/AVX2/AVX-512) intrinsics or compiler pragmas for further speedups.  
- To scale beyond a single core, add OpenMP/TBB parallelism.

## 9. Limitations & known issues
- Single-threaded only — no OpenMP/CUDA.  
- Autotuner runs full multiplies for each candidate block size (expensive for large sizes).  
- Hard-coded parameters in main() reduce usability — add CLI parsing.  
- Performance depends on compiler auto-vectorization; no intrinsics used.  
- Memory footprint is O(m*n + n*p + m*p) — large matrices may exceed RAM.

## 10. Suggested improvements & TODOs
- Add CLI argument parsing (e.g., cxxopts, getopt) to set m,n,p, dtype, --autotune, --bs, --verify.  
- Add warm-up runs and repeat measurements; average timings.  
- Expand autotuner to sample smaller sub-problems (to speed tuning).  
- Implement SIMD kernels or call BLAS for comparison (OpenBLAS, MKL).  
- Add OpenMP multithreaded variants and compare scaling.  
- Output benchmark results as CSV/JSON for plotting.  
- Add unit tests for small matrices and edge cases.  
- Provide a Makefile and an example run_benchmarks.sh.

## 11. Build / Test checklist (recommended before pushing)
- Compile with -O3 -march=native for benchmarking.  
- Test small sizes (m=n=p=16/64) to verify correctness quickly.  
- Run autotune on small problem to check functionality.  
- Run full benchmark on target machine and record results.

## 12. License & contact
- MIT License
