%%writefile matmul_single_core.cpp
// single-core matrix multiplication (naive | transpose-B | interchanged | blocked)
// matmul_single_core_opt.cpp
// Optimized single-core matrix multiplication (naive | transpose-B | interchanged | blocked + small autotuner)
// Compile: g++ -O3 -march=native -funroll-loops -std=c++17 matmul_single_core_opt.cpp -o matmul_single_core_opt

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace std;

//size_t => unsigned number(indices,size,etc)

//time measurement
using HRClock = chrono::high_resolution_clock;

template<typename T>
vector<T> createMatrix(int rows, int cols, unsigned int seed) {
    vector<T> M(static_cast<size_t>(rows) * cols);
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(-5.0, 5.0);
    for (size_t i = 0; i < M.size(); ++i) M[i] = static_cast<T>(dist(rng));
    return M;
}

template<typename T>
vector<T> matmul_naive(const vector<T>& A, const vector<T>& B, int m, int n, int p) {
    vector<T> C(static_cast<size_t>(m) * p, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            T sum = 0;
            const T* arow = &A[(size_t)i * n]; //first element(i th row)
            for (int k = 0; k < n; ++k) {
                sum += arow[k] * B[(size_t)k * p + j];
            }
            C[(size_t)i * p + j] = sum;
        }
    }
    return C;
}

template<typename T>
vector<T> matmul_interchanged(const vector<T>& A, const vector<T>& B, int m, int n, int p) {
    vector<T> C(static_cast<size_t>(m) * p, 0); //initializes all elements to 0
    for (int i = 0; i < m; ++i) {
        T* crow = &C[(size_t)i * p];
        const T* arow = &A[(size_t)i * n];
        for (int k = 0; k < n; ++k) {
            T a = arow[k];
            const T* brow = &B[(size_t)k * p];
            for (int j = 0; j < p; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
    return C;
}

template<typename T>
vector<T> transposeB(const vector<T>& B, int n, int p) {
    vector<T> Bt(static_cast<size_t>(p) * n);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < p; ++j)
            Bt[(size_t)j * n + k] = B[(size_t)k * p + j];
    return Bt;
}


//why transposeB => accesing columns(B) becomes sequential(faster accessing)
template<typename T>
vector<T> matmul_transpose(const vector<T>& A, const vector<T>& B, int m, int n, int p) {
    // This measures both transpose + multiply
    vector<T> Bt = transposeB(B, n, p);
    vector<T> C(static_cast<size_t>(m) * p, 0);
    for (int i = 0; i < m; ++i) {
        const T* arow = &A[(size_t)i * n];
        T* crow = &C[(size_t)i * p];
        for (int j = 0; j < p; ++j) {
            const T* btrow = &Bt[(size_t)j * n];
            T sum = 0;
            for (int k = 0; k < n; ++k)
                sum += arow[k] * btrow[k];
            crow[j] = sum;
        }
    }
    return C;
}

// Optimized blocked kernel: pointer arithmetic, restrict hints, small j-unroll
template<typename T>
vector<T> matmul_blocked_naive(const vector<T>& A, const vector<T>& B, int m, int n, int p, int bs = 128) {
    vector<T> C(static_cast<size_t>(m) * p, 0);
    for (int ii = 0; ii < m; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < p; jj += bs) {
              //Handle edges (partial blocks)
                int i_max = min(ii + bs, m);
                int k_max = min(kk + bs, n);
                int j_max = min(jj + bs, p);
                //Inner loops: compute the tile
                for (int i = ii; i < i_max; ++i) {
                    T* crow = &C[(size_t)i * p + jj];
                    const T* arow = &A[(size_t)i * n];
                    for (int k = kk; k < k_max; ++k) {
                        T a = arow[k];
                        const T* brow = &B[(size_t)k * p + jj];
                        int J = j_max - jj;
                        int j = 0;
                        // Unroll by 4:This reduces loop overhead
                        for (; j + 3 < J; j += 4) {
                            crow[j]     += a * brow[j];
                            crow[j + 1] += a * brow[j + 1];
                            crow[j + 2] += a * brow[j + 2];
                            crow[j + 3] += a * brow[j + 3];
                        }
                        for (; j < J; ++j) crow[j] += a * brow[j];
                    }
                }
            }
    return C;
}

// C is assumed preallocated (size m*p). It accumulates into C (C must be zeroed before call).
// Returns the resulting matrix C (m x p) using blocked tiles and loop-interchanged order inside each tile.
template<typename T> //uses loop_interchange for each submatrix (optimization under optimazation)
vector<T> matmul_blocked_interchange(const vector<T>& A, const vector<T>& B, int m, int n, int p, int bs) {
    // Allocate result and zero it
    vector<T> C(static_cast<size_t>(m) * p, (T)0);

    // Raw pointers for faster indexing inside hot loop
    const T* Ap = A.data();
    const T* Bp = B.data();
    T* Cp = C.data();

    // iterate over tiles
    for (int ii = 0; ii < m; ii += bs) {
        int i_max = min(ii + bs, m);
        for (int kk = 0; kk < n; kk += bs) {
            int k_max = min(kk + bs, n);
            for (int jj = 0; jj < p; jj += bs) {
                int j_max = min(jj + bs, p);

                // compute C[ii..i_max-1, jj..j_max-1] += A[ii..i_max-1, kk..k_max-1] * B[kk..k_max-1, jj..j_max-1]
                for (int i = ii; i < i_max; ++i) {
                    T* crow = Cp + (size_t)i * p + jj;       // &C[i][jj]
                    const T* arow = Ap + (size_t)i * n;      // &A[i][0]
                    for (int k = kk; k < k_max; ++k) {
                        T a = arow[k];                        // A[i][k] (load once)
                        const T* brow = Bp + (size_t)k * p + jj; // &B[k][jj]
                        int J = j_max - jj;
                        int j = 0;
                        // small unroll by 4 (reduces loop overhead)
                        for (; j + 3 < J; j += 4) {
                            crow[j]     += a * brow[j];
                            crow[j + 1] += a * brow[j + 1];
                            crow[j + 2] += a * brow[j + 2];
                            crow[j + 3] += a * brow[j + 3];
                        }
                        for (; j < J; ++j) crow[j] += a * brow[j];
                    }
                }
            }
        }
    }

    return C;
}

template<typename T>
bool almost_equal(const vector<T>& X, const vector<T>& Y, double reltol = 1e-8, double abseps = 1e-12) {
    if (X.size() != Y.size()) {
        cerr << "Error: Vector sizes differ! X size: " << X.size() << ", Y size: " << Y.size() << endl;
        return false;
    }
    for (size_t i = 0; i < X.size(); ++i) {
        double a = static_cast<double>(X[i]);
        double b = static_cast<double>(Y[i]);
        if (isnan(a) || isnan(b)) {
            if (isnan(a) != isnan(b)) {
                cerr << "Error at index " << i << ": One value is NaN, the other is not. X[" << i << "]: " << a << ", Y[" << i << "]: " << b << endl;
                return false;
            }
            continue; // both NaN => treat equal
        }
        double diff = fabs(a - b);
        double thresh = max(abseps, reltol * max(fabs(a), fabs(b)));
        if (diff > thresh) {
            cerr << "Error at index " << i << ": values differ beyond tolerance. X[" << i << "]: " << a << ", Y[" << i << "]: " << b << ", Diff: " << diff << ", Thresh: " << thresh << endl;
            return false;
        }
    }
    return true;
}

double sec_between(HRClock::time_point a, HRClock::time_point b) {
    return chrono::duration<double>(b - a).count();
}


int main() {
    using Type = double; // Changed from float to double

    // Matrix dimensions (change to test other shapes)
    int m = 2048, n = 2048, p = 2048;
    cout << "Benchmarking Matrix Multiplication (M = " << m << ", N = " << n << ", P = " << p << ")\n";
    cout << "-----------------------Benchmarking Started-----------------------\n";
    // Create matrices
    vector<Type> A = createMatrix<Type>(m, n, 1);
    vector<Type> B = createMatrix<Type>(n, p, 2);

    vector<Type> C_naive, C_transpose, C_loop, C_blocked_naive, C_blocked_interchange;

    double flops = 2.0 * (double)m * n * p;
    double gflops;

    // Naive
    {
        auto t0 = HRClock::now();
        C_naive = matmul_naive(A, B, m, n, p);
        auto t1 = HRClock::now();
        double s = sec_between(t0, t1);
        gflops = flops / (s * 1e9);
        cout << "Naive Matmul: " << s << " s"
             << " => " << gflops << " GFLOPS\n";
    }

    // Transpose: time the transpose separate from multiply
    {
        auto t0 = HRClock::now();
        auto Bt = transposeB(B, n, p);
        auto t1 = HRClock::now();
        // multiply using Bt
        auto t2 = HRClock::now();
        vector<Type> C(static_cast<size_t>(m) * p, 0);
        for (int i = 0; i < m; ++i) {
            const Type* arow = &A[(size_t)i * n];
            Type* crow = &C[(size_t)i * p];
            for (int j = 0; j < p; ++j) {
                const Type* btrow = &Bt[(size_t)j * n];
                Type sum = 0;
                for (int k = 0; k < n; ++k) sum += arow[k] * btrow[k];
                crow[j] = sum;
            }
        }
        auto t3 = HRClock::now();
        cout << "Transpose (transpose time + multiply time): " << sec_between(t0, t3) << " s"
             << " (transpose: " << sec_between(t0, t1) << " s, multiply: " << sec_between(t2, t3) << " s)";
        cout << " => " << (flops / (sec_between(t2, t3) * 1e9)) << " GFLOPS (multiply only)\n";
        C_transpose.swap(C);
    }

    // Loop-interchanged
    {
        auto t0 = HRClock::now();
        C_loop = matmul_interchanged(A, B, m, n, p);
        auto t1 = HRClock::now();
        double s = sec_between(t0, t1);
        cout << "Loop-Interchange Matmul: " << s << " s"
             << " => " << (flops / (s * 1e9)) << " GFLOPS\n";
        if (!almost_equal(C_naive, C_loop)) {
            cerr << "Error: Naive and Loop-Interchange results do not match!\n";
            return 1;
        }
    }

    // Blocked: try default block size and optionally an autotune
    int best_bs = 128;

    //autotune for blocked_naive
    const bool do_autotune_naive = true; // set true to autotune (slower but finds best bs)
    if (do_autotune_naive) {
        cout << "Autotuning block size for matmul_blocked_naive...\n";
        vector<int> cands = {16, 32, 48, 64, 96, 128};
        double best_time = 1e308;
        for (int bs : cands) {
            auto t0 = HRClock::now();
            auto Ctry = matmul_blocked_naive(A, B, m, n, p, bs);
            auto t1 = HRClock::now();
            double s = sec_between(t0, t1);
            cout << "  bs=" << bs << " -> " << s << " s\n";
            if (s < best_time) { best_time = s; best_bs = bs; }
            if (!almost_equal(C_naive, Ctry)) { cerr << "Autotune (naive): result mismatch at bs=" << bs << "\n"; return 1; }
        }
        cout << "Autotune (naive) picked bs = " << best_bs << "\n";
    }

    //Tiling Method (each submatrix in naive multiplication)
    {
        auto t0 = HRClock::now();
        C_blocked_naive = matmul_blocked_naive(A, B, m, n, p, best_bs);
        auto t1 = HRClock::now();
        double s = sec_between(t0, t1);
        cout << "Blocked Matmul_Naive (bs=" << best_bs << "): " << s << " s"
             << " => " << (flops / (s * 1e9)) << " GFLOPS\n";
        if (!almost_equal(C_naive, C_blocked_naive)) {
            cerr << "Error: Naive and Blocked_Naive results do not match!\n";
            return 1;
        }
    }

    //autotuning for block_size for Blocked_interchanged
    const bool do_autotune_interchange = true; // set true to autotune (slower but finds best bs)
    if (do_autotune_interchange) {
        cout << "Autotuning block size for matmul_blocked_interchange...\n";
        vector<int> cands = {16, 32, 48, 64, 96, 128};
        double best_time = 1e308;
        for (int bs : cands) {
            auto t0 = HRClock::now();
            auto Ctry = matmul_blocked_interchange(A, B, m, n, p, bs);
            auto t1 = HRClock::now();
            double s = sec_between(t0, t1);
            cout << "  bs=" << bs << " -> " << s << " s\n";
            if (s < best_time) { best_time = s; best_bs = bs; }
            if (!almost_equal(C_naive, Ctry)) { cerr << "Autotune (interchanged): result mismatch at bs=" << bs << "\n"; return 1; }
        }
        cout << "Autotune (interchanged) picked bs = " << best_bs << "\n";
    }

    //Tiling Method (with loop_interchanged_method for submatrices)
    {
        auto t0 = HRClock::now();
        C_blocked_interchange = matmul_blocked_interchange<Type>(A, B, m, n, p, best_bs); // Changed <double> to <Type>
        auto t1 = HRClock::now();
        double s = sec_between(t0, t1);
        cout << "Blocked Matmul_Interchanged (bs=" << best_bs << "): " << s << " s"
             << " => " << (flops / (s * 1e9)) << " GFLOPS\n";
        if (!almost_equal(C_naive, C_blocked_interchange)) {
            cerr << "Error: Naive and Blocked_Interchanged results do not match!\n";
            return 1;
        }

    }

    cout << "All matrix multiplication versions produced correct results (within tolerance)." << endl;
    return 0;
}


//compiler settings
//g++ -O3 -march=native -funroll-loops -std=c++17 matmul_single_core.cpp -o matmul_single_core

//code running
//./matmul_single_core

//include ! at starting for compiler settings and running code in colab