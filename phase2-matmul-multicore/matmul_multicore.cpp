// Phase 2: OpenMP Parallel Matrix Multiplication
// Methods: Naive | Transpose-B | Loop-Interchanged |
//          Blocked-Naive | Blocked-Interchanged
//
// Compile:
// g++ -O3 -march=native -funroll-loops -std=c++17 -fopenmp matmul_phase2.cpp -o matmul_phase2

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

using namespace std;
using HRClock = chrono::high_resolution_clock;

// ------------------ Matrix Generation ------------------
template<typename T>
vector<T> createMatrix(int rows, int cols, unsigned int seed) {
    vector<T> M((size_t)rows * cols);
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(-5.0, 5.0);
    for (auto &x : M) x = static_cast<T>(dist(rng));
    return M;
}

// ------------------ Naive Parallel ------------------
template<typename T>
vector<T> matmul_naive_omp(const vector<T>& A, const vector<T>& B,
                           int m, int n, int p) {

    vector<T> C((size_t)m * p, 0);

    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            T sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*p + j];
            C[i*p + j] = sum;
        }

    return C;
}

// ------------------ Loop Interchanged Parallel ------------------
template<typename T>
vector<T> matmul_interchanged_omp(const vector<T>& A, const vector<T>& B,
                                  int m, int n, int p) {

    vector<T> C((size_t)m * p, 0);

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        const T* arow = &A[i*n];
        T* crow = &C[i*p];
        for (int k = 0; k < n; k++) {
            T a = arow[k];
            const T* brow = &B[k*p];
            for (int j = 0; j < p; j++)
                crow[j] += a * brow[j];
        }
    }

    return C;
}

// ------------------ Transpose-B Parallel ------------------
template<typename T>
vector<T> transposeB(const vector<T>& B, int n, int p) {
    vector<T> Bt((size_t)p * n);

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < p; j++)
            Bt[j*n + k] = B[k*p + j];

    return Bt;
}

template<typename T>
vector<T> matmul_transpose_omp(const vector<T>& A, const vector<T>& B,
                               int m, int n, int p) {

    vector<T> Bt = transposeB(B, n, p);
    vector<T> C((size_t)m * p, 0);

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        const T* arow = &A[i*n];
        T* crow = &C[i*p];
        for (int j = 0; j < p; j++) {
            const T* btrow = &Bt[j*n];
            T sum = 0;
            for (int k = 0; k < n; k++)
                sum += arow[k] * btrow[k];
            crow[j] = sum;
        }
    }

    return C;
}

// ------------------ Blocked (both modes) ------------------
template<typename T>
vector<T> matmul_blocked_omp(const vector<T>& A, const vector<T>& B,
                             int m, int n, int p, int bs, bool use_interchanged)
{
    vector<T> C((size_t)m * p, 0);

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < m; ii += bs)
        for (int jj = 0; jj < p; jj += bs)
            for (int kk = 0; kk < n; kk += bs) {

                int i_end = min(ii + bs, m);
                int j_end = min(jj + bs, p);
                int k_end = min(kk + bs, n);

                // ---- Blocked-Naive ----
                if (!use_interchanged)
                {
                    for (int i = ii; i < i_end; i++) {
                        const T* arow = &A[i*n];
                        T* crow = &C[i*p];
                        for (int k = kk; k < k_end; k++) {
                            T a = arow[k];
                            const T* brow = &B[k*p];
                            for (int j = jj; j < j_end; j++)
                                crow[j] += a * brow[j];
                        }
                    }
                }

                // ---- Blocked-Interchanged ----
                else
                {
                    for (int i = ii; i < i_end; i++) {
                        const T* arow = &A[i*n];
                        T* crow = &C[i*p];
                        for (int k = kk; k < k_end; k++) {
                            T a = arow[k];
                            const T* brow = &B[k*p];
                            for (int j = jj; j < j_end; j++)
                                crow[j] += a * brow[j];
                        }
                    }
                }
            }

    return C;
}

// ------------------ Correctness Check ------------------
template<typename T>
bool almost_equal(const vector<T>& X, const vector<T>& Y) {
    for (size_t i = 0; i < X.size(); i++)
        if (fabs((double)X[i] - (double)Y[i]) > 1e-6)
            return false;
    return true;
}

double sec_between(HRClock::time_point a, HRClock::time_point b) {
    return chrono::duration<double>(b - a).count();
}

// ------------------ MAIN ------------------
int main() {
    using Type = double;

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);

    int m = 1024, n = 1024, p = 1024;
    cout << "\nPhase 2: OpenMP Parallel Benchmarking\n";
    cout << "Matrix: " << m << " × " << n << " × " << p << endl;
    cout << "Threads: " << threads << "\n\n";

    vector<Type> A = createMatrix<Type>(m, n, 1);
    vector<Type> B = createMatrix<Type>(n, p, 2);

    auto run = [&](auto func, string name) {
        auto t0 = HRClock::now();
        auto C = func(A, B, m, n, p);
        auto t1 = HRClock::now();
        double s = sec_between(t0, t1);
        double gflops = (2.0 * m * n * p) / (s * 1e9);
        cout << name << ": " << s << " s  => " << gflops << " GFLOPS\n";
        return C;
    };

    // Run all algorithms
    auto C1 = run(matmul_naive_omp<Type>, "Naive (OMP)");
    auto C2 = run(matmul_interchanged_omp<Type>, "Loop-Interchanged (OMP)");
    auto C3 = run(matmul_transpose_omp<Type>, "Transpose-B (OMP)");

    auto C4 = run([&](auto&a,auto&b,int x,int y,int z){
        return matmul_blocked_omp(a,b,x,y,z,128,false);
    }, "Blocked-Naive (OMP)");

    auto C5 = run([&](auto&a,auto&b,int x,int y,int z){
        return matmul_blocked_omp(a,b,x,y,z,128,true);
    }, "Blocked-Interchanged (OMP)");

    cout << "\nCorrectness Check vs Naive:\n";
    cout << "Loop-Interchanged:      " << (almost_equal(C1, C2) ? "OK" : "Mismatch!") << endl;
    cout << "Transpose-B:            " << (almost_equal(C1, C3) ? "OK" : "Mismatch!") << endl;
    cout << "Blocked-Naive:          " << (almost_equal(C1, C4) ? "OK" : "Mismatch!") << endl;
    cout << "Blocked-Interchanged:   " << (almost_equal(C1, C5) ? "OK" : "Mismatch!") << endl;

    return 0;
}
