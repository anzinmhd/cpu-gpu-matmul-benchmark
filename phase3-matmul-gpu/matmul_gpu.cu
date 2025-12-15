%%writefile matmul_gpu.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> 

using namespace std;
using namespace nvcuda;

// USER CONFIGURATION
 
// Matrix Size (Must be a multiple of 16)
#define N 8192        
#define BLOCK_SIZE 32 
#define DATA_TYPE float 

// 1. DATA GENERATION & HELPERS

// Templated Generator: Works for int, float, and double
template<typename T>
vector<T> createMatrix(int rows, int cols, unsigned int seed) {
    vector<T> M((size_t)rows * cols);
    srand(seed); 
    for (size_t i = 0; i < M.size(); i++) {
        // Generate random 0.0 to 1.0
        double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        // Scale to -5.0 to 5.0 and cast to target type
        M[i] = static_cast<T>((r * 10.0) - 5.0);
    }
    return M;
}

// Helper: Converts ANY type (int/float/double) to Half Precision
// This allows us to run the Tensor Core benchmark regardless of what DATA_TYPE you chose.
template <typename T>
vector<half> convertToHalf(const vector<T>& input) {
    vector<half> output(input.size());
    for(size_t i=0; i<input.size(); i++) {
        // Cast input to float first, then convert to half
        output[i] = __float2half(static_cast<float>(input[i]));
    }
    return output;
}

// 2. STANDARD KERNELS (Templated for DATA_TYPE)

// ALGORITHM 1: NAIVE (Global Memory)
template <typename T>
__global__ void matmul_naive(const T *A, const T *B, T *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        T sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ALGORITHM 2: TILED (Shared Memory)
template <typename T>
__global__ void matmul_tiled(const T *A, const T *B, T *C, int n) {
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = 0;

    for (int t = 0; t < (n / BLOCK_SIZE); ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * n + (t * BLOCK_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}


// 3. TENSOR CORE KERNEL (Specialized for FP16)

// ALGORITHM 3: WMMA (Hardware Acceleration)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void matmul_tensor_core(const half *A, const half *B, float *C, int n) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int globalWarpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int globalWarpN = (blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = 0; i < n; i += WMMA_K) {
        int aRow = globalWarpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = globalWarpN * WMMA_N;
        if (aRow < n && aCol < n && bRow < n && bCol < n) {
            wmma::load_matrix_sync(a_frag, A + aRow * n + aCol, n);
            wmma::load_matrix_sync(b_frag, B + bCol * n + bRow, n);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    int cRow = globalWarpM * WMMA_M;
    int cCol = globalWarpN * WMMA_N;
    if (cRow < n && cCol < n) {
        wmma::store_matrix_sync(C + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
    }
}


// MAIN RUNNER
template <typename T>
void run_complete_benchmark() {
    size_t bytes = N * N * sizeof(T);
    
    // --- 1. SETUP HOST DATA ---
    cout << "-------------------------Matrix Size: " << N << "x" << N <<"-------------------------"<< endl ;
    vector<T> h_A_vec = createMatrix<T>(N, N, 42);
    vector<T> h_B_vec = createMatrix<T>(N, N, 43);
    T *h_A = h_A_vec.data();
    T *h_B = h_B_vec.data();

    // --- 2. SETUP DEVICE DATA (Standard) ---
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cout << "\n-----------------------------------------------------------" << endl;
    cout << " BENCHMARK 1 & 2: STANDARD CORES (Config: " << BLOCK_SIZE << "x" << BLOCK_SIZE << " Tiles)" << endl;
    cout << "-----------------------------------------------------------" << endl;

    // --- RUN ALGORITHM 1: NAIVE ---
    cudaEventRecord(start);
    matmul_naive<T><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
    printf(">> 1. Naive (Global Mem):   %8.2f GFLOPS | Time: %.4f s\n", (2.0*N*N*N)/(ms/1000.0 * 1e9), ms/1000.0);

    // --- RUN ALGORITHM 2: TILED ---
    cudaEventRecord(start);
    matmul_tiled<T><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
    printf(">> 2. Tiled (Shared Mem):   %8.2f GFLOPS | Time: %.4f s\n", (2.0*N*N*N)/(ms/1000.0 * 1e9), ms/1000.0);

    // --- 3. SETUP TENSOR CORE DATA (Reference) ---
    cout << "\n-----------------------------------------------------------" << endl;
    cout << " BENCHMARK 3: TENSOR CORES (Reference Speed)" << endl;
    cout << "-----------------------------------------------------------" << endl;
    
    // We convert your chosen DATA_TYPE to Half Precision so Tensor Cores can process it
    vector<half> h_A_half = convertToHalf(h_A_vec);
    vector<half> h_B_half = convertToHalf(h_B_vec);
    
    half *d_A_h, *d_B_h; float *d_C_f;
    cudaMalloc(&d_A_h, N * N * sizeof(half));
    cudaMalloc(&d_B_h, N * N * sizeof(half));
    cudaMalloc(&d_C_f, N * N * sizeof(float));
    cudaMemcpy(d_A_h, h_A_half.data(), N*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_h, h_B_half.data(), N*N*sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid_tc(N / 16, N / 16);
    dim3 block_tc(32, 1, 1);
    
    // --- RUN ALGORITHM 3: TENSOR CORES ---
    cudaEventRecord(start);
    matmul_tensor_core<<<grid_tc, block_tc>>>(d_A_h, d_B_h, d_C_f, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
    printf(">> 3. Tensor Cores (WMMA):  %8.2f GFLOPS | Time: %.4f s\n", (2.0*N*N*N)/(ms/1000.0 * 1e9), ms/1000.0);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_h); cudaFree(d_B_h); cudaFree(d_C_f);
}

int main() {
    // This macro ensures we only instantiate the code for the type you selected
    run_complete_benchmark<DATA_TYPE>();
    return 0;
}



//compiler settings
//nvcc -arch=sm_75 matmul_gpu.cu -o matmul_gpu


//run
//./matmul_gpu

//in colab use "!" at starting for compiling and running the code.