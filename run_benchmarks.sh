#!/bin/bash

# MatMul Benchmark Suite - Automated Runner

echo "=========================================================="
echo " Starting MatMul Benchmark Suite Compilation and Execution "
echo "=========================================================="

# 1. Single-Core CPU
echo -e "\n[1/3] Compiling Phase 1: Single-Core CPU..."
cd phase1-matmul-singlecore
g++ -O3 -std=c++17 matmul-singlecore.cpp -o benchmark_singlecore
if [ $? -eq 0 ]; then
    echo "Running Phase 1..."
    ./benchmark_singlecore
else
    echo "Compilation failed for Phase 1."
fi
cd ..

# 2. Multi-Core CPU
echo -e "\n[2/3] Compiling Phase 2: Multi-Core CPU (OpenMP)..."
cd phase2-matmul-multicore
g++ -O3 -fopenmp -std=c++17 matmul_multicore.cpp -o benchmark_multicore
if [ $? -eq 0 ]; then
    echo "Running Phase 2..."
    ./benchmark_multicore
else
    echo "Compilation failed for Phase 2."
fi
cd ..

# 3. GPU CUDA
echo -e "\n[3/3] Compiling Phase 3: GPU CUDA (Naive, Tiled, Tensor Cores)..."
cd phase3-matmul-gpu
# Check if nvcc is available
if command -v nvcc &> /dev/null; then
    nvcc -arch=sm_75 matmul_gpu.cu -o benchmark_gpu
    if [ $? -eq 0 ]; then
        echo "Running Phase 3..."
        ./benchmark_gpu
    else
        echo "Compilation failed for Phase 3."
    fi
else
    echo "nvcc (CUDA compiler) not found. Skipping Phase 3."
fi
cd ..

echo -e "\n=========================================================="
echo " Benchmarking Complete."
echo "=========================================================="
