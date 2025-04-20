#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "mmio.h"
#include "loadmm.h"
#include "level.cuh"

// kernel 在这假设是 template<int TILE_ROWS, int TILE_NZ> __global__ void lts_levelset_tile(...)

// =================== 串行 CPU 解法 ===================
void reference_solve_csr(const std::vector<int>& rowptr,
                         const std::vector<int>& colidx,
                         const std::vector<double>& val,
                         const std::vector<double>& b,
                         std::vector<double>& x) {
    int N = b.size();
    x.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) {
            int j = colidx[k];
            if (j < i)
                sum += val[k] * x[j];
        }
        x[i] = (b[i] - sum) / val[rowptr[i + 1] - 1];  // assume diag at end
    }
}

// =================== 校验函数 ===================
bool compare_results(const std::vector<double>& ref,
                     const std::vector<double>& gpu,
                     double tol = 1e-6) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - gpu[i]) > tol) {
            std::cerr << "Mismatch at i=" << i << ": ref=" << ref[i]
                      << ", gpu=" << gpu[i] << "\n";
            return false;
        }
    }
    return true;
}

// =================== Matrix Market 读取函数略 ===================
// 可用你之前写的 load_mtx_to_csr(...)

int main(int argc, char** argv) {
    std::vector<int> rowptr, colidx;
    std::vector<double> val;
    std::vector<double> b, x_ref, x_gpu;
    int N;

    // ---- Step 1: load matrix
    load_mtx_to_csr(argv[1], rowptr, colidx, val);
    N = rowptr.size() - 1;

    // ---- Step 2: 构造 RHS b
    b.resize(N, 1.0);

    // ---- Step 3: 串行 CPU 参考结果
    reference_solve_csr(rowptr, colidx, val, b, x_ref);

    // ---- Step 4: 准备 GPU 数据
    int* d_rowptr, *d_col;
    double* d_val, *d_b, *d_y;
    cudaMalloc(&d_rowptr, sizeof(int) * rowptr.size());
    cudaMalloc(&d_col, sizeof(int) * colidx.size());
    cudaMalloc(&d_val, sizeof(double) * val.size());
    cudaMalloc(&d_b, sizeof(double) * b.size());
    cudaMalloc(&d_y, sizeof(double) * b.size());

    cudaMemcpy(d_rowptr, rowptr.data(), sizeof(int) * rowptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, colidx.data(), sizeof(int) * colidx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val.data(), sizeof(double) * val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(double) * b.size());

    // ---- Step 5: Levelset rows —这里可选用你生成好的 levels
    std::vector<int> rows(N);
    for (int i = 0; i < N; ++i) rows[i] = i;
    int* d_rows;
    cudaMalloc(&d_rows, sizeof(int) * N);
    cudaMemcpy(d_rows, rows.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

    // ---- Step 6: launch kernel
    constexpr int TILE_ROWS = 1;
    constexpr int TILE_NZ   = 128;
    dim3 grid((N + TILE_ROWS - 1) / TILE_ROWS);
    dim3 block(32, TILE_ROWS);  // 32 threads x TILE_ROWS rows

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    lts_levelset_tile<TILE_ROWS, TILE_NZ><<<grid, block>>>(
        d_rowptr, d_col, d_val, d_y, d_b, d_y, d_rows, N);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[CUDA kernel time] " << ms << " ms\n";

    // ---- Step 7: copy result and compare
    x_gpu.resize(N);
    cudaMemcpy(x_gpu.data(), d_y, sizeof(double) * N, cudaMemcpyDeviceToHost);

    bool ok = compare_results(x_ref, x_gpu);
    std::cout << (ok ? "[PASS] Results match.\n" : "[FAIL] Results mismatch.\n");

    // ---- Clean up
    cudaFree(d_rowptr); cudaFree(d_col); cudaFree(d_val);
    cudaFree(d_b); cudaFree(d_y); cudaFree(d_rows);

    return 0;
}
