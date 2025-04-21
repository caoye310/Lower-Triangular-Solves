#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include "mmio.h"
#include "loadmm.h"
#include "csr_matrix.h"
#include "dag.h"
#include "dag_cuda_solve.cuh"

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

int main(int argc, char** argv) {
    // ---- Step 1: load matrix
    CSRMatrix A;
    load_mtx_to_csr(argv[1], A);
    CSRMatrix L;
    csr_lower(A, L);

    int N = L.nrows;

    // ---------- Step‑2 RHS ----------
    std::vector<double> b(N, 1.0);

    // ---- Step 3: serial execution for reference
    std::vector<double> x_ref;
    reference_solve_csr(L.rowptr, L.colidx, L.data, b, x_ref);

    // ---- Step 4: Levelset rows
    std::vector<std::vector<int>> levels;
    compute_levels(L, levels);      

    // ---- Step 6: launch kernel
    std::vector<double> y(L.nrows, 0.0);

    /* ---------- CUDA DAG Triangular Solve ---------- */
    constexpr int TILE_ROWS = 1;      
    constexpr int TILE_NZ   = 128; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    parallel_dag_lower_triangular_solve_cuda<TILE_ROWS, TILE_NZ>(
            L, y, b, levels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[CUDA DAG Solve Time] " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* ---------- （可选）用 CPU 做参考结果并比较 ---------- */
    std::vector<double> y_ref;
    reference_solve_csr(A.rowptr, A.colidx, A.data, b, y_ref);

    bool ok = compare_results(y_ref, y);
    std::cout << (ok ? "[PASS] GPU == CPU\n" : "[FAIL] mismatch!\n");
    return 0;
}
