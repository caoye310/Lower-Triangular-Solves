#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include "mmio.h"
#include "loadmm.h"
#include "csr_matrix.h"
#include "levels.h"
#include "dag_cuda_solve.cuh"
#include "ilu.h"

// =================== serial solver ===================
void reference_solve_csr(const std::vector<int>& rowptr,
    const std::vector<int>& colidx,
    const std::vector<double>& val,
    const std::vector<double>& b,
    std::vector<double>& x)
{
    int n = static_cast<int>(b.size());               

    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        int    diag_idx = -1;

        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
        {
            int j = colidx[p];

            if (j <  i)         sum += val[p] * x[j];   
            else if (j == i)    diag_idx = p;          
        }

        if (diag_idx < 0)
        throw std::runtime_error("missing diagonal at row " + std::to_string(i));

        double a_ii = val[diag_idx];
        if (a_ii == 0.0)
        throw std::runtime_error("zero pivot at row " + std::to_string(i));

        x[i] = (b[i] - sum) / a_ii;
    }
}



// =================== compare results ===================
bool compare_results(const std::vector<double>& ref,
                     const std::vector<double>& gpu,
                     double tol = 1e-6) {
    for (size_t i = 0; i < ref.size(); ++i) {
        //std::cout<<ref[i] - gpu[i]<<std::endl;
        if (std::abs((ref[i] - gpu[i])/ref[i]) > tol) {
            std::cerr << "Mismatch at i=" << i << ": ref=" << ref[i]
                     << ", gpu=" << gpu[i] << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // ---- Step 1: load matrix
    CSRMatrix A, L, U;
    load_mtx_to_csr(argv[1], A);
    cpu_spilu0(A, L, U);
    int N = L.nrows;
    // ---------- Step‑2 RHS ----------
    std::vector<double> b(N, 1.0);

    // ---- Step 3: serial execution for reference
    std::vector<double> y_ref(L.nrows, 0.0);
    reference_solve_csr(L.rowptr, L.colidx, L.data, b, y_ref);

    // ---- Step 4: Levelset rows
    std::vector<std::vector<int>> levels;
    compute_levels(L, levels);
    // for(int i=0;i<levels[0].size();i++){
    //     std::cout<<levels[0][i]<<" ";
    // }
    // std::cout<<std::endl;
    
    // ---- Step 6: launch kernel
    std::vector<double> y(L.nrows, 0.0);

    /* ---------- CUDA DAG Triangular Solve ---------- */
    // Each thread process 1 row
    constexpr int TILE_ROWS = 1; 
    // Max # non-zero value     
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

    // /* ---------- （可选）用 CPU 做参考结果并比较 ---------- */
    // std::vector<double> y_ref;
    // reference_solve_csr(A.rowptr, A.colidx, A.data, b, y_ref);

    bool ok = compare_results(y_ref, y);
    std::cout << (ok ? "[PASS] GPU == CPU\n" : "[FAIL] mismatch!\n");
    return 0;
}
