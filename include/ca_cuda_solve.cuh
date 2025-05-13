#include "csr_matrix.h"
#include "cuda_runtime.h"
#include "dag_lts_kernel.cuh"
#include <vector>

template<int TILE_ROWS, int TILE_NZ>
void parallel_ca_lower_triangular_solve_cuda(const CSRMatrix &L,
                                              std::vector<double> &y,
                                              const std::vector<double> &b,
                                              const std::vector<std::vector<std::vector<int>>> &ca_levels)
{
    const int N = L.nrows;

    int    *d_rowptr, *d_col;
    double *d_val, *d_b, *d_y;
    cudaMalloc(&d_rowptr, sizeof(int)    * L.rowptr.size());
    cudaMalloc(&d_col   , sizeof(int)    * L.colidx.size());
    cudaMalloc(&d_val   , sizeof(double) * L.data.size());
    cudaMalloc(&d_b     , sizeof(double) * b.size());
    cudaMalloc(&d_y     , sizeof(double) * y.size());

    cudaMemcpy(d_rowptr, L.rowptr.data(), sizeof(int)    * L.rowptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col   , L.colidx.data(), sizeof(int)    * L.colidx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val   , L.data.data()  , sizeof(double) * L.data.size()  , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b     , b.data()       , sizeof(double) * b.size(),        cudaMemcpyHostToDevice);
    cudaMemset (d_y     , 0,              sizeof(double) * y.size());

    /* ---- for each level call kernel ---- */
    for (auto &outer_level : ca_levels) {
        for (auto &inter_level : outer_level) {
            const int rows = inter_level.size();
            if (rows == 0)
                continue;
            
            int *d_rows;
            cudaMalloc(&d_rows, sizeof(int) * rows);
            cudaMemcpy(d_rows, inter_level.data(), sizeof(int)*rows, cudaMemcpyHostToDevice);

            dim3 block(32, TILE_ROWS);
            dim3 grid ((rows + TILE_ROWS - 1) / TILE_ROWS);

            dag_lts_kernel<TILE_ROWS, TILE_NZ><<<grid, block>>>(
                d_rowptr, d_col, d_val, d_b, d_y,
                d_rows, rows);

            cudaFree(d_rows);

        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    /* ---- copy result ---- */
    cudaMemcpy(y.data(), d_y, sizeof(double)*N, cudaMemcpyDeviceToHost);

    cudaFree(d_rowptr); cudaFree(d_col); cudaFree(d_val);
    cudaFree(d_b);      cudaFree(d_y);
}