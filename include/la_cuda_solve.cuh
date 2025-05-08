#include "csr_matrix.h"
#include <vector>
#include "la.h"

template<int TILE_ROWS,int TILE_NZ>
void parallel_dag_lower_triangular_solve_cuda_la(const CSRMatrix& L,
                                                 std::vector<double>& y,
                                                 const std::vector<double>& b,
                                                 const LASchedule& S)
{
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

    /* ===== 双级循环 ===== */
    for (const auto& coarseLvl : S)                    // ⓐ coarse level
    {
        std::vector<cudaStream_t> streams(coarseLvl.size());
        for (size_t i = 0; i < streams.size(); ++i)
            cudaStreamCreate(&streams[i]);

        // 同一 coarse-level 内的 A-tasks 放不同 stream，可硬件并行
        for (size_t t = 0; t < coarseLvl.size(); ++t)  // ⓑ A-task
        {
            const auto& inner = coarseLvl[t];
            for (const auto& rows_vec : inner)         // ⓒ inner-level
            {
                if (rows_vec.empty()) continue;

                /* ---- 复制 rows ---- */
                int *d_rows;  const int R = rows_vec.size();
                cudaMallocAsync(&d_rows, R*sizeof(int), streams[t]);
                cudaMemcpyAsync(d_rows, rows_vec.data(),
                                 R*sizeof(int),
                                 cudaMemcpyHostToDevice, streams[t]);

                dim3 block(32, TILE_ROWS);
                dim3 grid ((R + TILE_ROWS - 1) / TILE_ROWS);

                dag_lts_kernel<TILE_ROWS,TILE_NZ><<<grid, block, 0, streams[t]>>>(
                    d_rowptr, d_col, d_val, d_b, d_y, d_rows, R);

                cudaFreeAsync(d_rows, streams[t]);
                /* —— 如果你愿意，也可以把同一 A-task 的所有 inner-level
                      拼成一个 rows_vec，在 kernel 内做 barrier-free 批量解，
                      这里只保持与你原逻辑一致 —— */
            }
        }
        // coarse-level barrier：必须等本层全部 streams 完毕
        for (auto& s : streams) { cudaStreamSynchronize(s); cudaStreamDestroy(s); }
    }

    /* ---- 拷贝结果 / 释放 ---- */
}