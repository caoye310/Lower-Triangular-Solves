#include "csr_matrix.h"
#include <vector>
#include "la.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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

    /* ===== Two-level loop ===== */
    for (size_t lvl = 0; lvl < S.size(); ++lvl) {
        const auto& coarseLvl = S[lvl];
        nvtxRangePushA(("Coarse Level " + std::to_string(lvl)).c_str());
        
        std::vector<cudaStream_t> streams(coarseLvl.size());
        for (size_t i = 0; i < streams.size(); ++i)
            cudaStreamCreate(&streams[i]);

        // A-tasks within the same coarse-level are placed in different streams for hardware parallelism
        for (size_t t = 0; t < coarseLvl.size(); ++t) {
            const auto& inner = coarseLvl[t];
            nvtxRangePushA(("A-Task " + std::to_string(t)).c_str());
            
            for (size_t i = 0; i < inner.size(); ++i) {
                const auto& rows_vec = inner[i];
                if (rows_vec.empty()) continue;

                nvtxRangePushA(("Inner Level " + std::to_string(i)).c_str());

                /* ---- Copy rows ---- */
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
                
                nvtxRangePop();
            }
            
            nvtxRangePop();
        }
        
        // Coarse-level barrier: must wait for all streams in this level to complete
        nvtxRangePushA("Stream Synchronization");
        for (auto& s : streams) { 
            cudaStreamSynchronize(s); 
            cudaStreamDestroy(s); 
        }
        nvtxRangePop();
        
        nvtxRangePop();
    }

    /* ---- Copy results / Free memory ---- */
    cudaMemcpy(y.data(), d_y, sizeof(double) * y.size(), cudaMemcpyDeviceToHost);
    
    cudaFree(d_rowptr);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_b);
    cudaFree(d_y);
}