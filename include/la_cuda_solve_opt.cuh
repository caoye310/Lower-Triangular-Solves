#include "csr_matrix.h"
#include "la_opt.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <cmath>

struct TaskDependency {
    int* u_deps;      // N                   (row‑local predecessor counter)
    int* succ_ptr;    // N+1                 (CSR prefix)
    int* succ_idx;    // M                   (flat successor list, local row‑id)
};

// -----------------------------------------------------------------------------
// Helper: warp‑reduce sum of doubles (no explicit warp‑sync needed on Ampere+)
// -----------------------------------------------------------------------------
__inline__ __device__ double warp_reduce_sum(double v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// -----------------------------------------------------------------------------
// Kernel –  one warp per row, push style, SET ordering guarantees inter‑warp deps
// -----------------------------------------------------------------------------
__global__ void set_scheduled_kernel(
    const int* __restrict__ rowptr,
    const int* __restrict__ colidx,
    const double* __restrict__ values,
    const double* __restrict__ b,
    double* __restrict__ x,
    const int* __restrict__ rows,   // local‑row id -> global CSR row id
    int num_rows,
    TaskDependency deps)
{
    constexpr unsigned WARP = 32;
    const unsigned lane = threadIdx.x&31;               // 0..31
    const unsigned row_id = blockIdx.x;                   // one block == one warp == one row
    if (row_id >= (unsigned)num_rows) return;
    const int row = rows[row_id];                         // global row

    // ---------------------------------------------------------------------
    // lane0 busy‑wait on its row's predecessor counter – other lanes idle
    // Because SET puts predecessor rows into *earlier* warps (row_id smaller),
    // there is no cyclic dependency across warps, hence safe.
    // ---------------------------------------------------------------------
    nvtxRangePushA("Wait for Dependencies");
    if (lane == 0)
        while (atomicAdd(&deps.u_deps[row_id], 0) > 0) { /* spin */ }

     __syncwarp();                                         // make sure everyone sees row ready
    nvtxRangePop();

    // ---------------------------------------------------------------------
    // Compute ∑ a_ij * x_j  (j < i)   with warp‑parallel reduction
    // ---------------------------------------------------------------------
    nvtxRangePushA("Compute Sum");
    double partial = 0.0;
    for (int p = rowptr[row] + lane; p < rowptr[row + 1]; p += WARP) {
        int    col = colidx[p];
        double a   = values[p];
        if (col < row) {
            double x_val = x[col];
            __threadfence();
            partial += a * x_val;
        }
    }
    double sum = warp_reduce_sum(partial);
    nvtxRangePop();

    // ---------------------------------------------------------------------
    // lane0 finds diagonal & writes solution, then pushes to successors
    // ---------------------------------------------------------------------
    nvtxRangePushA("Update Solution and Notify Successors");
    if (lane == 0) {
        double diag = 0.0;
        for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
            if (colidx[p] == row) { diag = values[p]; break; }
        }
        x[row] = (b[row] - sum) / diag;
       
        __threadfence();                                   // ensure x visible before unlock

        int s0 = deps.succ_ptr[row_id];
        int s1 = deps.succ_ptr[row_id + 1];
        for (int p = s0; p < s1; ++p) {
            int succ = deps.succ_idx[p];                   // local id in same A‑task
            atomicSub(&deps.u_deps[succ], 1);
        }
    }
    nvtxRangePop();
}

// ============================================================================
//  Host helper –  Scheme‑B launcher per A‑task
// ============================================================================
#include <thrust/device_vector.h>
#include <thrust/sort.h>

template<int TILE_ROWS, int TILE_NZ>
inline void parallel_dag_lower_triangular_solve_cuda_la_opt(
    const CSRMatrix& L,
    std::vector<double>& y,
    const std::vector<double>& b,
    const LAScheduleOPT& S)
{
    nvtxRangePushA("Memory Allocation and Initial Copy");
    // --- 1. copy static CSR & RHS to device ---
    int    *d_rptr, *d_cidx;   double *d_val;
    double *d_b,   *d_y;
    cudaMalloc(&d_rptr, sizeof(int)   * L.rowptr.size());
    cudaMalloc(&d_cidx, sizeof(int)   * L.colidx.size());
    cudaMalloc(&d_val,  sizeof(double)* L.data.size());
    cudaMalloc(&d_b,    sizeof(double)* b.size());
    cudaMalloc(&d_y,    sizeof(double)* y.size());
    cudaMemcpy(d_rptr, L.rowptr.data(), sizeof(int)   * L.rowptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cidx, L.colidx.data(), sizeof(int)   * L.colidx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,  L.data.data(),   sizeof(double)* L.data.size(),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,    b.data(),        sizeof(double)* b.size(),        cudaMemcpyHostToDevice);
    cudaMemset(d_y,    0,               sizeof(double)* y.size());
    nvtxRangePop();

    // --- 2. iterate over coarse levels (outer Topo levels) ---
    for (size_t lvl = 0; lvl < S.schedule.size(); ++lvl) {
        nvtxRangePushA(("Level " + std::to_string(lvl)).c_str());
        const auto &lvlTasks = S.schedule[lvl];
        std::vector<cudaStream_t> streams(lvlTasks.size());
        for (auto &s : streams) cudaStreamCreate(&s);

        for (size_t t = 0; t < lvlTasks.size(); ++t) {
            nvtxRangePushA(("Stream " + std::to_string(t)).c_str());
            const auto &rows_h = lvlTasks[t];                 // rows in this A‑task
            const int R = static_cast<int>(rows_h.size());
            if (!R) {
                nvtxRangePop();
                continue;
            }
            
            nvtxRangePushA("Task Memory Allocation");
            // ------ 2.1  Allocate & copy A‑task private tables ------
            int *d_rows, *d_udep, *d_sptr, *d_sidx;

            cudaMallocAsync(&d_rows,  sizeof(int)   * R,                 streams[t]);
            cudaMallocAsync(&d_udep,  sizeof(int)   * R,                 streams[t]);
            const auto &sptr_h  = S.row2succ_ptr[lvl][t];                 // size R+1
            const auto &sidx_h  = S.row2succ[lvl][t];
            const auto& rows    = S.schedule[lvl][t];
            cudaMallocAsync(&d_sptr,  sizeof(int)   * sptr_h.size(),      streams[t]);
            cudaMallocAsync(&d_sidx,  sizeof(int)   * sidx_h.size(),      streams[t]);
            nvtxRangePop();

            nvtxRangePushA("Task Memory Copy");
            cudaMemcpyAsync(d_rows,  rows_h.data(), sizeof(int)*R,           cudaMemcpyHostToDevice, streams[t]);
            cudaMemcpyAsync(d_udep,  S.u_deps[lvl][t].data(), sizeof(int)*R, cudaMemcpyHostToDevice, streams[t]);
            cudaMemcpyAsync(d_sptr,  sptr_h.data(), sizeof(int)*sptr_h.size(), cudaMemcpyHostToDevice, streams[t]);
            cudaMemcpyAsync(d_sidx,  sidx_h.data(), sizeof(int)*sidx_h.size(), cudaMemcpyHostToDevice, streams[t]);
            nvtxRangePop();

            // ------ 2.2  Build TaskDependency value ------
            TaskDependency dep{};
            dep.u_deps   = d_udep;
            dep.succ_ptr = d_sptr;
            dep.succ_idx = d_sidx;

            // ------ 2.3  Launch kernel : 1 block/row (32 threads) ------
            const dim3 block(32);
            const dim3 grid (R);                         // 1 warp per row
            set_scheduled_kernel<<<grid, block, 0, streams[t]>>>(
                d_rptr, d_cidx, d_val, d_b, d_y,
                d_rows, R, dep);
            cudaDeviceSynchronize();

            nvtxRangePushA("Task Memory Free");
            // free A‑task private buffers after stream done
            cudaFreeAsync(d_rows,  streams[t]);
            cudaFreeAsync(d_udep,  streams[t]);
            cudaFreeAsync(d_sptr,  streams[t]);
            cudaFreeAsync(d_sidx,  streams[t]);
            nvtxRangePop();
            
            nvtxRangePop(); // Stream
        }

        nvtxRangePushA("Level Synchronization");
        for (auto &s : streams) { 
            cudaStreamSynchronize(s); 
            cudaStreamDestroy(s); 
        }
        nvtxRangePop();
        
        nvtxRangePop(); // Level
    }

    nvtxRangePushA("Final Memory Operations");
    // --- 3. copy result back ---
    cudaMemcpy(y.data(), d_y, sizeof(double)*y.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_rptr); cudaFree(d_cidx); cudaFree(d_val);
    cudaFree(d_b);   cudaFree(d_y);
    nvtxRangePop();
}