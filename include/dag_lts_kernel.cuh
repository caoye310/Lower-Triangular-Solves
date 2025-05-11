// Improved version of dag_lts_kernel with warp-safe reduction and comments
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Warp-wide reduction helper (requires CUDA >= 9.0)
__inline__ __device__ double warpReduceSum(double val) {
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

template<int TILE_ROWS, int TILE_NZ>
__global__ void dag_lts_kernel(
        const int    *__restrict__ rowptr,
        const int    *__restrict__ colidx,
        const double *__restrict__ val,
        const double *__restrict__ b,
        double       *__restrict__ y,
        const int    *__restrict__ rows_lvl,
        int rows_lvl_len)
{
    const int lrow = threadIdx.y;                       // Local row index within the tile
    const int gidx = blockIdx.x * TILE_ROWS + lrow;     // Global index for rows_lvl
    if (gidx >= rows_lvl_len) return;

    const int row  = rows_lvl[gidx];
    const int nz_start = rowptr[row];
    const int nz_end   = rowptr[row + 1];
    const int nnz_row  = nz_end - nz_start;             // Number of nonzeros in this row


    __shared__ int    sh_col[TILE_ROWS][TILE_NZ];
    __shared__ double sh_val[TILE_ROWS][TILE_NZ];

    for (int k = threadIdx.x; k < nnz_row; k += blockDim.x) {
        sh_col[lrow][k] = colidx[nz_start + k];
        sh_val[lrow][k] = val[nz_start + k];
    }
    __syncthreads();

    double sum = 0.0;
    for (int k = threadIdx.x; k < nnz_row; k += blockDim.x) {
        int    j = sh_col[lrow][k];
        double a = sh_val[lrow][k];
        if (j < row) sum += a * y[j];
    }

    sum = warpReduceSum(sum);

    if (threadIdx.x == 0) {
        // Safer version to find diagonal if not guaranteed to be last
        double diag = 0.0;
        for (int k = 0; k < nnz_row; ++k) {
            if (sh_col[lrow][k] == row) {
                diag = sh_val[lrow][k];
                break;
            }
        }
        y[row] = (b[row] - sum) / diag;
    }
}