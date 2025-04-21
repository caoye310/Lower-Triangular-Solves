#pragma once
#include <cuda_runtime.h>

template<int TILE_ROWS, int TILE_NZ>
__global__ void dag_lts_kernel(
        const int    *__restrict__ rowptr,
        const int    *__restrict__ colidx,
        const double *__restrict__ val,
        const double *__restrict__ b,
        double       *__restrict__ y,          // in & out
        const int    *__restrict__ rows_lvl,   // which level 
        int rows_lvl_len)
{
    const int lrow = threadIdx.y;                       // tile 
    const int gidx = blockIdx.x * TILE_ROWS + lrow;     // level 
    if (gidx >= rows_lvl_len) return;

    const int row  = rows_lvl[gidx];

    const int nz_start = rowptr[row];
    const int nz_end   = rowptr[row+1];
    const int nnz_row  = nz_end - nz_start;             // ≤ TILE_NZ

    /* ------------- share memory ------------- */
    __shared__ int    sh_col [TILE_ROWS][TILE_NZ];
    __shared__ double sh_val [TILE_ROWS][TILE_NZ];

    for (int k = threadIdx.x; k < nnz_row; k += blockDim.x) {
        sh_col[lrow][k] = colidx[nz_start + k];
        sh_val[lrow][k] = val   [nz_start + k];
    }
    __syncthreads();

    /* ------------- accumulate ------------------- */
    double sum  = 0.0;
    for (int k = threadIdx.x; k < nnz_row; k += blockDim.x) {
        int    j = sh_col[lrow][k];
        double a = sh_val[lrow][k];
        if (j < row) sum += a * y[j];
    }
    /* warp reduction */
    for (int off = 16; off; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);

    /* ------------- write back ------------------------- */
    if (threadIdx.x == 0) {
        double diag = val[nz_end - 1];
        y[row] = (b[row] - sum) / diag;
    }
}
