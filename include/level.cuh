#pragma once

template<int TILE_ROWS, int TILE_NZ>
__global__ void lts_levelset_tile(
    const int* __restrict__ rowptr,
    const int* __restrict__ col,
    const double* __restrict__ val,
    const double* __restrict__ y,
    const double* __restrict__ b,
    double* __restrict__ out,
    const int* __restrict__ rows_this_lvl,
    int n_rows_lvl)
{
    __shared__ int    sh_col[TILE_NZ + 1];
    __shared__ double sh_val[TILE_NZ + 1];

    int local_row = threadIdx.y;
    int g_row_id  = rows_this_lvl[blockIdx.x*TILE_ROWS + local_row];
    if (g_row_id >= n_rows_lvl) return;

    int nz_start = rowptr[g_row_id];
    int nz_end   = rowptr[g_row_id + 1];
    for (int k = nz_start + threadIdx.x; k < nz_end; k += blockDim.x) {
        int loc = k - nz_start;
        sh_col[loc] = col[k];
        sh_val[loc] = val[k];
    }
    __syncthreads();

    double sum = 0.0;
    for (int k = threadIdx.x; k < nz_end - nz_start; k += blockDim.x)
        sum += sh_val[k] * y[sh_col[k]];
    for (int off = 16; off; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);

    if (threadIdx.x == 0)
        out[g_row_id] = b[g_row_id] - sum;
}
