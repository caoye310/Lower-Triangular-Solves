#include "greedy_gran.h"
#include <algorithm>
#include <cmath>

// Tesla T4 GPU memory limit (14GB)
const size_t TESLA_T4_MEMORY_LIMIT = 14ULL * 1024 * 1024 * 1024;

// Calculate memory requirement for each task (in bytes)
size_t compute_task_memory_requirement(const CSRMatrix &L, int start_row, int end_row) {
    size_t memory = 0;
    
    // Count non-zero elements in lower triangular part
    size_t nnz = 0;
    for (int row = start_row; row < end_row; ++row) {
        for (int p = L.rowptr[row]; p < L.rowptr[row + 1]; ++p) {
            if (L.colidx[p] <= row) {  // Only count lower triangular part
                nnz++;
            }
        }
    }
    
    // Calculate required memory:
    // 1. Input vector y (end_row - start_row) * sizeof(double)
    // 2. Output vector b (end_row - start_row) * sizeof(double)
    // 3. Lower triangular matrix data (nnz * sizeof(double))
    // 4. Column indices (nnz * sizeof(int))
    // 5. Row pointers ((end_row - start_row + 1) * sizeof(int))
    memory = (end_row - start_row) * sizeof(double) * 2 +  // y and b vectors
             nnz * sizeof(double) +                        // matrix values
             nnz * sizeof(int) +                          // column indices
             (end_row - start_row + 1) * sizeof(int);     // row pointers
    
    return memory;
}

// Compute optimal task granularity
int compute_optimal_granularity(const CSRMatrix &L, size_t gpu_memory_limit) {
    int nrows = L.nrows;
    
    // Calculate average non-zero elements per row
    double avg_nnz_per_row = 0.0;
    for (int row = 0; row < nrows; ++row) {
        int row_nnz = 0;
        for (int p = L.rowptr[row]; p < L.rowptr[row + 1]; ++p) {
            if (L.colidx[p] <= row) {  // Only count lower triangular part
                row_nnz++;
            }
        }
        avg_nnz_per_row += row_nnz;
    }
    avg_nnz_per_row /= nrows;
    
    // Estimate appropriate granularity range based on matrix characteristics
    int min_granularity = 32;  // Minimum one warp
    int max_granularity = std::min(nrows, 1024);  // Maximum 1024 rows or total rows
    
    // Adjust range based on average non-zero elements
    if (avg_nnz_per_row > 100) {  // Dense rows
        min_granularity = 64;
        max_granularity = std::min(nrows, 512);
    } else if (avg_nnz_per_row < 10) {  // Sparse rows
        min_granularity = 128;
        max_granularity = std::min(nrows, 2048);
    }
    
    int optimal_granularity = min_granularity;
    int current_granularity = min_granularity;
    
    // Binary search for optimal granularity
    int left = min_granularity;
    int right = max_granularity;
    
    while (left <= right) {
        current_granularity = (left + right) / 2;
        
        // Check if current granularity satisfies memory limit
        bool valid = true;
        for (int start = 0; start < nrows; start += current_granularity) {
            int end = std::min(start + current_granularity, nrows);
            size_t memory = compute_task_memory_requirement(L, start, end);
            
            if (memory > gpu_memory_limit) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            // Current granularity is valid, try larger
            optimal_granularity = current_granularity;
            left = current_granularity + 1;
        } else {
            // Current granularity is invalid, try smaller
            right = current_granularity - 1;
        }
    }
    
    // Ensure granularity is multiple of 32 (one warp)
    optimal_granularity = (optimal_granularity + 31) & ~31;
    
    return optimal_granularity;
} 