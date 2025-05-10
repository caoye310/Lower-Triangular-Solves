#pragma once

#include "csr_matrix.h"

// Calculate memory requirement for each task (in bytes)
size_t compute_task_memory_requirement(const CSRMatrix &L, int start_row, int end_row);

// Compute optimal task granularity
int compute_optimal_granularity(const CSRMatrix &L, size_t gpu_memory_limit); 