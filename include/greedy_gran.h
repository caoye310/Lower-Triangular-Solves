#pragma once

#include "csr_matrix.h"

// 计算每个任务的内存需求（以字节为单位）
size_t compute_task_memory_requirement(const CSRMatrix &L, int start_row, int end_row);

// 计算最优任务粒度
int compute_optimal_granularity(const CSRMatrix &L, size_t gpu_memory_limit); 