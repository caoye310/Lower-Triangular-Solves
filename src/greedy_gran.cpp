#include "greedy_gran.h"
#include <algorithm>

// Tesla T4 GPU memory limit (14GB)
const size_t TESLA_T4_MEMORY_LIMIT = 14ULL * 1024 * 1024 * 1024;

// 计算每个任务的内存需求（以字节为单位）
size_t compute_task_memory_requirement(const CSRMatrix &L, int start_row, int end_row) {
    size_t memory = 0;
    
    // 计算非零元素数量
    size_t nnz = 0;
    for (int row = start_row; row < end_row; ++row) {
        nnz += L.rowptr[row + 1] - L.rowptr[row];
    }
    
    // 计算所需内存：
    // 1. 输入向量 x (end_row - start_row) * sizeof(double)
    // 2. 输出向量 b (end_row - start_row) * sizeof(double)
    // 3. 矩阵数据 (nnz * sizeof(double))
    // 4. 列索引 (nnz * sizeof(int))
    // 5. 行指针 ((end_row - start_row + 1) * sizeof(int))
    memory = (end_row - start_row) * sizeof(double) * 2 +  // x 和 b 向量
             nnz * sizeof(double) +                        // 矩阵值
             nnz * sizeof(int) +                          // 列索引
             (end_row - start_row + 1) * sizeof(int);     // 行指针
    
    return memory;
}

// 计算最优任务粒度
int compute_optimal_granularity(const CSRMatrix &L, size_t gpu_memory_limit) {
    int nrows = L.nrows;
    int optimal_granularity = 1;  // 初始化为最小粒度
    int current_granularity = 1;
    
    // 二分查找最优粒度
    int left = 1;
    int right = nrows;
    
    while (left <= right) {
        current_granularity = (left + right) / 2;
        
        // 检查当前粒度是否满足内存限制
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
            // 当前粒度可行，尝试更大的粒度
            optimal_granularity = current_granularity;
            left = current_granularity + 1;
        } else {
            // 当前粒度不可行，尝试更小的粒度
            right = current_granularity - 1;
        }
    }
    
    return optimal_granularity;
} 