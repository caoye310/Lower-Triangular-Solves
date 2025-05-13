#ifndef LTS_PARALLEL_LA_H
#define LTS_PARALLEL_LA_H

#include <vector>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "csr_matrix.h"

struct LAScheduleOPT {
    // coarse levels -> A-tasks -> rows (已扁平化)
    std::vector<std::vector<std::vector<int>>> schedule;
    // 每个A-task中每个row的未完成前驱数
    std::vector<std::vector<std::vector<int>>> u_deps;
    // 每个A-task中每个row的后继列表
    std::vector<std::vector<std::vector<int>>> row2succ;
    // 每个A-task中每个row的后继指针
    std::vector<std::vector<std::vector<int>>> row2succ_ptr;
};

void compute_la_schedule(const CSRMatrix &L,
    int task_granularity,
    LAScheduleOPT &S);

#endif  // LTS_PARALLEL_LA_H