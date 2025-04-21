#ifndef LTS_PARALLEL_DAG_H
#define LTS_PARALLEL_DAG_H

#include "csr_matrix.h"
#include <vector>
#include <omp.h>
void compute_levels(const CSRMatrix &L, std::vector<std::vector<int>> &levels);
#endif  // LOWER_SOLVE_H