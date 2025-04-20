#pragma once
#include <vector>

void load_mtx_to_csr(const char* fname,
                     std::vector<int> &rowptr,
                     std::vector<int> &colidx,
                     std::vector<double> &val);
