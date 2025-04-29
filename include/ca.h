#include <vector>
#include <atomic>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>

#include "csr_matrix.h"
#include "levels.h"
#include "ilu.h"

void ca_aggregation_levels(const std::vector<std::vector<int>> &levels, const int s, std::vector<std::vector<int>> &ca_levels);
void ca_aggregation(const std::vector<std::vector<int>> &levels, const int s, std::vector<std::vector<std::vector<int>>> &ca_levels);