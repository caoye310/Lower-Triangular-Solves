#include <vector>
#include <atomic>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_set>

#include "csr_matrix.h"
#include "levels.h"
#include "ilu.h"

using LASchedule = std::vector<                      // coarse levels
                    std::vector<                    // A-tasks in this level
                        std::vector<                // inner levels
                            std::vector<int>>>>;    // rows

void compute_la_schedule(const CSRMatrix &L,
    int task_granularity,
    LASchedule &S);