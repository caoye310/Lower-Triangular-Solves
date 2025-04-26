#include "levels.h"
#include <omp.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <vector>
#include <iostream>
#include <omp.h>
#include <algorithm>

void compute_levels(const CSRMatrix &L, std::vector<std::vector<int>> &levels) {
    int N = L.nrows;
    std::vector<int> level_array(N, N + 1); // Store the level of each row

    // Step 1: Find level 0 rows. (rows with all off-diags =0 )
    for (int row = 0; row < N; row++) {
        bool has_dependency = false;
        
        for (int k = L.rowptr[row]; k < L.rowptr[row + 1]; k++) {
            if (L.colidx[k] < row) { // Check lower triangular dependencies
                has_dependency = true;
                break;
            }
        }
        
        if (!has_dependency) {
            level_array[row] = 0; // No dependencies, assign Level 0
        }
    }

    // Step 2: Assign levels based on level_array
    int max_level = 0;
    while (true) {
        bool added_levels = false;
        for (int row = 0; row < N; row++) {
            if (level_array[row] == N + 1) { // Row hasn't been assigned a level yet
                bool can_add = true;
                int row_level = 0;

                for (int k = L.rowptr[row]; k < L.rowptr[row + 1]; k++) {
                    int dep = L.colidx[k];

                    if (dep < row) { // Check dependencies
                        if (level_array[dep] == N + 1) { // Unresolved dependency
                            can_add = false;
                            break;
                        }
                        row_level = std::max(row_level, level_array[dep] + 1);
                    }
                }

                if (can_add) {
                    level_array[row] = row_level;
                    max_level = std::max(max_level, row_level);
                    added_levels = true;
                }
            }
        }

        if (!added_levels) break;
    }

    // Step 3: Construct levels[] using level_array
    levels.resize(max_level + 1); // Resize to accommodate all levels
    for (int row = 0; row < N; row++) {
        levels[level_array[row]].push_back(row);
    }
}
