#include <vector>
#include <atomic>
#include <iostream>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

#include "csr_matrix.h"
#include "levels.h"
#include "ilu.h"


void ca_aggregation(const CSRMatrix &L, const std::vector<std::vector<int>> &levels, const int s, std::vector<std::vector<std::vector<int>>> &ca_levels) {

    // int N = L.nnz;
    int i = 0; // outer level id
    int j = 0; // inner level id
    int num_task = 0;
    // iterate levels
    for (auto &level : levels) {
        if (level.size() >= s) { // if the original level size >= s, the level is an ATA
            // Ensure outer level exists
            if (j != 0) i++;

            
            if (ca_levels.size() <= i)
                ca_levels.resize(i + 1);


            // Clear and resize inner level to 1
            ca_levels[i].clear();
            ca_levels[i].resize(1);

            // Directly assign the level to the first inner vector
            ca_levels[i][0] = level;

            i++;  // move to next outer level
            j = 0;
            num_task = 0; // reset inner level id       
        }
        else {
            // ca_levels[i].resize(j+1);
            for (auto &task : level) {

                // Ensure outer level exists
                if (ca_levels.size() <= i)
                    ca_levels.resize(i + 1);

                // Ensure inner level exists
                if (ca_levels[i].size() <= j)
                    ca_levels[i].resize(j + 1);

                // Append task
                ca_levels[i][j].push_back(task);
                num_task++;

                // If current inner level exceeds size limit, move to next outer level
                if (num_task == s) {
                    i++;
                    j = 0;
                    num_task = 0;
                }
            }
            j++; // once into new level_set, inner level id + 1
        }      
    }
}



