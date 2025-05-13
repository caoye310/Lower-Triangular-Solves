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

void ca_ata(const CSRMatrix &L, const std::vector<std::vector<int>> &levels, const int s, std::vector<std::vector<int>> &ca_levels) {

    // int N = L.nnz;
    int i = 0; // outer level id
    int num_task = 0;
    // iterate levels
    for (auto &level : levels) {
        if (level.size() >= s) { // if the original level size >= s, the level is an ATA
            // Ensure outer level exists

            if (ca_levels.size() <= i)
                ca_levels.resize(i + 1);


            // Clear and resize inner level to 1
            // ca_levels[i].clear();
            // ca_levels[i].resize(1);

            // Directly assign the level to the first inner vector
            ca_levels[i] = level;

            i++;  // move to next outer level      
        }
        else {
            // ca_levels[i].resize(j+1);
            for (auto &task : level) {

                // Ensure outer level exists
                if (ca_levels.size() <= i)
                    ca_levels.resize(i + 1);

                // Append task
                ca_levels[i].push_back(task);

                // If current inner level exceeds size limit, move to next outer level
                if (ca_levels[i].size() == s) {
                    i++;
                }
            }
        }      
    }
}

void ata_inner_aggregation(const CSRMatrix &L, const std::vector<std::vector<int>> &ca_levels, const int s, std::vector<std::vector<std::vector<int>>> &ata) {

    int N = L.nrows;

    int i = 0; // outer level id
    int j = 0; // inner level id
    
    for (auto &level : ca_levels) {
        if (level.size() > s) {
            // if (j != 0) i++;

            if (ata.size() <= i)
                ata.resize(i + 1);

            ata[i].clear();
            ata[i].resize(1);

            ata[i][0] = level;

            i++;
            j = 0;
            
        }
        else {
            int numOfTasks = level.size();
            std::map<int, int> level_array; // store have solved tasks
            std::map<int, bool> has_Dependency;
            std::vector<int> dependency;

            int max_level = 0;

            for (auto &row : level) {

                has_Dependency[row] = false;
                
                // collect level 0

                for (int k = L.rowptr[row]; k < L.rowptr[row + 1]; k++) {
                    int target = L.colidx[k]; // less than row
                    auto cnt = std::count(level.begin(), level.end(), target);
                    bool exists = (std::find(level.begin(), level.end(), target) != level.end());

                    if (target != row && cnt > 0) {
                        has_Dependency[row] = true;
                        break;
                    }

                    if (!has_Dependency[row]) {
                        dependency.push_back(row);
                        level_array[row] = 0;
                    }

                }
            }

            // assigen the rest user task

            for (auto &row : level) {
                
                if(!has_Dependency[row])
                    continue;
                
                else {
                    for (int k = L.rowptr[row]; k < L.rowptr[row + 1]; k++) {
                        int target = L.colidx[k]; // less than row
                        // auto cnt = std::count(level.begin(), level.end(), target); // check if the target in current ata
                        auto cnt2 = std::count(dependency.begin(), dependency.end(), target); // check if the traget has been solved

                        if (target != row && cnt2 > 0) {
                            int old_level = level_array[row];
                            level_array[row] = std::max(old_level, level_array[target] + 1);
                            max_level = std::max(max_level, level_array[row]);
                        }

                    }
                    dependency.push_back(row);
                }   

            }

            if (ata.size() <= i)
            ata.resize(i + 1);

            ata[i].resize(max_level + 1);
            for (auto &pair : level_array) {
                int inner_level = pair.second;
                int task = pair.first;
                ata[i][inner_level].push_back(task);
            }
            i++;
        }
            
    }  
}
