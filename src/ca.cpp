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


// void ca_aggregation_levels(const std::vector<std::vector<int>> &levels, const int s, std::vector<std::vector<int>> &ca_levels) {
//     // int N = L.nnz;
//     // int num_levels = N / s + 1;
//     int i = 0;
//     // iterate levels
//     for (auto &level : levels) { 
//         if (level.size() >= s) { // if the original level size >= s, the level is an ATA
//             if (i != 0)
//                 i++;
//             ca_levels[i] = level;
//             i++;    
//         }
//         else {
//             for (auto &task : level) {
//                 ca_levels[i].push_back(task);
//                 if (ca_levels[i].size() > s) { // once size > s, strat a new ca_level
//                     i++;
//                     ca_levels[i].push_back(task);
//                 }
//             }
//         }
        
//     }
// }

void ca_aggregation(const std::vector<std::vector<int>> &levels, const int s, std::vector<std::vector<std::vector<int>>> &ca_levels) {

    // int N = L.nnz;
    // int num_levels = N / s + 1;
    int num_lv = levels.size();
    ca_levels.clear();
    ca_levels.resize(num_lv); 

    int i = 0; // outer level id
    int j = 0; // inner level id
    // iterate levels
    for (auto &level : levels) { 
        if (level.size() >= s) { // if the original level size >= s, the level is an ATA
            if (i != 0) {
                i++;
            }
            ca_levels[i].resize(1);
            ca_levels[i][0] = level;
            i++;       
        }
        else {
            ca_levels[i].resize(j+1);
            for (auto &task : level) {
                ca_levels[i][j].push_back(task);
                if (ca_levels[i].size() > s) { // once size > s, strat a new ca_level
                    i++;
                    j = 0;
                    ca_levels[i][j].push_back(task);
                }
            }
            j++; // once into new level_set, inner level id + 1
        }   
    }
}


