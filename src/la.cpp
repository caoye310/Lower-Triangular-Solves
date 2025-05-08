#include "la.h"

void compute_la_schedule(const CSRMatrix &L,
                         int task_granularity,
                         LASchedule &S)
{
    /* -------- 1. 先用之前的方法得到 G 个 A-task 及其 inner-levels -------- */
    std::vector<std::vector<std::vector<int>>> group2inner;
    std::vector<int> row2group(L.nrows, -1);
    {
        // 按照行的顺序划分A-tasks
        int current_group = 0;
        std::vector<int> current_task;
        
        for (int row = 0; row < L.nrows; ++row) {
            current_task.push_back(row);
            row2group[row] = current_group;
            
            if (current_task.size() >= task_granularity) {
                // 对当前A-task内的行进行level划分
                std::vector<std::vector<int>> inner_levels;
                std::vector<int> current_level;
                
                // 检查当前A-task内行之间的依赖关系
                for (int i = 0; i < current_task.size(); ++i) {
                    int r = current_task[i];
                    bool has_dependency = false;
                    
                    // 检查是否依赖当前level中的任何行
                    for (int j = 0; j < current_level.size(); ++j) {
                        int dep = current_level[j];
                        // 检查r是否依赖dep
                        for (int k = L.rowptr[r]; k < L.rowptr[r + 1]; ++k) {
                            if (L.colidx[k] == dep) {
                                has_dependency = true;
                                break;
                            }
                        }
                        if (has_dependency) break;
                    }
                    
                    if (has_dependency) {
                        // 如果有依赖，开始新的level
                        if (!current_level.empty()) {
                            inner_levels.push_back(current_level);
                            current_level.clear();
                        }
                    }
                    current_level.push_back(r);
                }
                
                // 添加最后一个level
                if (!current_level.empty()) {
                    inner_levels.push_back(current_level);
                }
                
                group2inner.push_back(inner_levels);
                current_task.clear();
                current_group++;
            }
        }
        
        // 处理最后一个不完整的A-task
        if (!current_task.empty()) {
            std::vector<std::vector<int>> inner_levels;
            std::vector<int> current_level;
            
            for (int i = 0; i < current_task.size(); ++i) {
                int r = current_task[i];
                bool has_dependency = false;
                
                for (int j = 0; j < current_level.size(); ++j) {
                    int dep = current_level[j];
                    for (int k = L.rowptr[r]; k < L.rowptr[r + 1]; ++k) {
                        if (L.colidx[k] == dep) {
                            has_dependency = true;
                            break;
                        }
                    }
                    if (has_dependency) break;
                }
                
                if (has_dependency) {
                    if (!current_level.empty()) {
                        inner_levels.push_back(current_level);
                        current_level.clear();
                    }
                }
                current_level.push_back(r);
            }
            
            if (!current_level.empty()) {
                inner_levels.push_back(current_level);
            }
            
            group2inner.push_back(inner_levels);
        }
    }
    const int G = group2inner.size();

    /* -------- 2. 构建 A-task 间依赖图 (邻接+入度) -------- */
    std::vector<std::vector<int>> adj(G); // g -> list of succ
    std::vector<int> indeg(G, 0);
    
    for (int g = 0; g < G; ++g) {
        for (const auto& level : group2inner[g]) {
            for (int row : level) {
                for (int k = L.rowptr[row]; k < L.rowptr[row + 1]; ++k) {
                    int dep = L.colidx[k];
                    if (dep < row) { 
                        int h = row2group[dep];
                        if (h != g) {
                            bool already_added = false;
                            for (int succ : adj[h]) {
                                if (succ == g) {
                                    already_added = true;
                                    break;
                                }
                            }
                            if (!already_added) {
                                adj[h].push_back(g);
                                ++indeg[g];
                            }
                        }
                    }
                }
            }
        }
    }

    /* -------- 3. 经典 Kahn topo-levelization 得 coarse levels -------- */
    std::vector<int> q;
    for (int g = 0; g < G; ++g)
        if (!indeg[g])
            q.push_back(g);

    while (!q.empty())
    {
        S.emplace_back(); 
        std::vector<int> next;
        for (int g : q)
        {
            S.back().push_back(group2inner[g]);
            for (int succ : adj[g])
                if (--indeg[succ] == 0)
                    next.push_back(succ);
        }
        q.swap(next);
    }
}
