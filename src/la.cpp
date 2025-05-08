#include "la.h"

void compute_la_schedule(const CSRMatrix &L,
                         int task_granularity,
                         LASchedule &S)
{
    /* -------- 1. 先用之前的方法得到 G 个 A-task 及其 inner-levels -------- */
    std::vector<std::vector<std::vector<int>>> group2inner;
    std::vector<int> row2group(L.nrows, -1);
    {
        /* 复制自 compute_la_levels —— 省略，得到 group2inner & row2group */
        /* group2inner[g][l] == 行号 */
    }
    const int G = group2inner.size();

    /* -------- 2. 构建 A-task 间依赖图 (邻接+入度) -------- */
    std::vector<std::vector<int>> adj(G); // g -> list of succ
    std::vector<int> indeg(G, 0);
    for (int g = 0; g < G; ++g)
        for (const auto &lvl : group2inner[g])
            for (int row : lvl)
            {
                for (int k = L.rowptr[row]; k < L.rowptr[row + 1]; ++k)
                {
                    int dep = L.colidx[k];
                    if (dep < row)
                    {
                        int h = row2group[dep];
                        if (h != g)
                        {
                            adj[h].push_back(g);
                            ++indeg[g];
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
        S.emplace_back(); // 新 coarse level
        std::vector<int> next;
        for (int g : q)
        {
            S.back().push_back(group2inner[g]); // 插入该 A-task
            for (int succ : adj[g])
                if (--indeg[succ] == 0)
                    next.push_back(succ);
        }
        q.swap(next);
    }
}
