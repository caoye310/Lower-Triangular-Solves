
#include "la_opt.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

void compute_la_schedule(const CSRMatrix &L,
                         int task_granularity,
                         LAScheduleOPT &S)
{
    const int N = L.nrows;

    /* ------------------------------------------------------------------
       1.  连续行 → A‑task
       ------------------------------------------------------------------ */
    std::vector<int> row2group(N, -1);
    std::vector< std::vector<int> > group2rows;
    for (int r = 0; r < N; ++r) {
        if (row2group[r] != -1) continue;
        group2rows.emplace_back();
        int gid = static_cast<int>(group2rows.size()) - 1;
        for (int a = 0; a < task_granularity && r + a < N; ++a) {
            int row = r + a;
            row2group[row] = gid;
            group2rows.back().push_back(row);
        }
    }
    const int G = static_cast<int>(group2rows.size());

    /* ------------------------------------------------------------------
       2.  A‑task DAG
       ------------------------------------------------------------------ */
    std::vector< std::vector<int> > adj(G);
    std::vector<int> indeg(G, 0);
    for (int g = 0; g < G; ++g) {
        for (int row : group2rows[g]) {
            for (int p = L.rowptr[row]; p < L.rowptr[row + 1]; ++p) {
                int dep = L.colidx[p];
                if (dep < row) {
                    int h = row2group[dep];
                    if (h != g) adj[h].push_back(g);
                }
            }
        }
    }
    for (int h = 0; h < G; ++h) {
        auto &v = adj[h];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (int g : v) ++indeg[g];
    }

    /* ------------------------------------------------------------------
       3.  Kahn 拓扑 → coarse levels  +  per‑task SET rows
       ------------------------------------------------------------------ */
    std::vector<int> Q;
    for (int g = 0; g < G; ++g) if (!indeg[g]) Q.push_back(g);

    while (!Q.empty()) {
        S.schedule.emplace_back();
        S.u_deps.emplace_back();
        S.row2succ.emplace_back();
        S.row2succ_ptr.emplace_back();

        std::vector<int> next;
        for (int g : Q) {
            /* ---------------- A‑task local ---------------- */
            std::vector<int> rows = group2rows[g];      // copy, will reorder
            const int R = static_cast<int>(rows.size());

            /* ---- 3.1  levelise inside task to get depth ---- */
            std::unordered_map<int,int> row2idx0;
            for (int i = 0; i < R; ++i) row2idx0[ rows[i] ] = i;

            std::vector<int> indegL(R, 0);
            std::vector< std::vector<int> > succL(R);
            for (int i = 0; i < R; ++i) {
                int row = rows[i];
                std::unordered_set<int> seen;            // 去重前驱
                for (int p = L.rowptr[row]; p < L.rowptr[row + 1]; ++p) {
                    int dep = L.colidx[p];
                    if (dep < row) {
                        auto it = row2idx0.find(dep);
                        if (it != row2idx0.end() && seen.insert(dep).second) {
                            ++indegL[i];
                            succL[it->second].push_back(i);
                        }
                    }
                }
            }
            std::vector<int> depth(R, 0), q0;
            for (int i = 0; i < R; ++i) if (!indegL[i]) q0.push_back(i);
            while (!q0.empty()) {
                int v = q0.back(); q0.pop_back();
                for (int s : succL[v]) {
                    depth[s] = std::max(depth[s], depth[v] + 1);
                    if (--indegL[s] == 0) q0.push_back(s);
                }
            }

            /* ---- 3.2  stable_sort rows by depth (SET) ---- */
            // Build row → depth map first to avoid stale indices
            std::unordered_map<int,int> row2depth;
            for (int i = 0; i < R; ++i) row2depth[ rows[i] ] = depth[i];

            std::stable_sort(rows.begin(), rows.end(),
                [&](int a, int b){ return row2depth[a] < row2depth[b]; });

            /* ---- 3.3  rebuild local maps according new order ---- */
            std::unordered_map<int,int> row2idx;
            for (int i = 0; i < R; ++i) row2idx[ rows[i] ] = i;

            std::vector<int>   u_deps(R, 0);
            std::vector< std::vector<int> > succ(R);
            for (int i = 0; i < R; ++i) {
                int row = rows[i];
                std::unordered_set<int> seen;
                for (int p = L.rowptr[row]; p < L.rowptr[row + 1]; ++p) {
                    int dep = L.colidx[p];
                    if (dep < row) {
                        auto it = row2idx.find(dep);
                        if (it != row2idx.end() && seen.insert(dep).second) {
                            ++u_deps[i];
                            succ[it->second].push_back(i);
                        }
                    }
                }
            }
            std::vector<int> ptr(R + 1, 0), idx; idx.reserve(64 * R);
            for (int i = 0; i < R; ++i) {
                ptr[i + 1] = ptr[i] + succ[i].size();
                idx.insert(idx.end(), succ[i].begin(), succ[i].end());
            }

            /* ---- push to LASchedule ---- */
            S.schedule.back().push_back(std::move(rows));
            S.u_deps.back().push_back(std::move(u_deps));
            S.row2succ.back().push_back(std::move(idx));
            S.row2succ_ptr.back().push_back(std::move(ptr));

            for (int s : adj[g]) if (--indeg[s] == 0) next.push_back(s);
        }
        Q.swap(next);
    }
}
