#include "ilu.h"

void cpu_spilu0(const CSRMatrix& A, CSRMatrix& L, CSRMatrix& U)
{
    const int n   = A.nrows;

    std::vector<double> val = A.data;      


    std::vector<int> diag_idx(n, -1);
    for (int i = 0; i < n; ++i)
        for (int p = A.rowptr[i]; p < A.rowptr[i+1]; ++p)
            if (A.colidx[p] == i) { diag_idx[i] = p; break; }

    for (int i = 0; i < n; ++i)
        if (diag_idx[i] == -1)
            val[diag_idx[i]] = EPS;  

    for (int i = 0; i < n; ++i)
    {
        for (int p = A.rowptr[i]; p < diag_idx[i]; ++p)       // colidx < i
        {
            int k = A.colidx[p];          // col k < i
            double Uki = val[diag_idx[k]];
            if (std::fabs(Uki) < EPS)
                val[diag_idx[k]] = EPS;  

            val[p] /= Uki;                // L(i,k) = A(i,k)/U(k,k)

            for (int pk = diag_idx[k] + 1; pk < A.rowptr[k+1]; ++pk)
            {
                int j = A.colidx[pk];
                for (int pi = diag_idx[i] + 1; pi < A.rowptr[i+1]; ++pi)
                    if (A.colidx[pi] == j)
                    {
                        val[pi] -= val[p] * val[pk];
                        break;
                    }
            }
        }

        if (std::fabs(val[diag_idx[i]]) < EPS)
            val[diag_idx[i]] = EPS;

    }

    std::vector<int> Lrow(n+1,0), Urow(n+1,0);
    for (int i=0;i<n;++i)
        for (int p=A.rowptr[i]; p<A.rowptr[i+1]; ++p)
            (A.colidx[p] < i ? Lrow[i+1] : Urow[i+1])++;

    for (int i=0;i<n;++i){ Lrow[i+1]+=Lrow[i]+1; Urow[i+1]+=Urow[i]; } // +1 unit diag

    L.allocate(n,n,Lrow[n]);
    U.allocate(n,n,Urow[n]);

    std::vector<int> nextL=Lrow, nextU=Urow;

    for (int i=0;i<n;++i)
        L.colidx[nextL[i]++] = i, L.data[nextL[i]-1] = 1.0;

    for (int i=0;i<n;++i)
        for (int p=A.rowptr[i]; p<A.rowptr[i+1]; ++p)
        {
            int j=A.colidx[p]; double v=val[p];
            if (j<i){
                int dst=nextL[i]++; L.colidx[dst]=j; L.data[dst]=v;
            }else{
                int dst=nextU[i]++; U.colidx[dst]=j; U.data[dst]=v;
            }
        }

    L.rowptr.swap(Lrow); U.rowptr.swap(Urow);
}