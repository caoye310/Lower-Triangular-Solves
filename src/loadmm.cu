// loadmm.cu
#include <vector>
#include <cstdio>
#include "mmio.h"
#include "csr_matrix.h"

void load_mtx_to_csr(const char* fname, CSRMatrix &A)
{
    FILE* f = fopen(fname, "r");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }

    MM_typecode matcode;
    int M, N, nz;
    mm_read_banner(f, &matcode);
    mm_read_mtx_crd_size(f, &M, &N, &nz);     

    std::vector<int> I(nz), J(nz);
    std::vector<double> V(nz);
    for (int k = 0; k < nz; ++k)
        fscanf(f, "%d %d %lf\n", &I[k], &J[k], &V[k]);
    fclose(f);

    // ---------- COO → CSR ----------
    A.allocate(M, N, nz);

    /* 1. count */
    for (int k = 0; k < nz; ++k)
        A.rowptr[I[k]]++;
    /* 2. exclusive-scan */
    int cumsum = 0;
    for (int i = 0; i <= M; ++i) {
        int tmp   = A.rowptr[i];
        A.rowptr[i] = cumsum;
        cumsum += tmp;
    }
    /* 3. coloumns and data */
    for (int k = 0; k < nz; ++k) {
        int row = I[k];
        int dst = A.rowptr[row]++;
        A.colidx[dst] = J[k];
        A.data  [dst] = V[k];
    }
    /* 4. modify rowptr */
    for (int i = M; i > 0; --i)
        A.rowptr[i] = A.rowptr[i-1];
    A.rowptr[0] = 0;
}
