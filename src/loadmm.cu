#include <vector>
#include <fstream>
#include <iostream>
#include "mmio.h"  // SuiteSparse or https://math.nist.gov/MatrixMarket/mmio-c.html

void load_mtx_to_csr(const char* fname,
                     std::vector<int> &rowptr,
                     std::vector<int> &colidx,
                     std::vector<double> &val) {
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;

    f = fopen(fname, "r");
    mm_read_banner(f, &matcode);
    mm_read_mtx_crd_size(f, &M, &N, &nz);

    std::vector<int> I(nz), J(nz);
    std::vector<float> V(nz);
    for (int i = 0; i < nz; i++)
        fscanf(f, "%d %d %f\n", &I[i], &J[i], &V[i]);

    // Convert COO to CSR
    rowptr.assign(M + 1, 0);
    colidx.resize(nz);
    val.resize(nz);

    for (int i = 0; i < nz; i++)
        rowptr[I[i]]++;  // count entries per row
    for (int i = 0, cumsum = 0; i <= M; i++) {
        int tmp = rowptr[i];
        rowptr[i] = cumsum;
        cumsum += tmp;
    }
    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dst = rowptr[row]++;
        colidx[dst] = J[i];
        val[dst] = V[i];
    }
    // Fix rowptr
    for (int i = M; i > 0; --i)
        rowptr[i] = rowptr[i-1];
    rowptr[0] = 0;
}
