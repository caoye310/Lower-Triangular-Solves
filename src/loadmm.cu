// loadmm.cu
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "mmio.h"
#include "csr_matrix.h"

void load_mtx_to_csr(const char* fname, CSRMatrix& A)
{
    /* ---------- read MatrixMarket head ---------- */
    FILE* f = fopen(fname, "r");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
        { fprintf(stderr, "Could not process Matrix Market banner\n"); exit(EXIT_FAILURE); }

    int M, N, nz;
    mm_read_mtx_crd_size(f, &M, &N, &nz);  

    const bool symmetric = mm_is_symmetric(matcode);

    /* ---------- read COO  ---------- */
    std::vector<int>    I;            // row index
    std::vector<int>    J;            // col index
    std::vector<double> V;            // value
    I.reserve(symmetric ? nz * 2 : nz);
    J.reserve(symmetric ? nz * 2 : nz);
    V.reserve(symmetric ? nz * 2 : nz);

    for (int k = 0; k < nz; ++k) {
        int r, c;  double val;
        if (fscanf(f, "%d %d %lf", &r, &c, &val) != 3)
            { fprintf(stderr, "Premature EOF at entry %d\n", k); exit(EXIT_FAILURE); }

        --r; --c;                        
        I.push_back(r);
        J.push_back(c);
        V.push_back(val);

        if (symmetric && r != c) {       
            I.push_back(c);
            J.push_back(r);
            V.push_back(val);
        }
    }
    fclose(f);

    std::vector<char> has_diag(M,0);
    for (size_t k=0;k<I.size();++k)
        if (I[k]==J[k]) has_diag[I[k]]=1;

    for (int i=0;i<M;++i)
        if(!has_diag[i]) {          // append (i,i,0.0) for LU
            I.push_back(i); J.push_back(i); V.push_back(0.0);
        }

    const int nnz = static_cast<int>(I.size());
    A.allocate(M, N, nnz);

    /* ---------- COO → CSR ---------- */
    for (int k = 0; k < nnz; ++k)
        ++A.rowptr[I[k] + 1];

    for (int i = 0; i < M; ++i)
        A.rowptr[i + 1] += A.rowptr[i];

    std::vector<int> offset = A.rowptr;     
    for (int k = 0; k < nnz; ++k) {
        int row = I[k];
        int pos = offset[row]++;           
        A.colidx[pos] = J[k];
        A.data  [pos] = V[k];
    }
}
