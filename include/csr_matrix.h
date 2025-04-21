#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <vector>
#include <string>

struct CSRMatrix {
    int nrows, ncols, nnz;
    std::vector<int> colidx;
    std::vector<int> rowptr;
    std::vector<double> data;

    CSRMatrix();
    void allocate(int rows, int cols, int nonzeros);
    void clear();
};

void csr_show(const CSRMatrix &M, const std::string &name);
void csr_laplacian(CSRMatrix &M, int nx, int ny, int nz);
void csr_lower(const CSRMatrix &M, CSRMatrix &L);
void csr_upper(const CSRMatrix &M, CSRMatrix &U);
void csr_transpose(const CSRMatrix &M, CSRMatrix &T);

#endif  // CSR_MATRIX_H

