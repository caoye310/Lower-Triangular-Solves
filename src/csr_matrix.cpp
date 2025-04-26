#include "csr_matrix.h"
#include <iostream>
#include <iomanip>

CSRMatrix::CSRMatrix() : nrows(0), ncols(0), nnz(0) {}

void CSRMatrix::allocate(int rows, int cols, int nonzeros) {
    nrows = rows;
    ncols = cols;
    nnz = nonzeros;
    colidx.resize(nnz);
    rowptr.resize(nrows + 1);
    data.resize(nnz, 0.0);
}

void CSRMatrix::clear() {
    colidx.clear();
    rowptr.clear();
    data.clear();
    nrows = ncols = nnz = 0;
}

// Function to display CSR matrix
void csr_show(const CSRMatrix &M, const std::string &name) {
    std::cout << name << " = [\n";
    int k = 0, j = 0;

    for (int n = 0; n < M.nrows; n++) {
        j = 0;
        while (k < M.rowptr[n + 1]) {
            while (j < M.colidx[k]) {
                std::cout << "   -  "; // Print empty spaces for zero entries
                j++;
            }
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << M.data[k] << " ";
            k++;
            j++;
        }
        while (j < M.ncols) {
            std::cout << "   -  ";
            j++;
        }
        std::cout << "\n";
    }
    std::cout << "];\n\n";
}

// Function to create Laplacian in CSR format
void csr_laplacian(CSRMatrix &M, int nx, int ny, int nz) {
    int n = nx * ny * nz;
    int nnz = n + 2 * (nx - 1) * ny * nz + 2 * nx * (ny - 1) * nz + 2 * nx * ny * (nz - 1);
    
    M.allocate(n, n, nnz);
    
    int m = 0;
    M.rowptr[0] = 0;

    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nx; k++) {
                if (i > 0) {
                    M.colidx[m] = (i - 1) * nx * ny + j * nx + k;
                    M.data[m] = -1.0;
                    m++;
                }
                if (j > 0) {
                    M.colidx[m] = i * nx * ny + (j - 1) * nx + k;
                    M.data[m] = -1.0;
                    m++;
                }
                if (k > 0) {
                    M.colidx[m] = i * nx * ny + j * nx + k - 1;
                    M.data[m] = -1.0;
                    m++;
                }

                M.colidx[m] = i * nx * ny + j * nx + k;
                M.data[m] = 6.0;
                m++;

                if ((nx - k) > 1) {
                    M.colidx[m] = i * nx * ny + j * nx + k + 1;
                    M.data[m] = -1.0;
                    m++;
                }
                if ((ny - j) > 1) {
                    M.colidx[m] = i * nx * ny + (j + 1) * nx + k;
                    M.data[m] = -1.0;
                    m++;
                }
                if ((nz - i) > 1) {
                    M.colidx[m] = (i + 1) * nx * ny + j * nx + k;
                    M.data[m] = -1.0;
                    m++;
                }
                M.rowptr[i * nx * ny + j * nx + k + 1] = m;
            }
        }
    }
}

// Extract strictly lower triangular part of M (excluding diagonal)
void csr_lower(const CSRMatrix &M, CSRMatrix &L) {
    int nnzL = 0;
    for (int i = 0; i < M.nrows; i++) {
        for (int k = M.rowptr[i]; k < M.rowptr[i + 1]; k++) {
            if (M.colidx[k] < i) nnzL++; // Count strictly lower elements
        }
    }

    L.allocate(M.nrows, M.ncols, nnzL);
    int m = 0;
    L.rowptr[0] = 0;

    for (int i = 0; i < M.nrows; i++) {
        for (int k = M.rowptr[i]; k < M.rowptr[i + 1]; k++) {
            int j = M.colidx[k];
            if (j < i) {
                L.colidx[m] = j;
                L.data[m] = M.data[k]; // Copy value
                m++;
            }
        }
        L.rowptr[i + 1] = m;
    }
}

// Extract upper triangular part of M (including diagonal)
void csr_upper(const CSRMatrix &M, CSRMatrix &U) {
    int nnzU = 0;
    for (int i = 0; i < M.nrows; i++) {
        for (int k = M.rowptr[i]; k < M.rowptr[i + 1]; k++) {
            if (M.colidx[k] >= i) nnzU++; // Count diagonal & upper elements
        }
    }

    U.allocate(M.nrows, M.ncols, nnzU);
    int m = 0;
    U.rowptr[0] = 0;

    for (int i = 0; i < M.nrows; i++) {
        for (int k = M.rowptr[i]; k < M.rowptr[i + 1]; k++) {
            int j = M.colidx[k];
            if (j >= i) {
                U.colidx[m] = j;
                U.data[m] = M.data[k]; // Copy value including diagonal
                m++;
            }
        }
        U.rowptr[i + 1] = m;
    }
}


// Function to compute CSR transpose
void csr_transpose(const CSRMatrix &M, CSRMatrix &T) {
    int nrows = M.ncols;
    int ncols = M.nrows;
    int nnz = M.nnz;
    T.allocate(nrows, ncols, nnz);

    for (int i = 0; i < nrows + 1; i++) T.rowptr[i] = 0;

    for (int k = 0; k < nnz; k++) {
        int j = M.colidx[k] + 1;
        T.rowptr[j]++;
    }

    for (int i = 0; i < nrows; i++) {
        T.rowptr[i + 1] += T.rowptr[i];
    }

    int k = 0;
    for (int i = 0; i < M.nrows; i++) {
        while (k < M.rowptr[i + 1]) {
            int j = M.colidx[k];
            int m = T.rowptr[j]++;
            T.colidx[m] = i;
            T.data[m] = M.data[k];
            k++;
        }
    }

    for (int i = nrows; i > 0; i--) {
        T.rowptr[i] = T.rowptr[i - 1];
    }
    T.rowptr[0] = 0;
}