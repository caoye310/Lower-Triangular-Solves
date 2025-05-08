#ifndef ILU_H
#define ILU_H

#ifndef EPS

#include "csr_matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>

static const double EPS = 1e-30;  

void cpu_spilu0(const CSRMatrix& A, CSRMatrix& L, CSRMatrix& U);
#endif
#endif