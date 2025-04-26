#include "csr_matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>

static const double EPS = 1e-30;  

void cpu_spilu0(const CSRMatrix& A, CSRMatrix& L, CSRMatrix& U);