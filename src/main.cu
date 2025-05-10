#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "mmio.h"
#include "loadmm.h"
#include "csr_matrix.h"
#include "levels.h"
#include "dag_cuda_solve.cuh"
#include "ilu.h"
#include "la.h"
#include "la_cuda_solve.cuh"
#include "greedy_gran.h"
#include <nvtx3/nvToolsExt.h>

// =================== serial solver ===================
void reference_solve_csr(const std::vector<int>& rowptr,
    const std::vector<int>& colidx,
    const std::vector<double>& val,
    const std::vector<double>& b,
    std::vector<double>& x)
{
    int n = static_cast<int>(b.size());               

    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        int    diag_idx = -1;

        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
        {
            int j = colidx[p];

            if (j <  i)         sum += val[p] * x[j];   
            else if (j == i)    diag_idx = p;          
        }

        if (diag_idx < 0)
        throw std::runtime_error("missing diagonal at row " + std::to_string(i));

        double a_ii = val[diag_idx];
        if (a_ii == 0.0)
        throw std::runtime_error("zero pivot at row " + std::to_string(i));

        x[i] = (b[i] - sum) / a_ii;
    }
}



// =================== compare results ===================
bool compare_results(const std::vector<double>& ref,
    const std::vector<double>& gpu,
    double rtol = 1e-6,
    double atol = 1e-12){
        for (size_t i = 0; i < ref.size(); ++i) {
            double diff = std::abs(ref[i] - gpu[i]);
            if (diff > atol + rtol * std::abs(ref[i])) {
                std::cerr << "Mismatch at i=" << i
                          << ": ref=" << ref[i]
                          << ", gpu=" << gpu[i]
                          << ", diff=" << diff << '\n';
                return false;
            }
        }
        return true;
}

void print_help() {
    std::cout << "This is a DAG-based lower triangular solver.\n";
    std::cout << "Usage:\n";
    std::cout << "  ./run_sparse <input_file_path> [algorithm] [task_granularity]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  <input_file_path>     Required. Path to the input data file (e.g., a Matrix Market file).\n";
    std::cout << "  [algorithm]           Optional. Scheduling algorithm to use. Options:\n";
    std::cout << "                          - LEVEL (default): level-scheduling\n";
    std::cout << "                          - LA: locality-aware scheduling\n";
    std::cout << "                          - CA: concurrency-aware scheduling\n";
    std::cout << "  [task_granularity]    Optional. Number of user tasks per aggregated task (default: 4).\n";
    std::cout << "\nExample:\n";
    std::cout << "  ./run_sparse data.mtx LA 8\n";
}

template <typename ComputeFunc>
void run_and_time(const std::string& algo_name,
                  const std::vector<double>& y_ref,
                  std::vector<double>& y,
                  ComputeFunc compute_kernel) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    compute_kernel(); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[CUDA " << algo_name << " Solve Time] " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    bool ok = compare_results(y_ref, y);
    std::cout << (ok ? "[PASS] GPU == CPU\n" : "[FAIL] mismatch!\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    char* input_file = argv[1];

    constexpr int TILE_ROWS = 1;  // Reduced from 16 to 8 for better granularity
    // Max # non-zero value     
    constexpr int TILE_NZ   = 128; 
    // ---- Step 1: load matrix
    CSRMatrix A, L, U;
    load_mtx_to_csr(input_file, A);
    cpu_spilu0(A, L, U);
    std::string algorithm = (argc >= 3) ? argv[2] : "LEVEL";
    int task_granularity = (argc >= 4) ? std::stoi(argv[3]) : compute_optimal_granularity(L, 14ULL * 1024 * 1024 * 1024);

    std::cout << "Input: " << input_file << "\n";
    std::cout << "Algorithm: " << algorithm << "\n";
    std::cout << "Granularity: " << task_granularity << "\n";

    int N = L.nrows;
    // ---------- Step‑2 RHS ----------
    std::vector<double> b(N, 1.0);

    // ---- Step 3: serial execution for reference
    std::vector<double> y_ref(L.nrows, 0.0);
    reference_solve_csr(L.rowptr, L.colidx, L.data, b, y_ref);

    std::vector<double> y(L.nrows, 0.0);
    if (algorithm == "LEVEL") {
        std::vector<std::vector<int>> levels;
        compute_levels(L, levels);
        run_and_time(algorithm, y_ref, y, [&]() {
            parallel_dag_lower_triangular_solve_cuda<TILE_ROWS, TILE_NZ>(L, y, b, levels);
        });
    }
    else if (algorithm == "LA") {
        LASchedule schedule;
        compute_la_schedule(L, task_granularity, schedule);
        run_and_time(algorithm, y_ref, y, [&]() {
            nvtxRangePushA("LA Algorithm Total");
            parallel_dag_lower_triangular_solve_cuda_la<TILE_ROWS, TILE_NZ>(L, y, b, schedule);
            nvtxRangePop();
        });
    }
    else if (algorithm == "CA") {
      
    }
    return 0;
}
