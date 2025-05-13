# Parallel Sparse Matrix Solver

This project implements a parallel sparse matrix solver using CUDA, featuring different optimization strategies including Level Analysis (LA) and Cache-Aware (CA) approaches.

## Prerequisites

- CUDA Toolkit (version 12.8 or compatible)
- NVIDIA GPU with CUDA support
- GCC/G++ compiler
- Make
- NVIDIA Nsight Systems (nsys) for performance profiling (optional)

## Project Structure

```
.
├── include/         # Header files
├── src/            # Source files
├── data/           # Test matrices
├── build/          # Build artifacts
├── Makefile        # Build configuration
└── run.sh          # Execution script
```

## Building the Project

1. Make sure you have CUDA toolkit installed. The default path is set to `/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.8`. If your CUDA installation is in a different location, modify the `CUDA_HOME` variable in the Makefile.

2. Build the project:
```bash
make clean    # Clean previous build artifacts
make         # Build the project
```

This will create an executable named `run_sparse` in the root directory.

## Running the Project

The project comes with a convenient `run.sh` script that runs the solver on multiple test matrices with different optimization strategies.

### Available Test Matrices
- onetone1
- onetone2
- bcircuit
- G2_circuit
- hcircuit
- parabolic_fem

### Running Options

1. Run with default settings:
```bash
./run.sh
```

2. Run with a specific size parameter:
```bash
./run.sh <size>
```

The script will:
- Run each matrix 20 times
- Use the LA_OPT optimization strategy
- Generate performance metrics for analysis

### Manual Execution

You can also run the solver manually:
```bash
./run_sparse <matrix_file> [optimization_strategy] [size]
```

Where:
- `<matrix_file>`: Path to the matrix file (e.g., "data/onetone1/onetone1.mtx")
- `[optimization_strategy]`: Optional optimization strategy (e.g., "LA_OPT")
- `[size]`: Optional size parameter

### Performance Profiling with Nsight Systems

To get detailed performance metrics using NVIDIA Nsight Systems:
```bash
nsys profile --stats=true ./run_sparse <matrix_file> [optimization_strategy] [size]
```

For example:
```bash
nsys profile --stats=true ./run_sparse ./data/onetone1/onetone1 LA_OPT
```

## Results

The solver compares its results with serial execution for validation. The console output shows the running time and verification status:

```
Input: ./data/onetone1/onetone1.mtx
Algorithm: LA_OPT
Granularity: 2048
[CUDA LA_OPT Solve Time] 114.96 ms
[PASS] GPU == CPU
```

When running with nsys, the solver generates performance metrics and analysis files:
- `.nsys-rep` files: NVIDIA Nsight Systems performance reports
- `.sqlite` files: Performance data in SQLite format

## Cleaning Up

To clean build artifacts:
```bash
make clean
```

## References

Helal, Ahmed E., et al. "Adaptive task aggregation for high-performance sparse solvers on GPUs." 2019 28th International Conference on Parallel Architectures and Compilation Techniques (PACT). IEEE, 2019.
