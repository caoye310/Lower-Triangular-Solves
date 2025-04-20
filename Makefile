NVCC=nvcc
CFLAGS=-std=c++17 -Iinclude

all: run_sparse

run_sparse: src/main.cu src/loadmm.cu
	$(NVCC) $(CFLAGS) -o run_sparse src/main.cu src/loadmm.cu

clean:
	rm -f run_sparse
