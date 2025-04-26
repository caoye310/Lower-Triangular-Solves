CUDA_HOME ?= /opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.8

INC_BASE   := -Iinclude
CUDA_INC   := -I$(CUDA_HOME)/include            # 仅 nvcc 使用
LDLIBS     := -L$(CUDA_HOME)/lib64

NVCC       := nvcc
GXX        := g++
GCC        := gcc

NVCCFLAGS  := -std=c++17 $(INC_BASE) $(CUDA_INC) -O2
CPPFLAGS   := -std=c++17 $(INC_BASE)             -O2
CFLAGS     :=            $(INC_BASE)             -O2

CU_SRC     := $(wildcard src/*.cu)
CPP_SRC    := $(wildcard src/*.cpp)
C_SRC      := $(wildcard src/*.c)

CU_OBJS    := $(patsubst src/%.cu,  build/%.o, $(CU_SRC))
CPP_OBJS   := $(patsubst src/%.cpp, build/%.o, $(CPP_SRC))
C_OBJS     := $(patsubst src/%.c,   build/%.o, $(C_SRC))
OBJS       := $(CU_OBJS) $(CPP_OBJS) $(C_OBJS)

TARGET     := run_sparse

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^ $(LDLIBS)

$(shell mkdir -p build)

build/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/%.o: src/%.cpp
	$(GXX)  $(CPPFLAGS)  -c $< -o $@

build/%.o: src/%.c
	$(GCC)  $(CFLAGS)    -c $< -o $@

clean:
	rm -rf build/*.o $(TARGET)

.PHONY: all clean
