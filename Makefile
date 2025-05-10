CUDA_HOME  := /usr/local/cuda
CUDA_INC   := -I$(CUDA_HOME)/include            # Only used by nvcc
CUDA_LIB   := -L$(CUDA_HOME)/lib64 -lcudart

CXX        := g++
CXXFLAGS   := -O3 -Wall -std=c++17
NVCC       := nvcc
NVCCFLAGS  := -O3 -arch=sm_75

TARGET     := run_sparse
SRCS       := src/main.cu src/loadmm.cpp src/mmio.cpp src/levels.cpp src/ilu.cpp src/greedy_gran.cpp
OBJS       := $(SRCS:.cpp=.o)
OBJS       := $(OBJS:.cu=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(CUDA_LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
