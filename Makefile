NVCC      := nvcc
GXX       := g++
GCC       := gcc

INC       := -Iinclude
NVCCFLAGS := -std=c++17 $(INC) -O2
CPPFLAGS  := -std=c++17 $(INC) -O2
CFLAGS    := $(INC) -O2

CU_SRC   := $(wildcard src/*.cu)
CPP_SRC  := $(wildcard src/*.cpp)
C_SRC    := $(wildcard src/*.c)

CU_OBJS  := $(patsubst src/%.cu,  build/%.o, $(CU_SRC))
CPP_OBJS := $(patsubst src/%.cpp, build/%.o, $(CPP_SRC))
C_OBJS   := $(patsubst src/%.c,   build/%.o, $(C_SRC))
OBJS     := $(CU_OBJS) $(CPP_OBJS) $(C_OBJS)

TARGET   := run_sparse

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

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
