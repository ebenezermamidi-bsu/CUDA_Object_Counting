CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

TARGET    := bin/coco_object_counter

INC       := -Isrc -I$(CUDA_PATH)/include
NVFLAGS   := -O2 -std=c++17 -gencode arch=compute_89,code=sm_89
CXXFLAGS  := -O2 -std=c++17
LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart

CUDA_SRCS := src/main.cu src/gpu_pipeline.cu
CPP_SRCS  := src/image_io.cpp src/region_analysis.cpp src/utils.cpp

all: $(TARGET)

$(TARGET): $(CUDA_SRCS) $(CPP_SRCS)
	mkdir -p bin
	$(NVCC) $(NVFLAGS) $(INC) $(CUDA_SRCS) $(CPP_SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf bin output/masks output/cleaned output/labeled output/stats output/run_log.txt

.PHONY: all clean
