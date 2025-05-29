# Makefile for Masked Attention CUDA Extension

# Python and CUDA paths
PYTHON := python3
NVCC := nvcc
CXX := g++

# Get PyTorch paths
TORCH_PATH := $(shell $(PYTHON) -c "import torch; print(torch.__path__[0])")
TORCH_INCLUDE := $(TORCH_PATH)/include
TORCH_LIB := $(TORCH_PATH)/lib

# Get Python include path
PYTHON_INCLUDE := $(shell $(PYTHON) -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

# Get pybind11 include path
PYBIND11_INCLUDE := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())")

# CUDA paths
CUDA_HOME := /usr/local/cuda
CUDA_INCLUDE := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

# Compiler flags
CXXFLAGS := -O3 -std=c++17 -fPIC -shared
NVCCFLAGS := -O3 -std=c++17 --use_fast_math -Xcompiler -fPIC
NVCCFLAGS += -gencode=arch=compute_70,code=sm_70
NVCCFLAGS += -gencode=arch=compute_75,code=sm_75
NVCCFLAGS += -gencode=arch=compute_80,code=sm_80
NVCCFLAGS += -gencode=arch=compute_86,code=sm_86
NVCCFLAGS += -gencode=arch=compute_89,code=sm_89
NVCCFLAGS += -gencode=arch=compute_90,code=sm_90

# Include directories
INCLUDES := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include
INCLUDES += -I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE)
INCLUDES += -I$(CUDA_INCLUDE)

# Library directories and libraries
LDFLAGS := -L$(TORCH_LIB) -L$(CUDA_LIB)
LIBS := -ltorch -ltorch_cuda -lcudart -lcublas -lcublasLt

# Source and target
SOURCE := masked_attention_kernel.cu
TARGET := masked_attention_cuda.so

.PHONY: all install clean test

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Building CUDA extension..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $(SOURCE) -o masked_attention_kernel.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) masked_attention_kernel.o -o $(TARGET) $(LDFLAGS) $(LIBS)
	@echo "Build completed: $(TARGET)"

install: all
	@echo "Installing via setup.py..."
	$(PYTHON) setup.py build_ext --inplace
	$(PYTHON) setup.py install
	@echo "Installation completed!"

clean:
	@echo "Cleaning build files..."
	rm -f *.o *.so
	rm -rf build/ dist/ *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "Clean completed!"

test: install
	@echo "Running tests..."
	$(PYTHON) test_masked_attention.py
	@echo "Tests completed!"

# Alternative quick build using PyTorch's extension utilities
quick:
	@echo "Quick build using PyTorch extension utilities..."
	$(PYTHON) -c "
import torch
from torch.utils.cpp_extension import load
ext = load(name='masked_attention_cuda', sources=['masked_attention_kernel.cu'], verbose=True)
print('Quick build completed!')
"

help:
	@echo "Masked Attention CUDA Extension Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build the CUDA extension"
	@echo "  install  - Build and install the extension"
	@echo "  clean    - Remove build artifacts"
	@echo "  test     - Run tests after installation"
	@echo "  quick    - Quick build using PyTorch utilities"
	@echo "  help     - Show this help message"