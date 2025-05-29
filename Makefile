# Makefile for Optimized Masked Attention CUDA Extension

# Python and CUDA paths
PYTHON := python3
NVCC := nvcc
CXX := g++

# Auto-detect CUDA installation
CUDA_HOME := $(shell if [ -d "/usr/local/cuda" ]; then echo "/usr/local/cuda"; elif [ -d "/opt/cuda" ]; then echo "/opt/cuda"; elif [ -d "$CUDA_HOME" ]; then echo "$CUDA_HOME"; else echo "/usr/local/cuda"; fi)

# Get PyTorch paths
TORCH_PATH := $(shell $(PYTHON) -c "import torch; print(torch.__path__[0])")
TORCH_INCLUDE := $(TORCH_PATH)/include
TORCH_LIB := $(TORCH_PATH)/lib

# Get Python include path
PYTHON_INCLUDE := $(shell $(PYTHON) -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

# Get pybind11 include path
PYBIND11_INCLUDE := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())")

# CUDA paths
CUDA_INCLUDE := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

# Detect GPU compute capability
GPU_ARCH := $(shell $(PYTHON) -c "import torch; cap=torch.cuda.get_device_capability() if torch.cuda.is_available() else (7,0); print(f'{cap[0]}{cap[1]}')" 2>/dev/null || echo "75")

# Compiler flags
CXXFLAGS := -O3 -std=c++17 -fPIC -shared -Wall -Wextra -ffast-math
NVCCFLAGS := -O3 -std=c++17 --use_fast_math --extended-lambda --expt-relaxed-constexpr -Xcompiler -fPIC -Xcompiler -Wall -lineinfo

# GPU architecture flags (updated for better performance)
NVCCFLAGS += -gencode=arch=compute_70,code=sm_70   # V100
NVCCFLAGS += -gencode=arch=compute_75,code=sm_75   # RTX 20xx, T4
NVCCFLAGS += -gencode=arch=compute_80,code=sm_80   # A100
NVCCFLAGS += -gencode=arch=compute_86,code=sm_86   # RTX 30xx
NVCCFLAGS += -gencode=arch=compute_89,code=sm_89   # RTX 40xx, L40
NVCCFLAGS += -gencode=arch=compute_90,code=sm_90   # H100

# Add specific architecture for current GPU
NVCCFLAGS += -gencode=arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

# Performance optimization flags
NVCCFLAGS += --ptxas-options=-v --compiler-options -ffast-math

# Include directories
INCLUDES := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include
INCLUDES += -I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE)
INCLUDES += -I$(CUDA_INCLUDE)

# Library directories and libraries (updated for cuBLAS)
LDFLAGS := -L$(TORCH_LIB) -L$(CUDA_LIB)
LIBS := -ltorch -ltorch_cuda -lcudart -lcublas -lcublasLt

# Source and target
SOURCE := masked_attention_kernel.cu
TARGET := masked_attention_cuda.so

.PHONY: all install clean test debug profile info

all: info $(TARGET)

info:
	@echo "======================================"
	@echo "Masked Attention CUDA Extension Build"
	@echo "======================================"
	@echo "CUDA Home: $(CUDA_HOME)"
	@echo "GPU Architecture: SM_$(GPU_ARCH)"
	@echo "PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA Version: $(shell $(NVCC) --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')"
	@echo "======================================"

$(TARGET): $(SOURCE)
	@echo "Building optimized CUDA extension..."
	@echo "Compiling CUDA kernel..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $(SOURCE) -o masked_attention_kernel.o
	@echo "Linking shared library..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) masked_attention_kernel.o -o $(TARGET) $(LDFLAGS) $(LIBS)
	@echo "✅ Build completed: $(TARGET)"

install: all
	@echo "Installing via setup.py..."
	$(PYTHON) setup.py build_ext --inplace
	$(PYTHON) setup.py install
	@echo "✅ Installation completed!"

# Debug build with additional flags
debug: NVCCFLAGS += -g -G -DDEBUG
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)
	@echo "✅ Debug build completed!"

# Profile build for performance analysis
profile: NVCCFLAGS += -lineinfo --generate-line-info
profile: $(TARGET)
	@echo "✅ Profile build completed!"

clean:
	@echo "Cleaning build files..."
	rm -f *.o *.so
	rm -rf build/ dist/ *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean completed!"

test: install
	@echo "Running tests..."
	$(PYTHON) test_masked_attention.py
	@echo "✅ Tests completed!"

# Performance benchmark
benchmark: install
	@echo "Running performance benchmark..."
	$(PYTHON) performance_benchmark.py
	@echo "✅ Benchmark completed!"

# Alternative quick build using PyTorch's extension utilities
quick:
	@echo "Quick build using PyTorch extension utilities..."
	$(PYTHON) -c "
import torch
from torch.utils.cpp_extension import load
ext = load(
    name='masked_attention_cuda', 
    sources=['masked_attention_kernel.cu'], 
    extra_cuda_cflags=[
        '-O3', '--use_fast_math', '--extended-lambda',
        '-gencode=arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)'
    ],
    extra_ldflags=['-lcublas', '-lcublasLt'],
    verbose=True
)
print('✅ Quick build completed!')
"

# Check system requirements
check:
	@echo "Checking system requirements..."
	@echo "Python version:"
	@$(PYTHON) --version
	@echo "PyTorch version and CUDA support:"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
	@echo "NVCC version:"
	@$(NVCC) --version || echo "NVCC not found!"
	@echo "GPU information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found!"
	@echo "CUDA libraries:"
	@ls -la $(CUDA_LIB)/libcublas* 2>/dev/null || echo "cuBLAS libraries not found in $(CUDA_LIB)"

# Validate installation
validate: install
	@echo "Validating installation..."
	@$(PYTHON) -c "
try:
    import masked_attention_cuda
    print('✅ CUDA extension imported successfully')
    from masked_vit_attention import MaskedViTAttention
    print('✅ Python module imported successfully')
    print('✅ Installation validation passed!')
except Exception as e:
    print(f'❌ Validation failed: {e}')
    exit(1)
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