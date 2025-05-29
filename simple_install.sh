#!/bin/bash

# Simple and Robust Installation Script for Masked Attention CUDA Extension
# Compatible with all Unix-like systems

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m' 
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================="
echo "Masked Attention CUDA Extension Installer"
echo "============================================="

# Check CUDA
log_info "Checking CUDA..."
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_success "Found CUDA $CUDA_VERSION"
    
    # Set CUDA_HOME if not set
    if [ -z "$CUDA_HOME" ]; then
        for cuda_path in /usr/local/cuda /opt/cuda /usr/cuda; do
            if [ -d "$cuda_path" ]; then
                export CUDA_HOME="$cuda_path"
                break
            fi
        done
        if [ -z "$CUDA_HOME" ]; then
            export CUDA_HOME="/usr/local/cuda"
        fi
    fi
    log_success "CUDA_HOME: $CUDA_HOME"
else
    log_error "CUDA not found! Please install CUDA toolkit first."
    exit 1
fi

# Check Python and PyTorch
log_info "Checking Python and PyTorch..."
if ! command -v python3 >/dev/null 2>&1; then
    log_error "python3 not found!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | sed 's/Python //')
log_success "Found Python $PYTHON_VERSION"

if ! python3 -c "import torch" >/dev/null 2>&1; then
    log_error "PyTorch not found! Please install PyTorch first."
    exit 1
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
log_success "Found PyTorch $TORCH_VERSION"

if ! python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    log_error "PyTorch CUDA support not available!"
    exit 1
fi

TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
log_success "PyTorch CUDA support: $TORCH_CUDA_VERSION"

# Check GPU
log_info "Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    log_success "Found $GPU_COUNT GPU(s)"
    
    # Get GPU arch for current device
    GPU_ARCH=$(python3 -c "
import torch
try:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f'{cap[0]}{cap[1]}')
    else:
        print('75')
except:
    print('75')
" 2>/dev/null)
    export GPU_ARCH="$GPU_ARCH"
    log_success "GPU architecture: SM_$GPU_ARCH"
else
    log_warning "nvidia-smi not found, using default GPU arch"
    export GPU_ARCH="75"
fi

# Install dependencies
log_info "Installing dependencies..."
python3 -m pip install --upgrade pip >/dev/null 2>&1

if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    log_success "Installed from requirements.txt"
else
    pip3 install torch torchvision transformers numpy pybind11 setuptools wheel
    log_success "Installed essential dependencies"
fi

# Build extension
log_info "Building CUDA extension..."

# Method 1: Try setup.py
if [ -f "setup.py" ]; then
    log_info "Trying setup.py build..."
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
    
    if python3 setup.py build_ext --inplace >/dev/null 2>&1 && \
       python3 setup.py install >/dev/null 2>&1; then
        log_success "Built with setup.py"
        BUILD_SUCCESS=true
    else
        log_warning "setup.py build failed"
        BUILD_SUCCESS=false
    fi
fi

# Method 2: Try Makefile if setup.py failed
if [ "$BUILD_SUCCESS" != "true" ] && [ -f "Makefile" ]; then
    log_info "Trying Makefile build..."
    if make install >/dev/null 2>&1; then
        log_success "Built with Makefile"
        BUILD_SUCCESS=true
    else
        log_warning "Makefile build failed"
        BUILD_SUCCESS=false
    fi
fi

# Method 3: JIT compilation as last resort
if [ "$BUILD_SUCCESS" != "true" ]; then
    log_info "Trying JIT compilation..."
    python3 -c "
import torch
from torch.utils.cpp_extension import load
import os

try:
    ext = load(
        name='masked_attention_cuda',
        sources=['masked_attention_kernel.cu'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        extra_ldflags=['-lcublas', '-lcublasLt'],
        verbose=False
    )
    print('JIT build successful')
except Exception as e:
    print(f'JIT build failed: {e}')
    exit(1)
" >/dev/null 2>&1

    if [ $? -eq 0 ]; then
        log_success "Built with JIT compilation"
        BUILD_SUCCESS=true
    else
        log_error "All build methods failed!"
        BUILD_SUCCESS=false
    fi
fi

if [ "$BUILD_SUCCESS" != "true" ]; then
    log_error "Build failed! Please check CUDA installation and try again."
    exit 1
fi

# Test installation
log_info "Testing installation..."

# Test CUDA extension import
if python3 -c "import masked_attention_cuda" >/dev/null 2>&1; then
    log_success "CUDA extension import OK"
else
    log_error "CUDA extension import failed"
    exit 1
fi

# Test Python module import
if python3 -c "from masked_vit_attention import MaskedViTAttention" >/dev/null 2>&1; then
    log_success "Python module import OK"
else
    log_error "Python module import failed"
    exit 1
fi

# Quick performance test
log_info "Running quick performance test..."
python3 -c "
import torch
from masked_vit_attention import MaskedViTAttention, create_random_mask
import time

class Config:
    hidden_size = 768
    num_attention_heads = 12
    attention_probs_dropout_prob = 0.0
    hidden_dropout_prob = 0.0

try:
    config = Config()
    device = torch.device('cuda')
    
    # Small test
    hidden_states = torch.randn(2, 197, 768, device=device)
    mask = create_random_mask(2, 197, 0.15, device)
    
    attention = MaskedViTAttention(config, use_cuda_kernel=True).to(device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = attention(hidden_states, mask)
    torch.cuda.synchronize()
    
    # Time test
    start_time = time.time()
    for _ in range(5):
        with torch.no_grad():
            _ = attention(hidden_states, mask)
    torch.cuda.synchronize()
    
    avg_time = (time.time() - start_time) / 5 * 1000
    print(f'Performance test OK: {avg_time:.2f}ms per iteration')
    
except Exception as e:
    print(f'Performance test failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_success "Performance test passed"
else
    log_warning "Performance test failed, but installation may still work"
fi

# Success message
echo
echo "============================================="
log_success "🎉 INSTALLATION COMPLETED SUCCESSFULLY! 🎉"
echo "============================================="
echo
echo "Quick start:"
echo "  from masked_vit_attention import MaskedViTAttention"
echo "  attention = MaskedViTAttention(config, use_cuda_kernel=True)"
echo
echo "Run examples:"
echo "  python3 example_usage.py"
echo
echo "Run performance benchmark:"
echo "  python3 performance_benchmark.py"
echo
echo "Run full tests:"
echo "  python3 test_masked_attention.py"
echo

exit 0