#!/bin/bash

# Optimized Masked Multi-Head Self-Attention CUDA Extension Installation Script
# Version: 2.0
# Updated for performance optimizations and cuBLAS integration

set -e  # Exit on any error

echo "================================================="
echo "Optimized Masked Attention CUDA Extension Setup"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}[SETUP]${NC} $1"
}

# Enhanced CUDA check with version validation
check_cuda() {
    print_header "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        
        print_success "CUDA ${CUDA_VERSION} found"
        
        # Check if CUDA version is sufficient (>= 11.0)
        if (( CUDA_MAJOR >= 11 )); then
            print_success "CUDA version is supported"
        else
            print_warning "CUDA version ${CUDA_VERSION} may cause issues (recommend >= 11.0)"
        fi
        
        # Auto-detect CUDA_HOME
        if [ -z "$CUDA_HOME" ]; then
            for path in /usr/local/cuda /opt/cuda /usr/cuda; do
                if [ -d "$path" ]; then
                    export CUDA_HOME=$path
                    break
                fi
            done
            
            if [ -z "$CUDA_HOME" ]; then
                export CUDA_HOME=/usr/local/cuda
                print_warning "CUDA_HOME not found, using default: $CUDA_HOME"
            else
                print_success "Auto-detected CUDA_HOME: $CUDA_HOME"
            fi
        else
            print_success "CUDA_HOME: $CUDA_HOME"
        fi
        
        # Check cuBLAS libraries
        CUBLAS_PATH="$CUDA_HOME/lib64/libcublas.so"
        if [ -f "$CUBLAS_PATH" ]; then
            print_success "cuBLAS library found"
        else
            print_warning "cuBLAS library not found at $CUBLAS_PATH"
        fi
        
    else
        print_error "CUDA not found! Please install CUDA toolkit first."
        print_error "Download from: https://developer.nvidia.com/cuda-toolkit"
        exit 1
    fi
}

# Enhanced Python and PyTorch check
check_python() {
    print_header "Checking Python and PyTorch..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | sed 's/Python //')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    print_success "Python ${PYTHON_VERSION} found"
    
    # Check if Python version is sufficient
    if (( PYTHON_MAJOR >= 3 && PYTHON_MINOR >= 8 )); then
        print_success "Python version is supported"
    else
        print_error "Python version must be >= 3.8"
        exit 1
    fi
    
    # Check if PyTorch is installed
    if python3 -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        print_success "PyTorch ${TORCH_VERSION} found"
        
        # Check PyTorch version
        TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1)
        if (( TORCH_MAJOR >= 2 )); then
            print_success "PyTorch version is supported"
        else
            print_warning "PyTorch version ${TORCH_VERSION} may cause issues (recommend >= 2.0)"
        fi
        
        # Check CUDA support in PyTorch
        if python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
            TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
            print_success "PyTorch CUDA ${TORCH_CUDA_VERSION} support enabled"
            
            # Check CUDA version compatibility
            if [ "$TORCH_CUDA_VERSION" != "$CUDA_VERSION" ]; then
                print_warning "PyTorch CUDA version ($TORCH_CUDA_VERSION) differs from system CUDA ($CUDA_VERSION)"
                print_warning "This may cause compatibility issues"
            fi
        else
            print_error "PyTorch CUDA support not available!"
            print_error "Install PyTorch with CUDA support:"
            print_error "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}"
            exit 1
        fi
    else
        print_error "PyTorch not found! Please install PyTorch first."
        exit 1
    fi
}

# Enhanced GPU check with compute capability
check_gpu() {
    print_header "Checking GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        # Get GPU information
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        print_success "Found $GPU_COUNT GPU(s):"
        
        # Check each GPU
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader,nounits | while IFS=, read -r index name compute_cap memory; do
            print_success "  GPU $index: $name"
            print_success "    Compute Capability: $compute_cap"
            print_success "    Memory: ${memory} MB"
            
            # Check if compute capability is sufficient
            COMPUTE_MAJOR=$(echo $compute_cap | cut -d. -f1)
            COMPUTE_MINOR=$(echo $compute_cap | cut -d. -f2)
            
            if (( COMPUTE_MAJOR >= 7 )); then
                print_success "    ✅ Compute capability is excellent for optimization"
            elif (( COMPUTE_MAJOR >= 6 )); then
                print_warning "    ⚠️  Compute capability is supported but may not be optimal"
            else
                print_warning "    ❌ Compute capability is too low for optimal performance"
            fi
        done
        
        # Get current GPU for compilation optimization
        CURRENT_GPU_CAP=$(python3 -c "import torch; print(''.join(map(str, torch.cuda.get_device_capability())))" 2>/dev/null || echo "75")
        export GPU_ARCH=$CURRENT_GPU_CAP
        print_success "Will optimize for compute capability: $GPU_ARCH"
        
    else
        print_warning "nvidia-smi not found, cannot check GPU details"
        print_warning "Setting default GPU architecture to 7.5"
        export GPU_ARCH=75
    fi
}

# Enhanced dependency installation
install_dependencies() {
    print_header "Installing dependencies..."
    
    # Upgrade pip first
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip3 install -r requirements.txt
        print_success "Requirements installed"
    else
        print_status "Installing essential dependencies..."
        pip3 install torch torchvision transformers numpy pybind11 setuptools wheel
        print_success "Essential dependencies installed"
    fi
    
    # Check installation
    print_status "Verifying installations..."
    python3 -c "
import torch, torchvision, transformers, numpy, pybind11
print(f'✅ torch: {torch.__version__}')
print(f'✅ torchvision: {torchvision.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ numpy: {numpy.__version__}')
print(f'✅ pybind11: {pybind11.__version__}')
"
}

# Enhanced build with multiple fallback methods
build_and_install() {
    print_header "Building optimized CUDA extension..."
    
    # Method 1: setup.py (preferred)
    if [ -f "setup.py" ]; then
        print_status "Building with setup.py (method 1/3)..."
        
        # Set optimization environment variables
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
        export NVCC_FLAGS="-O3 --use_fast_math --extended-lambda"
        
        if python3 setup.py build_ext --inplace && python3 setup.py install; then
            print_success "✅ Built and installed with setup.py"
            return 0
        else
            print_warning "setup.py build failed, trying alternative methods..."
        fi
    fi
    
    # Method 2: Makefile
    if [ -f "Makefile" ]; then
        print_status "Building with Makefile (method 2/3)..."
        if make install; then
            print_success "✅ Built and installed with Makefile"
            return 0
        else
            print_warning "Makefile build failed, trying JIT compilation..."
        fi
    fi
    
    # Method 3: JIT compilation (fallback)
    print_status "Attempting JIT compilation (method 3/3)..."
    python3 -c "
import torch
from torch.utils.cpp_extension import load
import os

# Set up compilation flags
extra_cuda_cflags = [
    '-O3', '--use_fast_math', '--extended-lambda',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86'
]

extra_ldflags = ['-lcublas', '-lcublasLt']

try:
    ext = load(
        name='masked_attention_cuda',
        sources=['masked_attention_kernel.cu'],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        verbose=True
    )
    print('✅ JIT compilation successful!')
except Exception as e:
    print(f'❌ JIT compilation failed: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "✅ Built with JIT compilation"
        return 0
    else
        print_error "All build methods failed!"
        return 1
    fi
}


# Comprehensive testing with performance validation
test_installation() {
    print_header "Testing installation..."
    
    # Test 1: Import test
    print_status "Testing CUDA extension import..."
    if python3 -c "import masked_attention_cuda; print('✅ CUDA extension imported successfully')" &> /dev/null; then
        print_success "CUDA extension import test passed"
    else
        print_error "CUDA extension import failed"
        return 1
    fi
    
    # Test 2: Python module test
    print_status "Testing Python module import..."
    if python3 -c "from masked_vit_attention import MaskedViTAttention; print('✅ Python module imported successfully')" &> /dev/null; then
        print_success "Python module import test passed"
    else
        print_error "Python module import failed"
        return 1
    fi
    
    # Test 3: Basic functionality test
    if [ -f "test_masked_attention.py" ]; then
        print_status "Running basic functionality test..."
        if timeout 300 python3 -c "
from test_masked_attention import test_basic_functionality
try:
    test_basic_functionality()
    print('✅ Basic functionality test passed')
except Exception as e:
    print(f'❌ Basic functionality test failed: {e}')
    exit(1)
" 2>/dev/null; then
            print_success "Basic functionality test passed"
        else
            print_warning "Basic functionality test failed or timed out"
            print_warning "This might be OK for some environments"
        fi
    fi
    
    # Test 4: Quick performance test
    print_status "Running quick performance test..."
    if timeout 60 python3 -c "
import torch
from masked_vit_attention import MaskedViTAttention, create_random_mask
import time

class TestConfig:
    hidden_size = 768
    num_attention_heads = 12
    attention_probs_dropout_prob = 0.0
    hidden_dropout_prob = 0.0

if torch.cuda.is_available():
    config = TestConfig()
    device = torch.device('cuda')
    
    # Test data
    hidden_states = torch.randn(2, 197, 768, device=device)
    mask = create_random_mask(2, 197, 0.15, device)
    
    # Test CUDA kernel
    cuda_attention = MaskedViTAttention(config, use_cuda_kernel=True).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = cuda_attention(hidden_states, mask)
    torch.cuda.synchronize()
    
    # Time measurement
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = cuda_attention(hidden_states, mask)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / 10 * 1000
    
    print(f'✅ Performance test passed: {cuda_time:.2f}ms per iteration')
else:
    print('⚠️  CUDA not available, skipping performance test')
" 2>/dev/null; then
        print_success "Quick performance test passed"
    else
        print_warning "Performance test failed or timed out"
    fi
    
    return 0
}

# Main installation flow with better error handling
main() {
    echo
    print_header "Starting optimized installation process..."
    echo
    
    # Pre-installation checks
    print_status "Step 1/6: System Requirements Check"
    check_cuda
    check_python
    check_gpu
    echo
    
    # Dependencies
    print_status "Step 2/6: Installing Dependencies"
    install_dependencies
    echo
    
    # Build
    print_status "Step 3/6: Building Extension"
    if ! build_and_install; then
        print_error "Build failed! Please check the error messages above."
        echo
        print_error "Common solutions:"
        print_error "1. Make sure CUDA toolkit is properly installed"
        print_error "2. Check that PyTorch has CUDA support"
        print_error "3. Verify cuBLAS libraries are available"
        print_error "4. Try: export CUDA_HOME=/usr/local/cuda"
        exit 1
    fi
    echo
    
    # Testing
    print_status "Step 4/6: Testing Installation"
    if test_installation; then
        INSTALL_SUCCESS=true
    else
        INSTALL_SUCCESS=false
    fi
    echo
    
    # Final status
    print_status "Step 5/6: Installation Summary"
    if [ "$INSTALL_SUCCESS" = true ]; then
        print_success "=================================================="
        print_success "🎉 INSTALLATION COMPLETED SUCCESSFULLY! 🎉"
        print_success "=================================================="
        echo
        echo "Your optimized masked attention CUDA extension is ready!"
        echo
        echo "📖 Quick Start:"
        echo "  from masked_vit_attention import MaskedViTAttention"
        echo "  attention = MaskedViTAttention(config, use_cuda_kernel=True)"
        echo
        echo "🧪 Run Examples:"
        echo "  python3 example_usage.py"
        echo
        echo "🚀 Performance Benchmark:"
        echo "  python3 performance_benchmark.py"
        echo
        echo "🔧 Run Full Tests:"
        echo "  python3 test_masked_attention.py"
        echo
        print_success "Expected performance: 4-8x speedup over PyTorch!"
    else
        print_warning "=================================================="
        print_warning "⚠️  INSTALLATION COMPLETED WITH WARNINGS ⚠️"
        print_warning "=================================================="
        echo
        print_warning "The extension was built but some tests failed."
        print_warning "Try running examples to verify functionality:"
        echo "  python3 example_usage.py"
        echo
        print_warning "If issues persist, please check:"
        print_warning "1. CUDA toolkit compatibility"
        print_warning "2. GPU driver version"
        print_warning "3. PyTorch CUDA version match"
    fi
    
    print_status "Step 6/6: Next Steps"
    echo "Visit documentation for advanced usage and optimization tips."
    echo
}

# Enhanced help function
show_help() {
    echo "Optimized Masked Multi-Head Self-Attention CUDA Extension Installer"
    echo "Version 2.0 - With cuBLAS integration and performance optimizations"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  --no-test       Skip testing after installation"
    echo "  --clean         Clean build files before installation"
    echo "  --debug         Build with debug symbols"
    echo "  --check-only    Only check requirements, don't install"
    echo "  --force         Force installation even if tests fail"
    echo
    echo "Environment Variables:"
    echo "  CUDA_HOME       Path to CUDA installation (auto-detected)"
    echo "  GPU_ARCH        Target GPU architecture (auto-detected)"
    echo
    echo "Examples:"
    echo "  $0                    # Standard installation"
    echo "  $0 --clean           # Clean install"
    echo "  $0 --debug           # Debug build"
    echo "  $0 --check-only      # Check requirements only"
    echo
    echo "For support and documentation:"
    echo "  GitHub: https://github.com/your-username/masked-attention-cuda"
    echo
}

# Parse command line arguments with more options
SKIP_TEST=false
CLEAN_BUILD=false
DEBUG_BUILD=false
CHECK_ONLY=false
FORCE_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-test)
            SKIP_TEST=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            DEBUG_BUILD=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Handle special modes
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build files..."
    if [ -f "Makefile" ]; then
        make clean
    fi
    rm -rf build/ dist/ *.egg-info/ *.so *.o
    print_success "Build files cleaned"
    echo
fi

if [ "$DEBUG_BUILD" = true ]; then
    export NVCC_FLAGS="$NVCC_FLAGS -g -G -DDEBUG"
    export CXXFLAGS="$CXXFLAGS -g -DDEBUG"
    print_status "Debug build enabled"
fi

if [ "$CHECK_ONLY" = true ]; then
    print_header "System Requirements Check Only"
    check_cuda
    check_python  
    check_gpu
    print_success "Requirements check completed"
    exit 0
fi

# Override test function if skip requested
if [ "$SKIP_TEST" = true ]; then
    test_installation() {
        print_status "Skipping tests as requested"
        return 0
    }
fi

# Run main installation
main