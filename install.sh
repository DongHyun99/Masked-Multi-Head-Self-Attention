#!/bin/bash

# Masked Multi-Head Self-Attention CUDA Extension Installation Script
# Author: Your Name
# Date: 2024

set -e  # Exit on any error

echo "======================================"
echo "Masked Attention CUDA Extension Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_success "CUDA ${CUDA_VERSION} found"
        
        # Check if CUDA_HOME is set
        if [ -z "$CUDA_HOME" ]; then
            export CUDA_HOME=/usr/local/cuda
            print_warning "CUDA_HOME not set, using default: $CUDA_HOME"
        else
            print_success "CUDA_HOME: $CUDA_HOME"
        fi
    else
        print_error "CUDA not found! Please install CUDA toolkit first."
        exit 1
    fi
}

# Check Python and PyTorch
check_python() {
    print_status "Checking Python and PyTorch..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | sed 's/Python //')
    print_success "Python ${PYTHON_VERSION} found"
    
    # Check if PyTorch is installed
    if python3 -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        print_success "PyTorch ${TORCH_VERSION} found"
        
        # Check CUDA support in PyTorch
        if python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
            TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
            print_success "PyTorch CUDA ${TORCH_CUDA_VERSION} support enabled"
        else
            print_error "PyTorch CUDA support not available!"
            exit 1
        fi
    else
        print_error "PyTorch not found! Please install PyTorch first."
        exit 1
    fi
}

# Check GPU
check_gpu() {
    print_status "Checking GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
        print_success "GPU found:"
        echo "$GPU_INFO" | while IFS=, read -r name compute_cap; do
            echo "  - $name (Compute Capability: $compute_cap)"
            
            # Check if compute capability is sufficient (>= 7.0)
            if (( $(echo "$compute_cap >= 7.0" | bc -l) )); then
                print_success "  Compute capability $compute_cap is supported"
            else
                print_warning "  Compute capability $compute_cap may not be optimal (recommend >= 7.0)"
            fi
        done
    else
        print_warning "nvidia-smi not found, cannot check GPU details"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_status "Installing essential dependencies..."
        pip3 install torch torchvision transformers numpy pybind11
        print_success "Essential dependencies installed"
    fi
}

# Build and install
build_and_install() {
    print_status "Building CUDA extension..."
    
    # Try multiple build methods
    if [ -f "setup.py" ]; then
        print_status "Building with setup.py..."
        python3 setup.py build_ext --inplace
        python3 setup.py install
        print_success "Built and installed with setup.py"
    elif [ -f "Makefile" ]; then
        print_status "Building with Makefile..."
        make install
        print_success "Built and installed with Makefile"
    else
        print_error "No build script found (setup.py or Makefile)"
        exit 1
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test import
    if python3 -c "import masked_attention_cuda; print('CUDA extension imported successfully')" &> /dev/null; then
        print_success "CUDA extension import test passed"
    else
        print_error "CUDA extension import failed"
        return 1
    fi
    
    # Test Python module
    if python3 -c "from masked_vit_attention import MaskedViTAttention; print('Python module imported successfully')" &> /dev/null; then
        print_success "Python module import test passed"
    else
        print_error "Python module import failed"
        return 1
    fi
    
    # Run basic functionality test
    if [ -f "test_masked_attention.py" ]; then
        print_status "Running basic functionality test..."
        if python3 -c "from test_masked_attention import test_basic_functionality; test_basic_functionality()" &> /dev/null; then
            print_success "Basic functionality test passed"
        else
            print_warning "Basic functionality test failed (this might be OK for some environments)"
        fi
    fi
    
    return 0
}

# Main installation flow
main() {
    echo
    print_status "Starting installation process..."
    echo
    
    # System checks
    check_cuda
    check_python
    check_gpu
    echo
    
    # Installation
    install_dependencies
    echo
    
    build_and_install
    echo
    
    # Testing
    if test_installation; then
        echo
        print_success "=============================================="
        print_success "Installation completed successfully!"
        print_success "=============================================="
        echo
        echo "You can now use the masked attention module:"
        echo "  from masked_vit_attention import MaskedViTAttention"
        echo
        echo "Run examples:"
        echo "  python3 example_usage.py"
        echo
        echo "Run tests:"
        echo "  python3 test_masked_attention.py"
        echo
    else
        echo
        print_error "=============================================="
        print_error "Installation completed with warnings!"
        print_error "=============================================="
        echo
        print_warning "Some tests failed, but the extension might still work."
        print_warning "Try running the examples to verify functionality."
        echo
    fi
}

# Help function
show_help() {
    echo "Masked Multi-Head Self-Attention CUDA Extension Installer"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --no-test      Skip testing after installation"
    echo "  --clean        Clean build files before installation"
    echo
    echo "Environment Variables:"
    echo "  CUDA_HOME      Path to CUDA installation (default: /usr/local/cuda)"
    echo
    echo "Examples:"
    echo "  $0              # Standard installation"
    echo "  $0 --clean      # Clean install"
    echo "  $0 --no-test    # Skip tests"
    echo
}

# Parse command line arguments
SKIP_TEST=false
CLEAN_BUILD=false

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
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build files..."
    if [ -f "Makefile" ]; then
        make clean
    fi
    rm -rf build/ dist/ *.egg-info/ *.so *.o
    print_success "Build files cleaned"
    echo
fi

# Skip test if requested
if [ "$SKIP_TEST" = true ]; then
    test_installation() {
        print_status "Skipping tests as requested"
        return 0
    }
fi

# Run main installation
main