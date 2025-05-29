#!/bin/bash

# Build and Install Script for Masked Attention CUDA Extension
# File: build_and_install.sh

echo "========================================="
echo "Building Masked Attention CUDA Extension"
echo "========================================="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: NVCC not found. Please install CUDA toolkit."
    exit 1
fi

# Check CUDA version
echo "CUDA Version:"
nvcc --version

# Check GPU architecture
echo -e "\nDetecting GPU architecture:"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} - Compute Capability: {props.major}.{props.minor}')
else:
    print('No CUDA devices found')
"

# Clean previous builds
echo -e "\nCleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.so" -delete

# Build the extension
echo -e "\nBuilding CUDA extension..."
python3 setup.py build_ext --inplace

# Install the extension
echo -e "\nInstalling extension..."
pip install -e .

# Test the installation
echo -e "\nTesting installation..."
python3 -c "
try:
    import masked_attention_cuda
    print('✓ CUDA extension loaded successfully')
    
    # Test basic functionality
    import torch
    if torch.cuda.is_available():
        print('✓ CUDA is available')
        device = torch.cuda.current_device()
        print(f'✓ Using GPU: {torch.cuda.get_device_name(device)}')
    else:
        print('⚠ CUDA not available, will use CPU fallback')
        
except ImportError as e:
    print(f'✗ Failed to import CUDA extension: {e}')
    print('Will fall back to PyTorch implementation')
"

echo -e "\n========================================="
echo "Build and Installation Complete!"
echo "========================================="

# Provide usage instructions
echo -e "\nUsage Example:"
echo "python3 example_usage.py"
echo ""
echo "To run the benchmark:"
echo "python3 -c \"from example_usage import benchmark_attention_layers; benchmark_attention_layers()\""