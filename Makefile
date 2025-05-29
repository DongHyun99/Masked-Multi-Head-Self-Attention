# Makefile for Masked Vision Transformer CUDA extension

# Default target
all: build

# Build the extension
build:
	python setup.py build_ext --inplace

# Install the package
install:
	pip install -e .

# Clean build files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf csrc/*.o
	rm -rf masked_attention_cuda*.so
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Test the installation
test:
	python -c "import masked_vit; print('Installation successful!')"
	python examples/test_masked_attention.py

# Development setup
dev-install: clean build install test

# Full clean and rebuild
rebuild: clean build install

# Check CUDA setup
check-cuda:
	@echo "Checking CUDA setup..."
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
	@nvcc --version || echo "NVCC not found in PATH"

.PHONY: all build install clean test dev-install rebuild check-cuda