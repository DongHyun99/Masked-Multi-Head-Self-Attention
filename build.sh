#!/bin/bash

# Build script for Masked Attention CUDA Extension
# This script handles the compilation step by step to avoid complex issues

echo "=== Masked Attention CUDA Extension Build Script ==="

# Check prerequisites
echo "Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found. Please install CUDA toolkit."
    exit 1
fi

echo "✓ NVCC found: $(nvcc --version | head -n1)"

# Check Python and PyTorch
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch with CUDA not found"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -f *.so
rm -rf build/
rm -rf __pycache__/

# Method 1: Try simple_setup.py
echo "Method 1: Trying simple setup..."
python3 simple_setup.py build_ext --inplace 2>&1 | tee build_log.txt

if [ -f "masked_attention_cuda*.so" ]; then
    echo "✅ Method 1 successful!"
    python3 build.py
    exit 0
fi

# Method 2: Manual JIT compilation
echo "Method 1 failed. Trying Method 2: JIT compilation..."
cat > test_jit.py << 'EOF'
import torch
from torch.utils.cpp_extension import load
import os

print("Testing JIT compilation...")

try:
    # Simple JIT load
    masked_attention = load(
        name="masked_attention_simple",
        sources=[
            "simple_masked_attention_wrapper.cpp",
            "kernel_launcher.cu"
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math', 
            '--expt-relaxed-constexpr',
            '-gencode=arch=compute_70,code=sm_70'
        ],
        verbose=True
    )
    print("✅ JIT compilation successful!")
    
    # Quick test
    batch_size, seq_len, d_model, num_heads = 2, 32, 128, 4
    
    x = torch.randn(batch_size, seq_len, d_model, device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    
    # Create simple weights
    w_q = torch.randn(d_model, d_model, device='cuda')
    w_k = torch.randn(d_model, d_model, device='cuda')
    w_v = torch.randn(d_model, d_model, device='cuda')
    b_q = torch.zeros(d_model, device='cuda')
    b_k = torch.zeros(d_model, device='cuda')
    b_v = torch.zeros(d_model, device='cuda')
    
    with torch.no_grad():
        output = masked_attention.masked_attention_forward(
            x, w_q, w_k, w_v, b_q, b_k, b_v, mask, num_heads
        )
    
    print(f"✅ Test successful! Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ JIT compilation failed: {e}")
    exit(1)
EOF

python3 test_jit.py
if [ $? -eq 0 ]; then
    echo "✅ Method 2 successful!"
    exit 0
fi

# Method 3: Fallback to PyTorch-only implementation
echo "Both methods failed. Creating PyTorch fallback..."
cat > pytorch_fallback.py << 'EOF'
"""
Pure PyTorch implementation as fallback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # QKV projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply mask to QKV
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
        q = q * mask_expanded
        k = k * mask_expanded
        v = v * mask_expanded
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~mask_2d, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = output * mask.unsqueeze(-1)
        
        return self.out_proj(output)

# Test the fallback
if __name__ == "__main__":
    print("Testing PyTorch fallback implementation...")
    
    attention = MaskedMultiHeadAttention(768, 12).cuda()
    x = torch.randn(4, 196, 768, device='cuda')
    mask = torch.rand(4, 196, device='cuda') > 0.3
    
    with torch.no_grad():
        output = attention(x, mask)
    
    print(f"✅ PyTorch fallback successful! Output shape: {output.shape}")
    print(f"Sparsity: {(~mask).float().mean().item():.1%}")
EOF

python3 pytorch_fallback.py
if [ $? -eq 0 ]; then
    echo "✅ PyTorch fallback works!"
    echo "You can use this implementation while debugging CUDA issues."
else
    echo "❌ Even PyTorch fallback failed!"
fi

echo ""
echo "=== Build Summary ==="
echo "If CUDA compilation failed, here are debugging steps:"
echo "1. Check CUDA compatibility: nvidia-smi"
echo "2. Check compiler version: gcc --version"
echo "3. Check PyTorch CUDA version: python3 -c 'import torch; print(torch.version.cuda)'"
echo "4. Try reducing CUDA architectures in setup.py"
echo "5. Use the PyTorch fallback for development"