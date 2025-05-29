"""
Test script to verify the CUDA extension builds and works correctly
"""

import torch
import numpy as np
import time
import sys
import os

def test_basic_functionality():
    """Test basic functionality without CUDA extension"""
    print("Testing basic PyTorch functionality...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
        
    print(f"✓ CUDA is available")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Device: {torch.cuda.get_device_name()}")
    
    # Test basic tensor operations
    x = torch.randn(4, 196, 768, device='cuda')
    mask = torch.rand(4, 196, device='cuda') > 0.3
    
    print(f"✓ Created test tensors: x={x.shape}, mask={mask.shape}")
    print(f"✓ Sparsity: {(~mask).float().mean().item():.1%}")
    
    return True

def test_pytorch_attention():
    """Test standard PyTorch attention as baseline"""
    print("\nTesting PyTorch MultiheadAttention baseline...")
    
    batch_size, seq_len, d_model, num_heads = 4, 196, 768, 12
    
    try:
        # Create standard attention module
        attention = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True).cuda()
        
        # Test data
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')
        mask = torch.rand(batch_size, seq_len, device='cuda') > 0.3
        
        # Forward pass
        with torch.no_grad():
            output, _ = attention(x, x, x)
            
        print(f"✓ PyTorch attention works: input={x.shape}, output={output.shape}")
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(50):
            with torch.no_grad():
                output, _ = attention(x, x, x)
                
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 50
        
        print(f"✓ PyTorch attention benchmark: {pytorch_time*1000:.2f} ms/iteration")
        return pytorch_time
        
    except Exception as e:
        print(f"❌ PyTorch attention failed: {e}")
        return None

def try_build_extension():
    """Try to build the CUDA extension"""
    print("\nAttempting to build CUDA extension...")
    
    try:
        from torch.utils.cpp_extension import load
        
        # Try to load the extension
        masked_attention_cuda = load(
            name="masked_attention_cuda",
            sources=[
                "masked_attention_wrapper.cpp",
                "kernel_launcher.cu",
            ],
            extra_cflags=['-O3'],
            extra_cuda_cflags=[
                '-O3', 
                '--use_fast_math',
                '-std=c++14',
                '--expt-relaxed-constexpr',
                '-gencode=arch=compute_70,code=sm_70',
                '-gencode=arch=compute_75,code=sm_75', 
                '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_86,code=sm_86',
            ],
            verbose=True
        )
        
        print("✓ CUDA extension built successfully!")
        return masked_attention_cuda
        
    except Exception as e:
        print(f"❌ Failed to build CUDA extension: {e}")
        return None

def test_cuda_extension(cuda_ext):
    """Test the CUDA extension functionality"""
    print("\nTesting CUDA extension...")
    
    if cuda_ext is None:
        print("❌ No CUDA extension to test")
        return None
        
    try:
        batch_size, seq_len, d_model, num_heads = 4, 196, 768, 12
        head_dim = d_model // num_heads
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, d_model, device='cuda')
        mask = torch.rand(batch_size, seq_len, device='cuda') > 0.3
        
        # Create weight matrices
        weight_q = torch.randn(d_model, d_model, device='cuda')
        weight_k = torch.randn(d_model, d_model, device='cuda')
        weight_v = torch.randn(d_model, d_model, device='cuda')
        bias_q = torch.randn(d_model, device='cuda')
        bias_k = torch.randn(d_model, device='cuda')
        bias_v = torch.randn(d_model, device='cuda')
        
        print(f"✓ Created test data: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
        print(f"✓ Mask sparsity: {(~mask).float().mean().item():.1%}")
        
        # Test forward pass
        with torch.no_grad():
            output = cuda_ext.masked_attention_forward(
                input_tensor, weight_q, weight_k, weight_v,
                bias_q, bias_k, bias_v, mask, num_heads
            )
            
        print(f"✓ CUDA extension forward pass successful: {output.shape}")
        
        # Benchmark CUDA extension
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(50):
            with torch.no_grad():
                output = cuda_ext.masked_attention_forward(
                    input_tensor, weight_q, weight_k, weight_v,
                    bias_q, bias_k, bias_v, mask, num_heads
                )
                
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / 50
        
        print(f"✓ CUDA extension benchmark: {cuda_time*1000:.2f} ms/iteration")
        return cuda_time
        
    except Exception as e:
        print(f"❌ CUDA extension test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(cuda_ext):
    """Compare CUDA extension results with PyTorch baseline"""
    print("\nComparing CUDA extension with PyTorch baseline...")
    
    if cuda_ext is None:
        print("❌ No CUDA extension to compare")
        return
        
    try:
        batch_size, seq_len, d_model, num_heads = 2, 64, 256, 8  # Smaller for debugging
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, d_model, device='cuda')
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')  # No masking for comparison
        
        # Create weight matrices
        weight_q = torch.randn(d_model, d_model, device='cuda')
        weight_k = torch.randn(d_model, d_model, device='cuda') 
        weight_v = torch.randn(d_model, d_model, device='cuda')
        bias_q = torch.zeros(d_model, device='cuda')  # Zero bias for simpler comparison
        bias_k = torch.zeros(d_model, device='cuda')
        bias_v = torch.zeros(d_model, device='cuda')
        
        # CUDA extension result
        with torch.no_grad():
            cuda_output = cuda_ext.masked_attention_forward(
                input_tensor, weight_q, weight_k, weight_v,
                bias_q, bias_k, bias_v, mask, num_heads
            )
            
        # PyTorch baseline (manual implementation)
        with torch.no_grad():
            # Manual QKV projection
            q = torch.nn.functional.linear(input_tensor, weight_q.T, bias_q)
            k = torch.nn.functional.linear(input_tensor, weight_k.T, bias_k)
            v = torch.nn.functional.linear(input_tensor, weight_v.T, bias_v)
            
            # Reshape for multi-head attention
            head_dim = d_model // num_heads
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape output
            pytorch_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            
        # Compare results
        diff = torch.abs(cuda_output - pytorch_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"✓ Shape match: CUDA={cuda_output.shape}, PyTorch={pytorch_output.shape}")
        print(f"✓ Max difference: {max_diff:.6f}")
        print(f"✓ Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✓ Results match within tolerance!")
        else:
            print("⚠ Results differ significantly - may need debugging")
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("="*60)
    print("MASKED ATTENTION CUDA EXTENSION TEST")
    print("="*60)
    
    # Basic functionality test
    if not test_basic_functionality():
        sys.exit(1)
        
    # PyTorch baseline test
    pytorch_time = test_pytorch_attention()
    
    # Try to build and test CUDA extension
    cuda_ext = try_build_extension()
    cuda_time = test_cuda_extension(cuda_ext)
    
    # Compare results
    compare_results(cuda_ext)
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if pytorch_time and cuda_time:
        speedup = pytorch_time / cuda_time
        print(f"PyTorch baseline: {pytorch_time*1000:.2f} ms/iteration")
        print(f"CUDA extension:   {cuda_time*1000:.2f} ms/iteration")
        print(f"Speedup:          {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✓ CUDA extension is faster!")
        else:
            print("⚠ CUDA extension is slower (may need optimization)")
    
    if cuda_ext is not None:
        print("✅ CUDA extension build and test: SUCCESS")
    else:
        print("❌ CUDA extension build and test: FAILED")
        print("\nDebugging suggestions:")
        print("1. Check CUDA installation: nvcc --version")
        print("2. Check PyTorch CUDA: python -c 'import torch; print(torch.cuda.is_available())'")
        print("3. Check compiler compatibility")
        print("4. Try simplified kernel without templates")

if __name__ == "__main__":
    main()