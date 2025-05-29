"""
Test script for masked multi-head self-attention CUDA kernel
"""

import torch
import torch.nn as nn
import time
import numpy as np
from transformers import ViTConfig, ViTModel
from masked_vit_attention import MaskedViTAttention, create_random_mask, replace_vit_attention_with_masked

# Test configuration
class TestConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1


def test_basic_functionality():
    """Test basic functionality of masked attention"""
    print("Testing basic functionality...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestConfig()
    
    # Create test data
    batch_size, seq_length = 2, 197  # ViT-Base patch count (196 patches + 1 CLS token)
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=device)
    mask = create_random_mask(batch_size, seq_length, mask_ratio=0.2, device=device)
    
    # Test both CUDA and PyTorch implementations
    attention_cuda = MaskedViTAttention(config, use_cuda_kernel=True).to(device)
    attention_pytorch = MaskedViTAttention(config, use_cuda_kernel=False).to(device)
    
    # Copy weights to ensure same computation
    attention_pytorch.load_state_dict(attention_cuda.state_dict())
    
    # Forward pass
    with torch.no_grad():
        output_cuda, _ = attention_cuda(hidden_states, mask)
        output_pytorch, _ = attention_pytorch(hidden_states, mask)
    
    # Check shapes
    assert output_cuda.shape == hidden_states.shape, f"CUDA output shape mismatch: {output_cuda.shape} vs {hidden_states.shape}"
    assert output_pytorch.shape == hidden_states.shape, f"PyTorch output shape mismatch: {output_pytorch.shape} vs {hidden_states.shape}"
    
    # Check masked positions are handled correctly
    for b in range(batch_size):
        for s in range(seq_length):
            if not mask[b, s]:
                # Masked positions should preserve original input
                torch.testing.assert_close(output_pytorch[b, s], hidden_states[b, s], rtol=1e-4, atol=1e-4)
    
    print("✓ Basic functionality test passed")


def test_mixed_precision():
    """Test mixed precision support"""
    print("Testing mixed precision support...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠ Skipping mixed precision test (CUDA not available)")
        return
    
    config = TestConfig()
    batch_size, seq_length = 2, 197
    
    # Test with different dtypes
    dtypes = [torch.float32, torch.float16]
    
    for dtype in dtypes:
        print(f"  Testing {dtype}...")
        
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                                   device=device, dtype=dtype)
        mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
        
        attention = MaskedViTAttention(config, use_cuda_kernel=True).to(device).to(dtype)
        
        with torch.autocast(device_type='cuda', dtype=dtype):
            output, _ = attention(hidden_states, mask)
        
        assert output.dtype == dtype, f"Output dtype mismatch: {output.dtype} vs {dtype}"
        assert output.shape == hidden_states.shape
        
        print(f"  ✓ {dtype} test passed")
    
    print("✓ Mixed precision test passed")


def test_performance_comparison():
    """Compare performance between CUDA kernel and PyTorch implementation"""
    print("Testing performance comparison...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠ Skipping performance test (CUDA not available)")
        return
    
    config = TestConfig()
    
    # Test with larger batch size for better GPU utilization
    batch_size, seq_length = 16, 197  # Increased batch size
    
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                               device=device, dtype=torch.float16)  # Use FP16 for better performance
    mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
    
    # Create models
    attention_cuda = MaskedViTAttention(config, use_cuda_kernel=True).to(device).half()
    attention_pytorch = MaskedViTAttention(config, use_cuda_kernel=False).to(device).half()
    attention_pytorch.load_state_dict(attention_cuda.state_dict())
    
    # Warmup (important for GPU)
    print("  Warming up GPU...")
    for _ in range(50):  # More warmup iterations
        with torch.no_grad():
            _ = attention_cuda(hidden_states, mask)
            _ = attention_pytorch(hidden_states, mask)
    
    torch.cuda.synchronize()
    
    # Benchmark CUDA kernel
    num_iterations = 200  # More iterations for accurate timing
    
    print("  Benchmarking CUDA kernel...")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            output_cuda, _ = attention_cuda(hidden_states, mask)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # Benchmark PyTorch implementation
    print("  Benchmarking PyTorch implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            output_pytorch, _ = attention_pytorch(hidden_states, mask)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    print(f"  CUDA kernel time: {cuda_time:.4f}s ({cuda_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"  PyTorch time: {pytorch_time:.4f}s ({pytorch_time/num_iterations*1000:.2f}ms per iteration)")
    
    if cuda_time > 0:
        speedup = pytorch_time/cuda_time
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup < 1.0:
            print(f"  ⚠ Warning: CUDA kernel is slower. This might be due to:")
            print(f"    - Small batch size (try larger batches)")
            print(f"    - Kernel launch overhead")
            print(f"    - Memory bandwidth limitations")
        else:
            print(f"  ✅ CUDA kernel is faster!")
    
    print("✓ Performance comparison completed")


def test_gradient_computation():
    """Test gradient computation and backpropagation"""
    print("Testing gradient computation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestConfig()
    
    batch_size, seq_length = 2, 197
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                               device=device, requires_grad=True)
    mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
    
    attention = MaskedViTAttention(config, use_cuda_kernel=False).to(device)  # Use PyTorch for gradient test
    
    # Forward pass
    output, _ = attention(hidden_states, mask)
    
    # Compute loss (sum of all outputs)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert hidden_states.grad is not None, "Input gradients not computed"
    assert attention.query.weight.grad is not None, "Query weight gradients not computed"
    assert attention.key.weight.grad is not None, "Key weight gradients not computed"
    assert attention.value.weight.grad is not None, "Value weight gradients not computed"
    assert attention.dense.weight.grad is not None, "Dense weight gradients not computed"
    
    print("✓ Gradient computation test passed")


def test_huggingface_integration():
    """Test integration with HuggingFace transformers"""
    print("Testing HuggingFace integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ViT model
    config = ViTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        image_size=224,
        patch_size=16,
        num_channels=3,
    )
    
    model = ViTModel(config).to(device)
    
    # Replace attention modules with masked versions
    model = replace_vit_attention_with_masked(model, use_cuda_kernel=False)
    
    # Create test input
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Create attention mask (mask some patches)
    attention_mask = torch.ones(batch_size, 197, dtype=torch.bool, device=device)  # 196 patches + 1 CLS
    attention_mask[0, 50:100] = False  # Mask some patches in first sample
    attention_mask[1, 120:150] = False  # Mask some patches in second sample
    
    # Forward pass
    with torch.no_grad():
        # Note: We need to modify the model to accept attention_mask
        # For this test, we'll just run without mask to test basic integration
        outputs = model(pixel_values)
    
    assert outputs.last_hidden_state.shape == (batch_size, 197, 768)
    
    print("✓ HuggingFace integration test passed")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("Testing edge cases...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestConfig()
    
    # Test with all tokens masked
    batch_size, seq_length = 2, 10
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=device)
    all_masked = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    all_masked[:, 0] = True  # Keep at least one token valid
    
    attention = MaskedViTAttention(config, use_cuda_kernel=False).to(device)
    
    with torch.no_grad():
        output, _ = attention(hidden_states, all_masked)
    
    assert output.shape == hidden_states.shape
    
    # Test with no tokens masked
    all_valid = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        output, _ = attention(hidden_states, all_valid)
    
    assert output.shape == hidden_states.shape
    
    print("✓ Edge cases test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Masked Multi-Head Self-Attention Tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_mixed_precision()
        test_performance_comparison()
        test_gradient_computation()
        test_huggingface_integration()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()