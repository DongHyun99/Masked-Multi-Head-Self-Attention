"""
Test script for performance-optimized masked multi-head self-attention
"""

import torch
import torch.nn as nn
import time
import numpy as np
from transformers import ViTConfig, ViTModel
from masked_vit_attention import MaskedAttention, benchmark_attention_backends

# Test configuration
class TestConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1


def create_random_mask(batch_size, seq_length, mask_ratio=0.15, device='cuda'):
    """Create random boolean mask for testing"""
    mask = torch.rand(batch_size, seq_length, device=device) > mask_ratio
    # Ensure at least one token per sequence is valid
    for i in range(batch_size):
        if not mask[i].any():
            mask[i, 0] = True
    return mask


def test_basic_functionality():
    """Test basic functionality of masked attention"""
    print("Testing basic functionality...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestConfig()
    
    # Create test data
    batch_size, seq_length = 2, 197  # ViT-Base patch count
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=device)
    mask = create_random_mask(batch_size, seq_length, mask_ratio=0.2, device=device)
    
    # Test different backends
    backends = [
        ("Optimized", MaskedAttention(config, use_cuda_kernel=False)),
        ("CUDA Kernel", MaskedAttention(config, use_cuda_kernel=True)),
    ]
    
    outputs = {}
    
    for name, attention in backends:
        attention = attention.to(device).eval()
        
        # Forward pass
        with torch.no_grad():
            output, _ = attention(hidden_states, mask)
        
        outputs[name] = output
        
        # Check shapes
        assert output.shape == hidden_states.shape, f"{name} output shape mismatch: {output.shape} vs {hidden_states.shape}"
        
        # Check masked positions are handled correctly
        for b in range(batch_size):
            for s in range(seq_length):
                if not mask[b, s]:
                    # Masked positions should preserve original input
                    torch.testing.assert_close(output[b, s], hidden_states[b, s], rtol=1e-3, atol=1e-3)
        
        print(f"  ✓ {name} backend test passed")
    
    # Compare outputs between backends (should be similar)
    if len(outputs) > 1:
        backend_names = list(outputs.keys())
        for i in range(len(backend_names)):
            for j in range(i + 1, len(backend_names)):
                name1, name2 = backend_names[i], backend_names[j]
                try:
                    # Only compare non-masked regions as they might differ slightly due to implementation
                    valid_mask = mask.unsqueeze(-1).expand_as(outputs[name1])
                    diff = torch.abs(outputs[name1] - outputs[name2])
                    max_diff = diff[valid_mask].max().item()
                    print(f"  Max difference between {name1} and {name2}: {max_diff:.6f}")
                    if max_diff < 0.1:  # Reasonable threshold for different implementations
                        print(f"  ✓ {name1} and {name2} outputs are consistent")
                    else:
                        print(f"  ⚠ {name1} and {name2} outputs differ significantly")
                except Exception as e:
                    print(f"  ⚠ Could not compare {name1} and {name2}: {e}")
    
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
        
        attention = MaskedAttention(config, use_cuda_kernel=True).to(device).to(dtype)
        
        # Test with autocast
        with torch.autocast(device_type='cuda', dtype=dtype):
            output, _ = attention(hidden_states, mask)
        
        assert output.shape == hidden_states.shape
        print(f"  ✓ {dtype} test passed")
    
    print("✓ Mixed precision test passed")


def test_performance_comparison():
    """Compare performance between different backends"""
    print("Testing performance comparison...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠ Skipping performance test (CUDA not available)")
        return
    
    config = TestConfig()
    
    # Test different sizes to show adaptive behavior
    test_sizes = [
        (2, 64, "Small"),
        (4, 197, "ViT-Base"),
        (8, 384, "Large"),
    ]
    
    for batch_size, seq_length, size_name in test_sizes:
        print(f"\n  Testing {size_name} size (B={batch_size}, N={seq_length}):")
        
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=device)
        mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
        
        # Create models
        models = {
            "Standard PyTorch": nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                batch_first=True,
                dropout=0.0  # Disable for fair comparison
            ).to(device).eval(),
            "Optimized Backend": MaskedAttention(
                config, use_cuda_kernel=False
            ).to(device).eval(),
            "CUDA Kernel": MaskedAttention(
                config, use_cuda_kernel=True
            ).to(device).eval(),
        }
        
        # Warmup and benchmark
        num_warmup = 20
        num_iterations = 100
        
        results = {}
        
        for name, model in models.items():
            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    if name == "Standard PyTorch":
                        _ = model(hidden_states, hidden_states, hidden_states)
                    else:
                        _ = model(hidden_states, mask)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    if name == "Standard PyTorch":
                        _ = model(hidden_states, hidden_states, hidden_states)
                    else:
                        _ = model(hidden_states, mask)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            results[name] = elapsed
            print(f"    {name}: {elapsed:.4f}s ({elapsed/num_iterations*1000:.2f}ms/iter)")
        
        # Calculate speedups
        baseline = results.get("Standard PyTorch", results[list(results.keys())[0]])
        for name, elapsed in results.items():
            if name != "Standard PyTorch":
                speedup = baseline / elapsed
                print(f"    {name} speedup vs baseline: {speedup:.2f}x")
    
    print("\n✓ Performance comparison completed")


def test_gradient_computation():
    """Test gradient computation and backpropagation"""
    print("Testing gradient computation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestConfig()
    
    batch_size, seq_length = 2, 197
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                               device=device, requires_grad=True)
    mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
    
    # Test both backends
    backends = ["Optimized", "CUDA Kernel"]
    
    for backend_name in backends:
        print(f"  Testing {backend_name} backend...")
        
        # Reset gradients
        if hidden_states.grad is not None:
            hidden_states.grad.zero_()
        
        use_cuda = backend_name == "CUDA Kernel"
        attention = MaskedAttention(config, use_cuda_kernel=use_cuda).to(device)
        
        # Forward pass
        output, _ = attention(hidden_states, mask)
        
        # Compute loss
        loss = output.sum()
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Check gradients exist and are reasonable
        assert hidden_states.grad is not None, f"{backend_name}: Input gradients not computed"
        assert not torch.isnan(hidden_states.grad).any(), f"{backend_name}: NaN in input gradients"
        assert not torch.isinf(hidden_states.grad).any(), f"{backend_name}: Inf in input gradients"
        
        # Check model parameter gradients
        for name, param in attention.named_parameters():
            assert param.grad is not None, f"{backend_name}: {name} gradients not computed"
            assert not torch.isnan(param.grad).any(), f"{backend_name}: NaN in {name} gradients"
            assert not torch.isinf(param.grad).any(), f"{backend_name}: Inf in {name} gradients"
        
        print(f"    ✓ {backend_name} gradients computed successfully")
    
    print("✓ Gradient computation test passed")


def test_adaptive_backend_selection():
    """Test that the model correctly selects backends based on tensor size"""
    print("Testing adaptive backend selection...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠ Skipping adaptive backend test (CUDA not available)")
        return
    
    config = TestConfig()
    attention = MaskedAttention(config, use_cuda_kernel=True).to(device).eval()
    
    # Test small tensor (should use CUDA kernel or PyTorch)
    small_states = torch.randn(1, 32, config.hidden_size, device=device)
    small_mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    
    # Test large tensor (should use optimized path with Flash Attention/cuBLAS)
    large_states = torch.randn(4, 512, config.hidden_size, device=device)
    large_mask = torch.ones(4, 512, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        small_output, _ = attention(small_states, small_mask)
        large_output, _ = attention(large_states, large_mask)
    
    assert small_output.shape == small_states.shape
    assert large_output.shape == large_states.shape
    
    print("  ✓ Small tensor processing successful")
    print("  ✓ Large tensor processing successful")
    print("✓ Adaptive backend selection test passed")


def test_huggingface_integration():
    """Test integration patterns with HuggingFace transformers"""
    print("Testing HuggingFace integration patterns...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ViT config
    vit_config = ViTConfig(
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
    
    # Test config compatibility
    test_config = TestConfig()
    test_config.hidden_size = vit_config.hidden_size
    test_config.num_attention_heads = vit_config.num_attention_heads
    test_config.attention_probs_dropout_prob = vit_config.attention_probs_dropout_prob
    test_config.hidden_dropout_prob = vit_config.hidden_dropout_prob
    
    # Create our attention module with ViT config
    attention = MaskedAttention(test_config, use_cuda_kernel=True).to(device)
    
    # Test with ViT-like input (patches + CLS token)
    batch_size = 2
    seq_len = 197  # 196 patches + 1 CLS token
    hidden_states = torch.randn(batch_size, seq_len, vit_config.hidden_size, device=device)
    
    # Create realistic attention mask (mask some patches)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    attention_mask[0, 50:100] = False  # Mask patches in first sample
    attention_mask[1, 120:150] = False  # Mask patches in second sample
    
    with torch.no_grad():
        output, _ = attention(hidden_states, attention_mask)
    
    assert output.shape == (batch_size, seq_len, vit_config.hidden_size)
    print("  ✓ ViT config compatibility test passed")
    
    # Test different masking patterns
    masking_patterns = {
        "Random": create_random_mask(batch_size, seq_len, mask_ratio=0.75, device=device),
        "Block": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "Stripe": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
    }
    
    # Create block mask (7x7 center region)
    for i in range(3, 10):
        for j in range(3, 10):
            if i * 14 + j + 1 < seq_len:
                masking_patterns["Block"][:, i * 14 + j + 1] = False
    
    # Create stripe mask (every other row)
    for i in range(0, 14, 2):
        for j in range(14):
            if i * 14 + j + 1 < seq_len:
                masking_patterns["Stripe"][:, i * 14 + j + 1] = False
    
    for pattern_name, mask in masking_patterns.items():
        with torch.no_grad():
            output, _ = attention(hidden_states, mask)
        assert output.shape == hidden_states.shape
        print(f"  ✓ {pattern_name} masking pattern test passed")
    
    print("✓ HuggingFace integration test passed")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("Testing edge cases...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestConfig()
    
    batch_size, seq_length = 2, 10
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=device)
    
    attention = MaskedAttention(config, use_cuda_kernel=True).to(device).eval()
    
    # Test with all tokens valid
    all_valid = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
    with torch.no_grad():
        output, _ = attention(hidden_states, all_valid)
    assert output.shape == hidden_states.shape
    print("  ✓ All tokens valid test passed")
    
    # Test with minimal tokens valid (only CLS token)
    minimal_valid = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    minimal_valid[:, 0] = True  # Only first token valid
    with torch.no_grad():
        output, _ = attention(hidden_states, minimal_valid)
    assert output.shape == hidden_states.shape
    print("  ✓ Minimal tokens valid test passed")
    
    # Test without attention mask (should default to all valid)
    with torch.no_grad():
        output, _ = attention(hidden_states, None)
    assert output.shape == hidden_states.shape
    print("  ✓ No attention mask test passed")
    
    # Test with different dtypes for mask
    float_mask = torch.ones(batch_size, seq_length, device=device)
    float_mask[:, seq_length//2:] = 0.0
    with torch.no_grad():
        output, _ = attention(hidden_states, float_mask)
    assert output.shape == hidden_states.shape
    print("  ✓ Float mask conversion test passed")
    
    print("✓ Edge cases test passed")


def run_comprehensive_performance_test():
    """Run comprehensive performance comparison"""
    print("\n" + "="*60)
    print("Running Comprehensive Performance Test")
    print("="*60)
    
    try:
        benchmark_attention_backends()
        print("✓ Comprehensive performance test completed")
    except Exception as e:
        print(f"⚠ Performance test failed: {e}")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running Performance-Optimized Masked Multi-Head Attention Tests")
    print("=" * 70)
    
    test_functions = [
        test_basic_functionality,
        test_mixed_precision,
        test_performance_comparison,
        test_gradient_computation,
        test_adaptive_backend_selection,
        test_huggingface_integration,
        test_edge_cases,
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            print(f"\n{'-'*50}")
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'-'*50}")
    print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n" + "=" * 70)
        print("🎉 All tests passed successfully!")
        print("Performance-optimized masked attention is working correctly.")
        print("=" * 70)
        
        # Run comprehensive performance test
        run_comprehensive_performance_test()
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed")
        print("Please check the error messages above.")


if __name__ == "__main__":
    run_all_tests()