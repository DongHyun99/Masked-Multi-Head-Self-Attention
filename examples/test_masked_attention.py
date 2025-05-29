#!/usr/bin/env python3
"""
Test and example script for MaskedAttention.
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masked_vit import MaskedAttention, create_random_mask, benchmark_attention
from masked_vit.utils import test_correctness, run_comprehensive_tests, profile_memory_usage


def basic_functionality_test():
    """Test basic functionality of MaskedAttention."""
    print("🔍 Basic Functionality Test")
    print("-" * 40)
    
    # Configuration
    batch_size = 2
    seq_len = 197  # Standard ViT sequence length (196 patches + 1 CLS token)
    d_model = 768
    num_heads = 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
    
    # Create input data
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"Input shape: {input_tensor.shape}")
    
    # Create mask (75% masking ratio, typical for MAE)
    mask = create_random_mask(batch_size, seq_len, mask_ratio=0.75, device=device)
    num_visible = mask.sum(dim=1)
    print(f"Visible tokens per sample: {num_visible.tolist()}")
    
    # Initialize MaskedAttention
    attention = MaskedAttention(d_model, num_heads).to(device)
    print(f"Model parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output, attn_weights = attention(input_tensor, mask=mask, output_attentions=False)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Check masking is applied correctly
    mask_expanded = mask.unsqueeze(2).expand_as(output)
    masked_values = output[~mask_expanded]
    max_masked_value = torch.max(torch.abs(masked_values)).item()
    print(f"Max absolute value in masked positions: {max_masked_value:.2e}")
    
    # Check output statistics
    visible_output = output[mask_expanded]
    print(f"Visible output - mean: {visible_output.mean():.4f}, std: {visible_output.std():.4f}")
    
    print("✅ Basic functionality test completed!\n")


def huggingface_compatibility_test():
    """Test compatibility with HuggingFace transformers."""
    print("🤗 HuggingFace Compatibility Test")
    print("-" * 40)
    
    try:
        from transformers import ViTConfig
        from masked_vit.masked_attention import MaskedViTAttention
        
        # Create ViT config
        config = ViTConfig(
            hidden_size=768,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.1
        )
        
        # Initialize with config
        attention = MaskedViTAttention(config)
        print(f"✅ Successfully created MaskedViTAttention from ViTConfig")
        print(f"   d_model: {attention.d_model}")
        print(f"   num_heads: {attention.num_heads}")
        print(f"   dropout: {attention.dropout}")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attention = attention.to(device)
        
        input_tensor = torch.randn(1, 197, 768, device=device)
        mask = torch.ones(1, 197, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            output, _ = attention(input_tensor, mask=mask)
        
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        
    except ImportError:
        print("⚠️  transformers not available, skipping HuggingFace compatibility test")
    except Exception as e:
        print(f"❌ HuggingFace compatibility test failed: {e}")
    
    print()


def mixed_precision_test():
    """Test mixed precision support."""
    print("🔥 Mixed Precision Test")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping mixed precision test")
        return
    
    device = 'cuda'
    batch_size = 4
    seq_len = 197
    d_model = 768
    num_heads = 12
    
    # Test with different dtypes
    dtypes = [torch.float32, torch.float16]
    
    for dtype in dtypes:
        print(f"Testing with {dtype}")
        
        # Create input
        input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        mask = create_random_mask(batch_size, seq_len, mask_ratio=0.5, device=device)
        
        # Initialize model
        attention = MaskedAttention(d_model, num_heads).to(device).to(dtype)
        
        # Forward pass
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            output, _ = attention(input_tensor, mask=mask)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"  ✅ {dtype} - Time: {elapsed_time:.2f}ms, Output shape: {output.shape}")
    
    print()


def performance_benchmark():
    """Run performance benchmark."""
    print("🚀 Performance Benchmark")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping performance benchmark")
        return
    
    # Standard ViT-Base configuration
    results = benchmark_attention(
        d_model=768,
        num_heads=12,
        seq_len=197,
        batch_size=8,
        mask_ratio=0.75,
        num_iterations=50,
        warmup_iterations=5
    )
    
    print()


def stress_test():
    """Run stress test with large inputs."""
    print("💪 Stress Test")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping stress test")
        return
    
    device = 'cuda'
    
    # Large configuration
    configs = [
        {'batch_size': 1, 'seq_len': 1024, 'd_model': 768, 'num_heads': 12},
        {'batch_size': 32, 'seq_len': 197, 'd_model': 768, 'num_heads': 12},
        {'batch_size': 8, 'seq_len': 577, 'd_model': 1024, 'num_heads': 16},
    ]
    
    for i, config in enumerate(configs):
        print(f"Config {i+1}: {config}")
        
        try:
            # Create input
            input_tensor = torch.randn(
                config['batch_size'], 
                config['seq_len'], 
                config['d_model'], 
                device=device, 
                dtype=torch.float16
            )
            mask = create_random_mask(
                config['batch_size'], 
                config['seq_len'], 
                mask_ratio=0.75, 
                device=device
            )
            
            # Initialize model
            attention = MaskedAttention(
                config['d_model'], 
                config['num_heads']
            ).to(device).half()
            
            # Measure memory before
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Forward pass
            with torch.no_grad():
                output, _ = attention(input_tensor, mask=mask)
            
            memory_after = torch.cuda.memory_allocated() / 1024 / 1024
            memory_used = memory_after - memory_before
            
            print(f"  ✅ Success! Memory used: {memory_used:.1f} MB")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print()


def main():
    """Run all tests."""
    print("🧪 MaskedAttention Test Suite")
    print("=" * 50)
    print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
    else:
        print("⚠️  CUDA not available, some tests will be skipped")
    
    print()
    
    # Run tests
    try:
        basic_functionality_test()
        huggingface_compatibility_test()
        mixed_precision_test()
        
        # Correctness test
        print("🔬 Correctness Test")
        print("-" * 40)
        if torch.cuda.is_available():
            passed = test_correctness(device='cuda')
            if passed:
                print("✅ Correctness test PASSED!")
            else:
                print("❌ Correctness test FAILED!")
        else:
            print("⚠️  CUDA not available, skipping correctness test")
        print()
        
        performance_benchmark()
        stress_test()
        
        # Memory profiling
        if torch.cuda.is_available():
            print("📊 Memory Profiling")
            print("-" * 40)
            profile_memory_usage()
            print()
        
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()