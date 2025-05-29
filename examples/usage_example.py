#!/usr/bin/env python3
"""
Usage examples for MaskedAttention with HuggingFace Vision Transformer.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masked_vit import MaskedAttention, create_random_mask, create_block_mask


def example_1_basic_usage():
    """Example 1: Basic usage of MaskedAttention."""
    print("📝 Example 1: Basic Usage")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Standard ViT-Base configuration
    batch_size = 8
    seq_len = 197  # 196 patches + 1 CLS token
    d_model = 768
    num_heads = 12
    
    # Create input tensor (simulating image patches)
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Create mask (75% masking like in MAE)
    mask = create_random_mask(batch_size, seq_len, mask_ratio=0.75, device=device)
    
    # Initialize MaskedAttention
    attention = MaskedAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1,
        use_cuda_kernel=True
    ).to(device)
    
    # Forward pass
    output, attention_weights = attention(input_tensor, mask=mask)
    
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Mask shape:   {mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Visible tokens: {mask.sum(dim=1).tolist()}")
    print("✅ Basic usage completed!\n")


def example_2_replace_huggingface_vit():
    """Example 2: Replace HuggingFace ViT attention."""
    print("📝 Example 2: Replace HuggingFace ViT Attention")
    print("-" * 50)
    
    try:
        from transformers import ViTModel, ViTConfig
        from masked_vit.masked_attention import replace_vit_attention
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create ViT model
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        
        model = ViTModel(config).to(device)
        print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Replace attention layers
        model = replace_vit_attention(model, use_cuda_kernel=True)
        print(f"Modified model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with dummy input
        batch_size = 4
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Create mask for some tokens
        seq_len = 197  # 196 patches + 1 CLS token
        mask = create_random_mask(batch_size, seq_len, mask_ratio=0.5, device=device)
        
        with torch.no_grad():
            # Note: This is a simplified example. In practice, you'd need to modify
            # the model's forward method to accept and use the mask
            outputs = model(pixel_values)
        
        print(f"✅ Model replacement successful!")
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        
    except ImportError:
        print("⚠️  transformers not available. Install with: pip install transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def example_3_custom_masking_patterns():
    """Example 3: Different masking patterns."""
    print("📝 Example 3: Custom Masking Patterns")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    seq_len = 196  # 14x14 patches (excluding CLS token for this example)
    d_model = 768
    num_heads = 12
    
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
    attention = MaskedAttention(d_model, num_heads).to(device)
    
    # Pattern 1: Random masking
    print("Pattern 1: Random masking (75%)")
    random_mask = create_random_mask(batch_size, seq_len, mask_ratio=0.75, device=device)
    output1, _ = attention(input_tensor, mask=random_mask)
    print(f"  Visible tokens: {random_mask.sum(dim=1).tolist()}")
    
    # Pattern 2: Block masking
    print("Pattern 2: Block masking")
    block_mask = create_block_mask(batch_size, seq_len, block_size=16, mask_ratio=0.5, device=device)
    output2, _ = attention(input_tensor, mask=block_mask)
    print(f"  Visible tokens: {block_mask.sum(dim=1).tolist()}")
    
    # Pattern 3: Keep only corners (simulating partial occlusion)
    print("Pattern 3: Corner masking")
    corner_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    # Assuming 14x14 spatial arrangement
    spatial_size = 14
    for i in range(batch_size):
        # Keep top-left corner
        corner_mask[i, :49] = True  # 7x7 corner
    output3, _ = attention(input_tensor, mask=corner_mask)
    print(f"  Visible tokens: {corner_mask.sum(dim=1).tolist()}")
    
    print("✅ Custom masking patterns completed!\n")


def example_4_memory_efficient_processing():
    """Example 4: Memory-efficient processing of large sequences."""
    print("📝 Example 4: Memory-Efficient Processing")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory-efficient example")
        return
    
    device = 'cuda'
    
    # Large sequence configuration
    batch_size = 2
    seq_len = 1024  # Large sequence
    d_model = 768
    num_heads = 12
    
    print(f"Processing large sequence: batch_size={batch_size}, seq_len={seq_len}")
    
    # Use mixed precision for memory efficiency
    with torch.cuda.amp.autocast():
        input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
        
        # High masking ratio to reduce computation
        mask = create_random_mask(batch_size, seq_len, mask_ratio=0.9, device=device)
        
        attention = MaskedAttention(d_model, num_heads).to(device).half()
        
        # Monitor memory usage
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Forward pass
        with torch.no_grad():
            output, _ = attention(input_tensor, mask=mask)
        
        memory_after = torch.cuda.memory_allocated() / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"✅ Processing completed!")
        print(f"Memory used: {memory_used:.1f} MB")
        print(f"Visible tokens: {mask.sum(dim=1).tolist()}")
        print(f"Output shape: {output.shape}")
    
    print()


def example_5_comparison_with_standard_attention():
    """Example 5: Performance comparison with standard attention."""
    print("📝 Example 5: Performance Comparison")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping performance comparison")
        return
    
    device = 'cuda'
    batch_size = 8
    seq_len = 197
    d_model = 768
    num_heads = 12
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    mask = create_random_mask(batch_size, seq_len, mask_ratio=0.75, device=device)
    
    # MaskedAttention with CUDA kernel
    masked_attention = MaskedAttention(d_model, num_heads, use_cuda_kernel=True).to(device).half()
    
    # Standard PyTorch MultiheadAttention
    standard_attention = nn.MultiheadAttention(
        d_model, num_heads, dropout=0.0, batch_first=True
    ).to(device).half()
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = masked_attention(input_tensor, mask=mask)
            _ = standard_attention(input_tensor, input_tensor, input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark MaskedAttention
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        with torch.no_grad():
            _ = masked_attention(input_tensor, mask=mask)
    end_event.record()
    torch.cuda.synchronize()
    
    masked_time = start_event.elapsed_time(end_event) / 50
    
    # Benchmark standard attention
    start_event.record()
    for _ in range(50):
        with torch.no_data():
            _ = standard_attention(input_tensor, input_tensor, input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    standard_time = start_event.elapsed_time(end_event) / 50
    
    print(f"MaskedAttention (75% masked): {masked_time:.2f} ms")
    print(f"Standard Attention:           {standard_time:.2f} ms")
    print(f"Speedup factor:               {standard_time/masked_time:.2f}x")
    print("✅ Performance comparison completed!\n")


def example_6_integration_with_mae():
    """Example 6: Integration with Masked Autoencoder (MAE) style training."""
    print("📝 Example 6: MAE-style Integration")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # MAE configuration
    batch_size = 4
    seq_len = 196  # 14x14 patches (no CLS token for encoder)
    d_model = 768
    num_heads = 12
    mask_ratio = 0.75  # MAE default
    
    # Simulate patch embeddings
    patch_embeddings = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Create random mask
    mask = create_random_mask(batch_size, seq_len, mask_ratio=mask_ratio, device=device)
    
    # Encoder with MaskedAttention (only processes visible patches)
    encoder_attention = MaskedAttention(d_model, num_heads).to(device)
    
    print(f"Original sequence length: {seq_len}")
    print(f"Masked tokens per sample: {(~mask).sum(dim=1).tolist()}")
    print(f"Visible tokens per sample: {mask.sum(dim=1).tolist()}")
    
    # Encoder forward pass (only on visible tokens)
    with torch.no_grad():
        encoded_features, _ = encoder_attention(patch_embeddings, mask=mask)
    
    # Extract only visible tokens for further processing
    visible_features = []
    for b in range(batch_size):
        visible_indices = mask[b].nonzero(as_tuple=True)[0]
        visible_features.append(encoded_features[b, visible_indices])
    
    print(f"Encoded visible features shapes: {[f.shape for f in visible_features]}")
    
    # For decoder, you would typically:
    # 1. Add mask tokens back
    # 2. Add positional embeddings
    # 3. Process with decoder attention (standard, not masked)
    
    print("✅ MAE-style integration example completed!\n")


def main():
    """Run all examples."""
    print("🚀 MaskedAttention Usage Examples")
    print("=" * 70)
    print()
    
    # Check setup
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    print()
    
    # Run examples
    example_1_basic_usage()
    example_2_replace_huggingface_vit()
    example_3_custom_masking_patterns()
    example_4_memory_efficient_processing()
    example_5_comparison_with_standard_attention()
    example_6_integration_with_mae()
    
    print("🎉 All examples completed!")
    print("\n💡 Tips for best performance:")
    print("  - Use CUDA when available")
    print("  - Use mixed precision (float16) for large models")
    print("  - Higher mask ratios = better performance gains")
    print("  - Batch multiple sequences together when possible")


if __name__ == "__main__":
    main()