"""
Detailed performance analysis and optimization guide
"""

import torch
import time
import numpy as np
from masked_vit_attention import MaskedViTAttention, create_random_mask

class TestConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.0  # Disable for fair comparison
        self.hidden_dropout_prob = 0.0


def profile_memory_bandwidth():
    """Profile memory bandwidth utilization"""
    print("=== Memory Bandwidth Analysis ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    config = TestConfig()
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_length = 197
    
    attention_cuda = MaskedViTAttention(config, use_cuda_kernel=True).to(device).half()
    attention_pytorch = MaskedViTAttention(config, use_cuda_kernel=False).to(device).half()
    attention_pytorch.load_state_dict(attention_cuda.state_dict())
    
    results = []
    
    for batch_size in batch_sizes:
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                                   device=device, dtype=torch.float16)
        mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = attention_cuda(hidden_states, mask)
                _ = attention_pytorch(hidden_states, mask)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        
        # CUDA timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = attention_cuda(hidden_states, mask)
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time
        
        # PyTorch timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = attention_pytorch(hidden_states, mask)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        # Calculate throughput
        total_elements = batch_size * seq_length * config.hidden_size
        cuda_throughput = (total_elements * num_iterations) / cuda_time / 1e9  # GFLOPS
        pytorch_throughput = (total_elements * num_iterations) / pytorch_time / 1e9
        
        speedup = pytorch_time / cuda_time
        
        results.append({
            'batch_size': batch_size,
            'cuda_time': cuda_time / num_iterations * 1000,  # ms
            'pytorch_time': pytorch_time / num_iterations * 1000,  # ms
            'speedup': speedup,
            'cuda_throughput': cuda_throughput,
            'pytorch_throughput': pytorch_throughput
        })
        
        print(f"Batch {batch_size:2d}: CUDA {cuda_time/num_iterations*1000:5.2f}ms, "
              f"PyTorch {pytorch_time/num_iterations*1000:5.2f}ms, "
              f"Speedup {speedup:4.2f}x")
    
    # Find optimal batch size
    best_speedup = max(results, key=lambda x: x['speedup'])
    print(f"\nBest speedup: {best_speedup['speedup']:.2f}x at batch size {best_speedup['batch_size']}")
    
    return results


def analyze_masking_efficiency():
    """Analyze performance with different masking ratios"""
    print("\n=== Masking Ratio Analysis ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    config = TestConfig()
    batch_size, seq_length = 16, 197
    
    attention_cuda = MaskedViTAttention(config, use_cuda_kernel=True).to(device).half()
    
    mask_ratios = [0.0, 0.15, 0.25, 0.5, 0.75, 0.9]
    
    for mask_ratio in mask_ratios:
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                                   device=device, dtype=torch.float16)
        mask = create_random_mask(batch_size, seq_length, mask_ratio=mask_ratio, device=device)
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = attention_cuda(hidden_states, mask)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = attention_cuda(hidden_states, mask)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        active_tokens = mask.sum().item()
        total_tokens = mask.numel()
        
        print(f"Mask ratio {mask_ratio:.2f} ({active_tokens}/{total_tokens} active): "
              f"{elapsed/num_iterations*1000:.2f}ms")


def compare_precision_performance():
    """Compare FP16 vs FP32 performance"""
    print("\n=== Precision Comparison ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    config = TestConfig()
    batch_size, seq_length = 16, 197
    
    dtypes = [torch.float32, torch.float16]
    
    for dtype in dtypes:
        print(f"\nTesting {dtype}...")
        
        attention = MaskedViTAttention(config, use_cuda_kernel=True).to(device).to(dtype)
        
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                                   device=device, dtype=dtype)
        mask = create_random_mask(batch_size, seq_length, mask_ratio=0.15, device=device)
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                if dtype == torch.float16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        _ = attention(hidden_states, mask)
                else:
                    _ = attention(hidden_states, mask)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                if dtype == torch.float16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        _ = attention(hidden_states, mask)
                else:
                    _ = attention(hidden_states, mask)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # Memory usage
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        torch.cuda.reset_peak_memory_stats()
        
        print(f"  Time: {elapsed/num_iterations*1000:.2f}ms")
        print(f"  Memory: {memory_used:.2f}GB")


def optimization_recommendations():
    """Provide optimization recommendations"""
    print("\n=== Optimization Recommendations ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - performance will be limited")
        return
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    compute_capability = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
    
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f}GB")
    print(f"Compute Capability: {compute_capability/10:.1f}")
    
    recommendations = []
    
    # Compute capability recommendations
    if compute_capability >= 80:  # Ampere or newer
        recommendations.append("✅ Use mixed precision (FP16) for optimal performance")
        recommendations.append("✅ Use larger batch sizes (16-32) for better GPU utilization")
    elif compute_capability >= 70:  # Volta
        recommendations.append("✅ Mixed precision supported, use FP16")
        recommendations.append("⚠ Consider moderate batch sizes (8-16)")
    else:
        recommendations.append("⚠ Older GPU - limited mixed precision support")
        recommendations.append("⚠ Use FP32 and smaller batch sizes")
    
    # Memory recommendations
    if gpu_memory >= 24:  # High-end GPUs
        recommendations.append("✅ Use large batch sizes and long sequences")
    elif gpu_memory >= 12:  # Mid-range GPUs
        recommendations.append("✅ Good for typical workloads")
    else:
        recommendations.append("⚠ Limited memory - use gradient checkpointing")
    
    # General recommendations
    recommendations.extend([
        "✅ Ensure sufficient warmup iterations (20-50)",
        "✅ Use contiguous tensors for optimal memory access",
        "✅ Enable cuDNN autotuner: torch.backends.cudnn.benchmark = True",
        "✅ Use appropriate masking ratios (15-25% for best performance)",
    ])
    
    for rec in recommendations:
        print(rec)


def run_comprehensive_analysis():
    """Run comprehensive performance analysis"""
    print("Masked Multi-Head Attention Performance Analysis")
    print("=" * 60)
    
    try:
        # Enable cuDNN benchmark for optimal performance
        torch.backends.cudnn.benchmark = True
        
        profile_memory_bandwidth()
        analyze_masking_efficiency()
        compare_precision_performance()
        optimization_recommendations()
        
        print("\n" + "=" * 60)  
        print("✅ Performance analysis completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_analysis()