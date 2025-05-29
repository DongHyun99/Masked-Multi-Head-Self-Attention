"""
Advanced Performance Benchmark for Masked Multi-Head Self-Attention
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from masked_vit_attention import MaskedViTAttention, create_random_mask

class PerformanceBenchmark:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.results = defaultdict(list)
        
    def benchmark_attention(self, config, batch_size, seq_length, mask_ratio=0.15, 
                          num_iterations=100, warmup_iterations=10):
        """Comprehensive attention benchmarking"""
        
        # Create models
        cuda_attention = MaskedViTAttention(config, use_cuda_kernel=True).to(self.device)
        pytorch_attention = MaskedViTAttention(config, use_cuda_kernel=False).to(self.device)
        standard_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
            dropout=0.0
        ).to(self.device)
        
        # Ensure same weights
        pytorch_attention.load_state_dict(cuda_attention.state_dict())
        
        # Create test data
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, 
                                  device=self.device, dtype=torch.float32)
        mask = create_random_mask(batch_size, seq_length, mask_ratio, self.device)
        
        # Warmup
        print(f"Warmup ({warmup_iterations} iterations)...")
        self._warmup(cuda_attention, pytorch_attention, standard_attention, 
                    hidden_states, mask, warmup_iterations)
        
        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        
        # CUDA kernel
        torch.cuda.synchronize()
        cuda_times = []
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = cuda_attention(hidden_states, mask)
            end_event.record()
            
            torch.cuda.synchronize()
            cuda_times.append(start_event.elapsed_time(end_event))
        
        # PyTorch masked
        torch.cuda.synchronize()
        pytorch_times = []
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = pytorch_attention(hidden_states, mask)
            end_event.record()
            
            torch.cuda.synchronize()
            pytorch_times.append(start_event.elapsed_time(end_event))
        
        # Standard attention
        torch.cuda.synchronize()
        standard_times = []
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = standard_attention(hidden_states, hidden_states, hidden_states)
            end_event.record()
            
            torch.cuda.synchronize()
            standard_times.append(start_event.elapsed_time(end_event))
        
        # Statistics
        cuda_mean = np.mean(cuda_times)
        cuda_std = np.std(cuda_times)
        pytorch_mean = np.mean(pytorch_times)
        pytorch_std = np.std(pytorch_times)
        standard_mean = np.mean(standard_times)
        standard_std = np.std(standard_times)
        
        # Store results
        key = f"B{batch_size}_N{seq_length}_M{mask_ratio}"
        self.results[key] = {
            'cuda': {'mean': cuda_mean, 'std': cuda_std, 'times': cuda_times},
            'pytorch': {'mean': pytorch_mean, 'std': pytorch_std, 'times': pytorch_times},
            'standard': {'mean': standard_mean, 'std': standard_std, 'times': standard_times},
            'speedup_vs_pytorch': pytorch_mean / cuda_mean,
            'speedup_vs_standard': standard_mean / cuda_mean,
            'config': {
                'batch_size': batch_size,
                'seq_length': seq_length,
                'mask_ratio': mask_ratio,
                'hidden_size': config.hidden_size,
                'num_heads': config.num_attention_heads
            }
        }
        
        return self.results[key]
    
    def _warmup(self, cuda_att, pytorch_att, standard_att, hidden_states, mask, iterations):
        """Warmup GPU kernels"""
        for _ in range(iterations):
            with torch.no_grad():
                _ = cuda_att(hidden_states, mask)
                _ = pytorch_att(hidden_states, mask)
                _ = standard_att(hidden_states, hidden_states, hidden_states)
        torch.cuda.synchronize()
    
    def print_results(self, key):
        """Print benchmark results"""
        result = self.results[key]
        config = result['config']
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results: {key}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Sequence Length: {config['seq_length']}")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Num Heads: {config['num_heads']}")
        print(f"  Mask Ratio: {config['mask_ratio']:.1%}")
        print(f"\nTiming Results (ms per iteration):")
        print(f"  Standard Attention: {result['standard']['mean']:.3f} ± {result['standard']['std']:.3f}")
        print(f"  PyTorch Masked:     {result['pytorch']['mean']:.3f} ± {result['pytorch']['std']:.3f}")
        print(f"  CUDA Masked:        {result['cuda']['mean']:.3f} ± {result['cuda']['std']:.3f}")
        print(f"\nSpeedup:")
        print(f"  CUDA vs PyTorch:    {result['speedup_vs_pytorch']:.2f}x")
        print(f"  CUDA vs Standard:   {result['speedup_vs_standard']:.2f}x")
        
        # Performance classification
        if result['speedup_vs_pytorch'] > 2.0:
            print(f"  🚀 Excellent performance!")
        elif result['speedup_vs_pytorch'] > 1.5:
            print(f"  ✅ Good performance")
        elif result['speedup_vs_pytorch'] > 1.0:
            print(f"  ⚠️  Marginal improvement")
        else:
            print(f"  ❌ Performance regression")
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks across different configurations"""
        
        class BenchConfig:
            def __init__(self, hidden_size, num_heads):
                self.hidden_size = hidden_size
                self.num_attention_heads = num_heads
                self.attention_probs_dropout_prob = 0.0
                self.hidden_dropout_prob = 0.0
        
        # Test configurations
        configs = [
            ("ViT-Small", BenchConfig(384, 6)),
            ("ViT-Base", BenchConfig(768, 12)),
            ("ViT-Large", BenchConfig(1024, 16)),
        ]
        
        # Test scenarios
        scenarios = [
            (2, 197, 0.15),   # Small batch, standard ViT
            (4, 197, 0.25),   # Medium batch, higher masking
            (8, 197, 0.15),   # Large batch, standard masking
            (2, 577, 0.15),   # Larger image (384x384)
        ]
        
        print("Running Comprehensive Performance Benchmark")
        print("=" * 60)
        
        all_results = {}
        
        for config_name, config in configs:
            print(f"\nTesting {config_name} configuration...")
            
            for batch_size, seq_length, mask_ratio in scenarios:
                scenario_name = f"{config_name}_B{batch_size}_N{seq_length}_M{int(mask_ratio*100)}"
                
                try:
                    print(f"  Scenario: B={batch_size}, N={seq_length}, Mask={mask_ratio:.1%}")
                    
                    result = self.benchmark_attention(
                        config, batch_size, seq_length, mask_ratio,
                        num_iterations=50, warmup_iterations=10
                    )
                    
                    all_results[scenario_name] = result
                    
                    # Quick summary
                    speedup = result['speedup_vs_pytorch']
                    cuda_time = result['cuda']['mean']
                    pytorch_time = result['pytorch']['mean']
                    
                    print(f"    CUDA: {cuda_time:.2f}ms, PyTorch: {pytorch_time:.2f}ms, "
                          f"Speedup: {speedup:.2f}x")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        # Summary
        self.print_summary(all_results)
        return all_results
    
    def print_summary(self, results):
        """Print overall summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        speedups = [r['speedup_vs_pytorch'] for r in results.values()]
        cuda_times = [r['cuda']['mean'] for r in results.values()]
        pytorch_times = [r['pytorch']['mean'] for r in results.values()]
        
        print(f"Overall Statistics:")
        print(f"  Average Speedup: {np.mean(speedups):.2f}x")
        print(f"  Median Speedup:  {np.median(speedups):.2f}x")
        print(f"  Best Speedup:    {np.max(speedups):.2f}x")
        print(f"  Worst Speedup:   {np.min(speedups):.2f}x")
        
        print(f"\nPerformance Distribution:")
        excellent = sum(1 for s in speedups if s > 2.0)
        good = sum(1 for s in speedups if 1.5 < s <= 2.0)
        marginal = sum(1 for s in speedups if 1.0 < s <= 1.5)
        regression = sum(1 for s in speedups if s <= 1.0)
        
        total = len(speedups)
        print(f"  🚀 Excellent (>2x):     {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"  ✅ Good (1.5-2x):       {good}/{total} ({good/total*100:.1f}%)")
        print(f"  ⚠️  Marginal (1-1.5x):  {marginal}/{total} ({marginal/total*100:.1f}%)")
        print(f"  ❌ Regression (<1x):    {regression}/{total} ({regression/total*100:.1f}%)")
        
        if np.mean(speedups) > 1.5:
            print(f"\n🎉 Overall: CUDA kernel shows significant performance improvement!")
        elif np.mean(speedups) > 1.0:
            print(f"\n✅ Overall: CUDA kernel shows modest performance improvement")
        else:
            print(f"\n❌ Overall: CUDA kernel needs further optimization")

    def memory_benchmark(self, config, batch_size, seq_length):
        """Benchmark memory usage"""
        print(f"\nMemory Benchmark (B={batch_size}, N={seq_length}):")
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        cuda_attention = MaskedViTAttention(config, use_cuda_kernel=True).to(self.device)
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=self.device)
        mask = create_random_mask(batch_size, seq_length, 0.15, self.device)
        
        # Measure CUDA kernel memory
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = cuda_attention(hidden_states, mask)
        cuda_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Reset and measure PyTorch memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        pytorch_attention = MaskedViTAttention(config, use_cuda_kernel=False).to(self.device)
        with torch.no_grad():
            _ = pytorch_attention(hidden_states, mask)
        pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"  CUDA Memory:    {cuda_memory:.1f} MB")
        print(f"  PyTorch Memory: {pytorch_memory:.1f} MB")
        print(f"  Memory Ratio:   {pytorch_memory/cuda_memory:.2f}x")


def main():
    """Run performance benchmarks"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    # GPU info
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    benchmark = PerformanceBenchmark()
    
    # Quick test first
    class QuickConfig:
        hidden_size = 768
        num_attention_heads = 12
        attention_probs_dropout_prob = 0.0
        hidden_dropout_prob = 0.0
    
    print("\n" + "="*60)
    print("QUICK PERFORMANCE TEST")
    print("="*60)
    
    config = QuickConfig()
    result = benchmark.benchmark_attention(config, 4, 197, 0.15, num_iterations=50)
    benchmark.print_results("B4_N197_M0.15")
    
    # Memory test
    benchmark.memory_benchmark(config, 4, 197)
    
    # Ask for comprehensive test
    response = input("\nRun comprehensive benchmark? (y/n): ")
    if response.lower() == 'y':
        benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    main()