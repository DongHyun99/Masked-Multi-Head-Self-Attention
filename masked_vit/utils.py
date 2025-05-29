"""
Utility functions for masked vision transformer.
"""

import torch
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
from .masked_attention import MaskedAttention


def create_random_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.75,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create random boolean mask for Vision Transformer.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_ratio: Ratio of tokens to mask (0.0 = no masking, 1.0 = all masked)
        device: Device to create tensor on
        
    Returns:
        Boolean mask tensor [batch_size, seq_len] where True = keep token
    """
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    if mask_ratio > 0:
        num_masked = int(seq_len * mask_ratio)
        for b in range(batch_size):
            # Randomly select tokens to mask
            masked_indices = torch.randperm(seq_len)[:num_masked]
            mask[b, masked_indices] = False
    
    return mask


def create_block_mask(
    batch_size: int,
    seq_len: int,
    block_size: int = 16,
    mask_ratio: float = 0.5,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create block-wise masking pattern (useful for image patches).
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        block_size: Size of each block to mask together
        mask_ratio: Ratio of blocks to mask
        device: Device to create tensor on
        
    Returns:
        Boolean mask tensor [batch_size, seq_len]
    """
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    if mask_ratio > 0:
        num_blocks = seq_len // block_size
        num_masked_blocks = int(num_blocks * mask_ratio)
        
        for b in range(batch_size):
            # Randomly select blocks to mask
            masked_blocks = torch.randperm(num_blocks)[:num_masked_blocks]
            for block_idx in masked_blocks:
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, seq_len)
                mask[b, start_idx:end_idx] = False
    
    return mask


def benchmark_attention(
    d_model: int = 768,
    num_heads: int = 12,
    seq_len: int = 197,
    batch_size: int = 8,
    mask_ratio: float = 0.75,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Benchmark MaskedAttention vs PyTorch native attention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size
        mask_ratio: Ratio of tokens to mask
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        device: Device to run benchmark on
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking MaskedAttention...")
    print(f"Config: d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}")
    print(f"Batch size={batch_size}, mask_ratio={mask_ratio}")
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    mask = create_random_mask(batch_size, seq_len, mask_ratio, device)
    
    # Initialize models
    masked_attn = MaskedAttention(d_model, num_heads, use_cuda_kernel=True).to(device).half()
    pytorch_attn = MaskedAttention(d_model, num_heads, use_cuda_kernel=False).to(device).half()
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = masked_attn(input_tensor, mask)
            _ = pytorch_attn(input_tensor, mask)
    
    torch.cuda.synchronize()
    
    # Benchmark CUDA kernel
    print("Benchmarking CUDA kernel...")
    cuda_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            cuda_output = masked_attn(input_tensor, mask)[0]
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        cuda_times.append(end_time - start_time)
    
    # Benchmark PyTorch fallback
    print("Benchmarking PyTorch fallback...")
    pytorch_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            pytorch_output = pytorch_attn(input_tensor, mask)[0]
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        pytorch_times.append(end_time - start_time)
    
    # Calculate statistics
    cuda_mean = np.mean(cuda_times) * 1000  # Convert to ms
    cuda_std = np.std(cuda_times) * 1000
    pytorch_mean = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000
    
    speedup = pytorch_mean / cuda_mean
    
    # Check numerical accuracy
    max_diff = torch.max(torch.abs(cuda_output - pytorch_output)).item()
    mean_diff = torch.mean(torch.abs(cuda_output - pytorch_output)).item()
    
    results = {
        'cuda_time_ms': {
            'mean': cuda_mean,
            'std': cuda_std,
            'min': min(cuda_times) * 1000,
            'max': max(cuda_times) * 1000
        },
        'pytorch_time_ms': {
            'mean': pytorch_mean,
            'std': pytorch_std,
            'min': min(pytorch_times) * 1000,
            'max': max(pytorch_times) * 1000
        },
        'speedup': speedup,
        'accuracy': {
            'max_diff': max_diff,
            'mean_diff': mean_diff
        },
        'config': {
            'd_model': d_model,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'mask_ratio': mask_ratio
        }
    }
    
    # Print results
    print(f"\nBenchmark Results:")
    print(f"CUDA Kernel: {cuda_mean:.2f} ± {cuda_std:.2f} ms")
    print(f"PyTorch:     {pytorch_mean:.2f} ± {pytorch_std:.2f} ms")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"Max diff:    {max_diff:.2e}")
    print(f"Mean diff:   {mean_diff:.2e}")
    
    return results


def plot_benchmark_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot benchmark results.
    
    Args:
        results: Results from benchmark_attention()
        save_path: Optional path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison
    methods = ['CUDA Kernel', 'PyTorch']
    times = [results['cuda_time_ms']['mean'], results['pytorch_time_ms']['mean']]
    errors = [results['cuda_time_ms']['std'], results['pytorch_time_ms']['std']]
    
    bars = ax1.bar(methods, times, yerr=errors, capsize=5, 
                   color=['#2E86C1', '#E74C3C'], alpha=0.8)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Attention Forward Pass Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add speedup annotation
    ax1.text(0.5, max(times) * 0.8, f'Speedup: {results["speedup"]:.2f}x', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Accuracy comparison
    accuracy_metrics = ['Max Difference', 'Mean Difference']
    accuracy_values = [results['accuracy']['max_diff'], results['accuracy']['mean_diff']]
    
    ax2.bar(accuracy_metrics, accuracy_values, color='#27AE60', alpha=0.8)
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Numerical Accuracy')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def profile_memory_usage(
    d_model: int = 768,
    num_heads: int = 12,
    seq_len: int = 197,
    batch_size: int = 8,
    mask_ratio: float = 0.75,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Profile memory usage of MaskedAttention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size
        mask_ratio: Ratio of tokens to mask
        device: Device to run profiling on
        
    Returns:
        Dictionary with memory usage statistics
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for memory profiling")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    mask = create_random_mask(batch_size, seq_len, mask_ratio, device)
    
    # Initialize model
    model = MaskedAttention(d_model, num_heads, use_cuda_kernel=True).to(device).half()
    
    # Measure memory before forward pass
    memory_before = torch.cuda.memory_allocated()
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor, mask)[0]
    
    # Measure memory after forward pass
    memory_after = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    memory_stats = {
        'memory_before_mb': memory_before / 1024 / 1024,
        'memory_after_mb': memory_after / 1024 / 1024,
        'peak_memory_mb': peak_memory / 1024 / 1024,
        'forward_memory_mb': (memory_after - memory_before) / 1024 / 1024,
        'input_size_mb': input_tensor.numel() * input_tensor.element_size() / 1024 / 1024,
        'output_size_mb': output.numel() * output.element_size() / 1024 / 1024
    }
    
    print(f"Memory Usage Profile:")
    print(f"Input size:      {memory_stats['input_size_mb']:.2f} MB")
    print(f"Output size:     {memory_stats['output_size_mb']:.2f} MB")
    print(f"Forward memory:  {memory_stats['forward_memory_mb']:.2f} MB")
    print(f"Peak memory:     {memory_stats['peak_memory_mb']:.2f} MB")
    
    return memory_stats


def test_correctness(
    d_model: int = 768,
    num_heads: int = 12,
    seq_len: int = 197,
    batch_size: int = 4,
    mask_ratio: float = 0.5,
    device: str = 'cuda',
    tolerance: float = 1e-3
) -> bool:
    """
    Test correctness of MaskedAttention against reference implementation.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads  
        seq_len: Sequence length
        batch_size: Batch size
        mask_ratio: Ratio of tokens to mask
        device: Device to run test on
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if test passes
    """
    print(f"Testing correctness with tolerance {tolerance}...")
    
    # Create test data
    torch.manual_seed(42)  # For reproducibility
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    mask = create_random_mask(batch_size, seq_len, mask_ratio, device)
    
    # Initialize models with same weights
    cuda_model = MaskedAttention(d_model, num_heads, use_cuda_kernel=True).to(device)
    pytorch_model = MaskedAttention(d_model, num_heads, use_cuda_kernel=False).to(device)
    
    # Copy weights to ensure identical models
    pytorch_model.load_state_dict(cuda_model.state_dict())
    
    # Forward pass
    with torch.no_grad():
        cuda_output, _ = cuda_model(input_tensor, mask)
        pytorch_output, _ = pytorch_model(input_tensor, mask)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(cuda_output - pytorch_output)).item()
    mean_diff = torch.mean(torch.abs(cuda_output - pytorch_output)).item()
    relative_diff = mean_diff / torch.mean(torch.abs(pytorch_output)).item()
    
    print(f"Max absolute difference:      {max_diff:.2e}")
    print(f"Mean absolute difference:     {mean_diff:.2e}")
    print(f"Mean relative difference:     {relative_diff:.2e}")
    
    # Check masked positions are zero
    mask_expanded = mask.unsqueeze(2).expand_as(cuda_output)
    masked_cuda = cuda_output[~mask_expanded]
    masked_pytorch = pytorch_output[~mask_expanded]
    
    masked_diff = torch.max(torch.abs(masked_cuda)).item()
    print(f"Max value in masked positions: {masked_diff:.2e}")
    
    # Test passes if differences are within tolerance
    passed = (max_diff < tolerance and masked_diff < tolerance)
    
    print(f"Correctness test: {'PASSED' if passed else 'FAILED'}")
    return passed


def generate_test_cases() -> list:
    """
    Generate comprehensive test cases for different configurations.
    
    Returns:
        List of test configurations
    """
    test_cases = [
        # Standard ViT configurations
        {'d_model': 768, 'num_heads': 12, 'seq_len': 197, 'batch_size': 1},
        {'d_model': 768, 'num_heads': 12, 'seq_len': 197, 'batch_size': 8},
        {'d_model': 768, 'num_heads': 12, 'seq_len': 197, 'batch_size': 16},
        
        # Different model sizes
        {'d_model': 384, 'num_heads': 6, 'seq_len': 197, 'batch_size': 8},
        {'d_model': 1024, 'num_heads': 16, 'seq_len': 197, 'batch_size': 8},
        
        # Different sequence lengths
        {'d_model': 768, 'num_heads': 12, 'seq_len': 50, 'batch_size': 8},
        {'d_model': 768, 'num_heads': 12, 'seq_len': 577, 'batch_size': 4},  # 24x24 patches
        
        # Edge cases
        {'d_model': 768, 'num_heads': 12, 'seq_len': 1, 'batch_size': 1},    # Single token
        {'d_model': 768, 'num_heads': 12, 'seq_len': 2, 'batch_size': 1},    # Two tokens
    ]
    
    return test_cases


def run_comprehensive_tests(device: str = 'cuda') -> Dict[str, bool]:
    """
    Run comprehensive tests on various configurations.
    
    Args:
        device: Device to run tests on
        
    Returns:
        Dictionary mapping test case names to pass/fail status
    """
    test_cases = generate_test_cases()
    results = {}
    
    print("Running comprehensive correctness tests...")
    print("=" * 50)
    
    for i, config in enumerate(test_cases):
        test_name = f"test_{i+1}_{config['d_model']}d_{config['num_heads']}h_{config['seq_len']}s_{config['batch_size']}b"
        print(f"\nTest {i+1}/{len(test_cases)}: {test_name}")
        
        try:
            passed = test_correctness(**config, device=device)
            results[test_name] = passed
        except Exception as e:
            print(f"Test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed_tests = sum(results.values())
    total_tests = len(results)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
        for test_name, passed in results.items():
            if not passed:
                print(f"  - {test_name}: FAILED")
    
    return results