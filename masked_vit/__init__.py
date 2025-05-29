"""
Masked Vision Transformer - Optimized CUDA implementation for masked attention.

This package provides an optimized CUDA kernel implementation for Vision Transformer
attention computation with support for token masking, designed to replace HuggingFace's
standard ViT attention mechanism while maintaining compatibility.
"""

__version__ = "1.0.0"
__author__ = "Masked ViT Developer"

from .masked_attention import MaskedAttention
from .utils import create_random_mask, benchmark_attention

__all__ = [
    "MaskedAttention",
    "create_random_mask", 
    "benchmark_attention"
]

# Initialize CUDA backend on import
try:
    import masked_attention_cuda
    masked_attention_cuda.init_cublas()
    _cuda_available = True
except ImportError as e:
    _cuda_available = False
    import warnings
    warnings.warn(f"CUDA extension not available: {e}. Using fallback implementation.")

def is_cuda_available():
    """Check if CUDA extension is available."""
    return _cuda_available

def get_version():
    """Get package version."""
    return __version__