"""
Simplified setup script for Masked Multi-Head Attention CUDA extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This extension requires CUDA.")

# Define CUDA extension with simplified flags
ext_modules = [
    CUDAExtension(
        name='masked_attention_cuda',
        sources=[
            'masked_attention_wrapper.cpp',
            'masked_attention_kernel.cu',  # Use the fixed kernel
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-std=c++14',
                '--expt-relaxed-constexpr',
                # Use only stable architectures to avoid compatibility issues
                '-gencode=arch=compute_70,code=sm_70',  # V100
                '-gencode=arch=compute_75,code=sm_75',  # RTX 2080 Ti  
                '-gencode=arch=compute_80,code=sm_80',  # A100
                '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
            ]
        },
        define_macros=[
            ('TORCH_EXTENSION_NAME', 'masked_attention_cuda'),
        ]
    )
]

setup(
    name='masked_attention_cuda',
    version='1.0.0',
    description='Optimized CUDA kernels for masked multi-head attention',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)  # Disable ninja
    },
    install_requires=[
        'torch>=1.12.0',
        'numpy',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)