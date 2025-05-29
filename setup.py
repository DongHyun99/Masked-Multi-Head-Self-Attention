"""
Setup script for Masked Multi-Head Attention CUDA extension
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUDA version for compatibility
cuda_version = torch.version.cuda

# Define CUDA extension
ext_modules = [
    CUDAExtension(
        name='masked_attention_cuda',
        sources=[
            'masked_attention_wrapper.cpp',
            'masked_attention_impl.cu',
        ],
        include_dirs=[
            # Add any additional include directories here
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-Xptxas=-v',
                '--expt-relaxed-constexpr',
                # Architecture-specific optimizations
                '-gencode=arch=compute_70,code=sm_70',  # V100
                '-gencode=arch=compute_75,code=sm_75',  # RTX 2080 Ti  
                '-gencode=arch=compute_80,code=sm_80',  # A100
                '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                '-gencode=arch=compute_87,code=sm_87',  # RTX 4080/4090
                '-gencode=arch=compute_89,code=sm_89',  # RTX 4090 (Ada)
                '-gencode=arch=compute_90,code=sm_90',  # H100
            ]
        },
        define_macros=[
            ('TORCH_EXTENSION_NAME', 'masked_attention_cuda'),
            ('TORCH_API_INCLUDE_EXTENSION_H', None),
        ]
    )
]

setup(
    name='masked_attention_cuda',
    version='1.0.0',
    description='Optimized CUDA kernels for masked multi-head attention',
    author='Your Name',
    author_email='your.email@example.com',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.12.0',
        'numpy',
    ],
    python_requires='>=3.8',
    zip_safe=False,
)