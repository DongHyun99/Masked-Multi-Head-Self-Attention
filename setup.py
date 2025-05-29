from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11
import torch

# CUDA extension
ext_modules = [
    CUDAExtension(
        name='masked_attention_cuda',
        sources=[
            'masked_attention_kernel.cu',
        ],
        include_dirs=[
            pybind11.get_cmake_dir(),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17', '-ffast-math'],
            'nvcc': [
                '-O3',
                '-std=c++17',
                '--use_fast_math',
                '--ptxas-options=-v',
                '--compiler-options', '-ffast-math',
                '-maxrregcount=64',
                '-gencode=arch=compute_70,code=sm_70',  # V100
                '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
                '-gencode=arch=compute_80,code=sm_80',  # A100
                '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
                '-gencode=arch=compute_90,code=sm_90',  # H100
            ]
        },
        language='c++',
    )
]

setup(
    name='masked_attention_cuda',
    version='1.0.0',
    description='Masked Multi-Head Self-Attention CUDA Extension for Vision Transformers',
    author='Your Name',
    author_email='your.email@example.com',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy',
    ],
)