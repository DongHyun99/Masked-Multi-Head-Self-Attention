from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available")

if CUDA_HOME is None:
    raise RuntimeError("CUDA_HOME environment variable is not set")

# CUDA extension
cuda_extension = CUDAExtension(
    name='masked_attention_cuda',
    sources=[
        'csrc/masked_attention.cpp',
        'csrc/masked_attention_kernel.cu',
    ],
    include_dirs=[
        'csrc',
        os.path.join(CUDA_HOME, 'include'),
    ],
    library_dirs=[
        os.path.join(CUDA_HOME, 'lib64'),
    ],
    libraries=['cublas', 'cublasLt', 'cudart'],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode', 'arch=compute_70,code=sm_70',
            '-gencode', 'arch=compute_75,code=sm_75',
            '-gencode', 'arch=compute_80,code=sm_80',
            '-gencode', 'arch=compute_86,code=sm_86',
            '-gencode', 'arch=compute_89,code=sm_89',
            '-gencode', 'arch=compute_90,code=sm_90',
            '--ptxas-options=-v',
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
    }
)

setup(
    name='masked-vision-transformer',
    version='1.0.0',
    description='Optimized CUDA kernel for masked Vision Transformer attention',
    author='Masked ViT Developer',
    author_email='developer@example.com',
    packages=['masked_vit'],
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.40.0',
        'numpy>=1.21.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    zip_safe=False,
)