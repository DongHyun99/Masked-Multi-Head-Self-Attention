from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='masked_attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='masked_attention_cuda',
            sources=[
                'masked_attention_kernel.cu',
                'masked_attention_binding.cpp'
            ],
            include_dirs=[
                '/usr/local/cuda/include',
            ],
            libraries=['cublas', 'cudnn'],
            library_dirs=['/usr/local/cuda/lib64'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_70',  # Adjust based on your GPU architecture
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)