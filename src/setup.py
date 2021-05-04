from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

'''
Use the following to build extension:
python3 setup.py install
'''

setup(
    name='custom_mm',
    ext_modules=[
        CUDAExtension('custom_mm', [
            'custom_mm.cpp',
            'cublas_mm_kernel.cu',
            'cublas_bmm_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
