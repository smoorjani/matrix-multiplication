from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

'''
Use the following to build extension:
sudo python3 setup.py install
'''

setup(
    name='cublas_mm',
    ext_modules=[
        CUDAExtension('cublas_mm', [
            'cublas_mm.cpp',
            'cublas_mm_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })