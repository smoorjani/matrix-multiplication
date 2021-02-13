from setuptools import setup

setup(
    name='matmuls',
    version='1.0',
    description='Wrapper for matrix multiplications implemented in custom_mm.',
    author='Samraj Moorjani',
    author_email='samrajm2@illinois.edu',
    py_modules=['matmuls'],
    install_requires=['custom_mm', 'torch', 'numpy'],
    entry_points='''
        [console_scripts]
        matmuls=matmuls:matmuls
    ''',
)
