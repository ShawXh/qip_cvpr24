from setuptools import setup, Extension
from torch.utils import cpp_extension



setup(name='torch_qip',
      version='0.1',
      ext_modules=[cpp_extension.CppExtension(
            'torch_qip', 
            ['./src/torch_qip.cpp'], 
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-lgomp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
