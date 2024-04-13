from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dlgn_train',
      ext_modules=[cpp_extension.CppExtension('dlgn_train', ['example-app.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})