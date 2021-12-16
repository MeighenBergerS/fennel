from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [Extension('generated', ["generated.pyx"])]

setup(
    ext_modules=cythonize(extensions)
)
