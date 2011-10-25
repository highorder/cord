#!/usr/bin/env python

from numpy import get_include as numpy_get_include
from distutils.core import setup, Extension

ndarray_ext = Extension('cord.ndarray',
                        ['cord/ndarray.cpp'],
                        include_dirs=[numpy_get_include()],
                        libraries=['boost_python-mt-py27']
                        )
ndarray_header = 'cord/ndarray.hpp'

setup(name='cord',
      version='0.1',
      packages=['cord'],
      ext_modules=[ndarray_ext],
      headers=[ndarray_header],
      )
