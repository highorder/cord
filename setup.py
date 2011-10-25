#!/usr/bin/env python

from distutils.core import setup
from cord.ext import CordExtension

ndarray_ext = CordExtension('cord.ndarray',
                            ['cord/ndarray.cpp'])
ndarray_header = 'cord/ndarray.hpp'

setup(name='cord',
      version='0.1',
      packages=['cord'],
      ext_modules=[ndarray_ext],
      headers=[ndarray_header],
      )
