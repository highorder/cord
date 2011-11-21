#!/usr/bin/env python

from setuptools import setup
from cord.ext import CordExtension

ndarray_ext = CordExtension('cord.ndarray',
                            ['cord/ndarray.cpp'])
ndarray_header = 'cord/ndarray.hpp'

file_ext = CordExtension('cord.file',
                         ['cord/file.cpp'])

setup(name='cord',
      version='0.1',
      packages=['cord'],
      ext_modules=[ndarray_ext, file_ext],
      headers=[ndarray_header],
      )
