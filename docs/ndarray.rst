=========
 ndarray
=========

Getting Started
===============

The ndarray module comes in two parts.

1. cord/ndarray.hpp: A header only library providing a thin wrapper for
   NumPy array objects (PyArrayObject*) allowing direct access to array elements
   and a number of convenience methods for creating copies and checking the
   memory layout.
2. cord.ndarray: A python module which registers several Boost.Python type
   converters for seamlessly converting a numpy.ndarray object in Python
   to a cord::ndarray oject in C++.

Usage
=====

Write a C++ module similar to::

  #include <cassert>
  #include <cord/ndarray.hpp>

  typedef cord::ndarray<double> darray;

  void add_two(darray A) {
    assert(A.ndim() == 2);
    for (int i=0; i<A.shape(0); i++)
      for (int j=0; j<A.shape(1); j++)
        A(i,j) += 2;
  }

  BOOST_PYTHON_MODULE(demo) {
    namespace bp = boost::python;
    bp::def("add_two", &add_two);
  }

Then compile the module, and test::

  >>> import cord.ndarray
  >>> import demo
  >>> import numpy

  >>> A = numpy.random.randn(3,4)
  >>> print A

  >>> demo.add_two(A)
  >>> print A


