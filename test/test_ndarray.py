from numpy.random import standard_normal
A = standard_normal((3,4))

# Runtime numpy.ndarray <-> cord::ndarray conversion
import cord.ndarray

from test_ndarray import (print_ndim, print_shape, print_array, 
                          new_array)

print_ndim(A)
print_shape(A)
print_array(A)

A = A.T
print_ndim(A)
print_shape(A)
print_array(A)

B = new_array(8,6)
print B
