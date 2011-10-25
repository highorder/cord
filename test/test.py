import numpy as np
A = np.random.randn(3,4)
import ndarray
import test
test.print_ndim(A)
test.print_shape(A)
test.print_array(A)

A = A.T
test.print_ndim(A)
test.print_shape(A)
test.print_array(A)

