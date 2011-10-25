#include <boost/python.hpp>
#include "cord/ndarray.hpp"

typedef cord::ndarray<double> darray;

void print_ndim(darray A) {
  printf("%i\n", A.ndim());
}

void print_shape(darray A) {
  for (int i=0; i<A.ndim(); i++)
    printf("%li ", A.shape(i));
  printf("\n");
}

void print_array(darray A) {
  for (int i=0; i<A.shape(0); i++) {
    for (int j=0; j<A.shape(1); j++) {
      printf("%f ", A(i,j));
    }
    printf("\n");
  }
}

darray new_array(int s0, int s1) {
  darray out(s0, s1);

  for (int i0=0; i0<s0; i0++)
    for (int i1=0; i1<s1; i1++)
      out(i0,i1) = i0*i1;

  return out;
}

BOOST_PYTHON_MODULE(test_ndarray) {
  import_array(); // necessary?

  namespace bp = boost::python;

  bp::def("print_ndim", &print_ndim);
  bp::def("print_shape", &print_shape);
  bp::def("print_array", &print_array);
  bp::def("new_array", &new_array);
}
