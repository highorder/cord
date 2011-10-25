#ifndef CORD_NDARRAY_HPP_INCLUDED
#define CORD_NDARRAY_HPP_INCLUDED

#include <Python.h>
#include <numpy/noprefix.h>
#include <stdexcept>

namespace cord {

/**
 * NumPy type code registration.
 */
template <typename T>
struct numpy_traits {
  static inline int get_type_code();
};

#define SET_NUMPY_TYPE_CODE(type, code)				\
  template<>							\
  inline int numpy_traits<type>::get_type_code()		\
  { return code; }

SET_NUMPY_TYPE_CODE(bool,         NPY_BOOL)
SET_NUMPY_TYPE_CODE(npy_float,    NPY_FLOAT)
SET_NUMPY_TYPE_CODE(npy_double,   NPY_DOUBLE)
SET_NUMPY_TYPE_CODE(npy_int,      NPY_INT)
SET_NUMPY_TYPE_CODE(npy_long,     NPY_LONG)
SET_NUMPY_TYPE_CODE(npy_longlong, NPY_LONGLONG)
#undef SET_NUMPY_TYPE_CODE

/**
 * Wrapper for NumPy's ndarray.
 *
 * A thin wrapper which provides array indexing to the PyArrayObject
 * type.
 *
 */
template <typename V>
class ndarray {
public:
  typedef V value_type;

  // Reference an existing Python array object.
  ndarray(PyObject *ptr) {
    if (ptr == NULL)
      throw std::runtime_error("ndarray: ptr cannot be NULL");
    if (PyArray_TYPE(ptr) != numpy_traits<V>::get_type_code())
      throw std::runtime_error("ndarray: dtype mismatch");
    m_ptr = (PyArrayObject*) ptr;
    PyArray_INCREF(m_ptr);
  }

  // Create a new Python array object.
  ndarray(int ndim, npy_intp *shape) {
    const int type_code = numpy_traits<V>::get_type_code();
    m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  }

  // ndarray() {
  //   const int ndim = 0;
  //   npy_intp shape[0] = {};
  //   const int type_code = numpy_traits<V>::get_type_code();
  //   m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  // }
  ndarray(npy_intp s0) {
    const int ndim = 1;
    npy_intp shape[ndim] = {s0};
    const int type_code = numpy_traits<V>::get_type_code();
    m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  }
  ndarray(npy_intp s0, npy_intp s1) {
    const int ndim = 2;
    npy_intp shape[ndim] = {s0, s1};
    const int type_code = numpy_traits<V>::get_type_code();
    m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  }
  ndarray(npy_intp s0, npy_intp s1, npy_intp s2) {
    const int ndim = 3;
    npy_intp shape[ndim] = {s0, s1, s2};
    const int type_code = numpy_traits<V>::get_type_code();
    m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  }
  ndarray(npy_intp s0, npy_intp s1, npy_intp s2, npy_intp s3) {
    const int ndim = 4;
    npy_intp shape[ndim] = {s0, s1, s2, s3};
    const int type_code = numpy_traits<V>::get_type_code();
    m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  }
  ndarray(npy_intp s0, npy_intp s1, npy_intp s2, npy_intp s3, npy_intp s4) {
    const int ndim = 5;
    npy_intp shape[ndim] = {s0, s1, s2, s3, s4};
    const int type_code = numpy_traits<V>::get_type_code();
    m_ptr = (PyArrayObject*) PyArray_SimpleNew(ndim, shape, type_code);
  }

  ~ndarray() {
    PyArray_XDECREF(m_ptr);
  }

  // Element access.
  V& operator()(npy_intp i0) const {
    return *((V*) (PyArray_BYTES(m_ptr)+offset(i0)));
  }
  V& operator()(npy_intp i0, npy_intp i1) const {
    return *((V*) (PyArray_BYTES(m_ptr)+offset(i0,i1)));
  }
  V& operator()(npy_intp i0, npy_intp i1, npy_intp i2) const {
    return *((V*) (PyArray_BYTES(m_ptr)+offset(i0,i1,i2)));
  }
  V& operator()(npy_intp i0, npy_intp i1, npy_intp i2, npy_intp i3) const {
    return *((V*) (PyArray_BYTES(m_ptr)+offset(i0,i1,i2,i3)));
  }
  V& operator()(npy_intp i0, npy_intp i1, npy_intp i2, npy_intp i3, npy_intp i4) const {
    return *((V*) (PyArray_BYTES(m_ptr)+offset(i0,i1,i2,i3,i4)));
  }

  // Array size access.
  int ndim() { return PyArray_NDIM(m_ptr); }
  npy_intp shape(int i) { return PyArray_DIM(m_ptr, i); }
  npy_intp strides(int i) { return PyArray_STRIDE(m_ptr, i)/sizeof(V); }

  // Raw access to Python pointer
  PyObject* ptr() { return (PyObject*) m_ptr; }

  // Copy
  ndarray<V> copy(char order = 'C') {
    if (order == 'C')
      return ndarray<V>(PyArray_NewCopy(m_ptr, NPY_CORDER));
    else if (order == 'F')
      return ndarray<V>(PyArray_NewCopy(m_ptr, NPY_FORTRANORDER));
    else
      throw std::runtime_error("ndarray:copy(): unknown order.");
  }

  // Flatten
  // Returns a 1-dim'l _copy_ of the array.
  ndarray<V> flatten(char order = 'C') {
    if (order == 'C')
      return ndarray<V>(PyArray_Flatten(m_ptr, NPY_CORDER));
    else if (order == 'F')
      return ndarray<V>(PyArray_Flatten(m_ptr, NPY_FORTRANORDER));
    else
      throw std::runtime_error("ndarray:flatten(): unknown order.");
  }

  bool is_carray() {
    return PyArray_ISCARRAY(m_ptr);
  }
  bool is_farray() {
    return PyArray_ISFARRAY(m_ptr);
  }

private:
  PyArrayObject *m_ptr;

  // Return the offset to a particular item. The returned offset is in BYTES,
  // not item location!
  inline npy_intp offset(npy_intp i0) const {
    return PyArray_STRIDE(m_ptr, 0)*i0;
  }
  inline npy_intp offset(npy_intp i0, npy_intp i1) const {
    return PyArray_STRIDE(m_ptr, 0)*i0 +
      PyArray_STRIDE(m_ptr,1)*i1;
  }
  inline npy_intp offset(npy_intp i0, npy_intp i1, npy_intp i2) const {
    return PyArray_STRIDE(m_ptr, 0)*i0 +
      PyArray_STRIDE(m_ptr,1)*i1 +
      PyArray_STRIDE(m_ptr,2)*i2;
  }
  inline npy_intp offset(npy_intp i0, npy_intp i1, npy_intp i2, npy_intp i3) const {
    return PyArray_STRIDE(m_ptr, 0)*i0 +
      PyArray_STRIDE(m_ptr,1)*i1 +
      PyArray_STRIDE(m_ptr,2)*i2 +
      PyArray_STRIDE(m_ptr,3)*i3;
  }
  inline npy_intp offset(npy_intp i0, npy_intp i1, npy_intp i2, npy_intp i3, npy_intp i4) const {
    return PyArray_STRIDE(m_ptr, 0)*i0 +
      PyArray_STRIDE(m_ptr,1)*i1 +
      PyArray_STRIDE(m_ptr,2)*i2 +
      PyArray_STRIDE(m_ptr,3)*i3 +
      PyArray_STRIDE(m_ptr,4)*i4;
  }
};

} // namespace cord

#endif
