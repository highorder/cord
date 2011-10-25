#include <boost/python.hpp>
#include "ndarray.hpp"

namespace bp = boost::python;
using namespace cord;

namespace {

// Register ndarray->python converter.
template <typename V>
struct ndarray_to_python {
  ndarray_to_python() {
    bp::to_python_converter< ndarray<V>, ndarray_to_python<V> >();
  }
  static PyObject* convert(ndarray<V> obj) {
    return bp::incref(obj.ptr());
  }
};

// Register python->ndarray converter.
template <typename V>
struct ndarray_from_python {
  ndarray_from_python() {
    bp::converter::registry::push_back(&convertible, &construct,
				       bp::type_id< ndarray<V> >());
  }
  static void* convertible(PyObject *obj) {
    // Ensure that obj is a PyArray with correct type.
    if (!PyArray_Check(obj))
      return NULL;
    if (PyArray_TYPE(obj) != numpy_traits<V>::get_type_code())
      return NULL;
    if (!PyArray_ISBEHAVED(obj))
      return NULL;
    return obj;
  }
  static void construct(PyObject *obj, bp::converter::rvalue_from_python_stage1_data* data) {
    void *storage = ((bp::converter::rvalue_from_python_storage< ndarray<V> >*)data)->storage.bytes;
    new (storage) ndarray<V>(obj);
    data->convertible = storage;
  }
};

} // anonymous namespace

BOOST_PYTHON_MODULE(ndarray) {
  // Import NumPy C-API.
  import_array();

  // Register common types.
  ndarray_to_python<bool>();
  ndarray_to_python<int>();
  ndarray_to_python<long>();
  ndarray_to_python<double>();

  ndarray_from_python<bool>();
  ndarray_from_python<int>();
  ndarray_from_python<long>();
  ndarray_from_python<double>();
}
