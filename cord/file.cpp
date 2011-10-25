#include <boost/python.hpp>

namespace {
  void *convert_to_FILEptr(PyObject* obj) {
    return PyFile_Check(obj) ? PyFile_AsFile(obj) : 0;
  }
}

BOOST_PYTHON_MODULE(file) {
  using namespace boost::python;
  converter::registry::insert
    (convert_to_FILEptr,
     type_id<FILE>(),
     &converter::wrap_pytype<&PyFile_Type>::get_pytype);
}
