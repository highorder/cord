"""
Utilities for compiling Boost.Python and NumPy extension modules.
"""

from distutils.core import Extension

def add_numpy(kwargs):
    """
    Add NumPy requirements to the keyword arguments dictionary,
    to be used in creating a new Extension.
    """

    # Find NumPy include directory, without actually importing it.
    from imp import find_module
    from os import path
    np = path.join(find_module('numpy')[1], 'core', 'include')

    # Add to the list of include_dirs
    kwargs.setdefault('include_dirs', []).append(np)

def add_boost_python(kwargs):
    """
    Add Boost.Python requirements to the keyword arguments dictionary,
    to be used in creating a new Extension.
    """

    # @todo: Currently this works on Ubuntu 11.10 only.
    # Figure out some generality here for the future...
    kwargs.setdefault('libraries', []).append('boost_python-mt-py27')    

class CordExtension(Extension):
    def __init__(self, *args, **kwargs):
        add_numpy(kwargs)
        add_boost_python(kwargs)
        Extension.__init__(self, *args, **kwargs)
