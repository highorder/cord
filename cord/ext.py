"""
Utilities for compiling Boost.Python and NumPy extension modules.
"""

from setuptools import Extension

def add_numpy(kwargs):
    """
    Add NumPy requirements to the keyword arguments dictionary,
    to be used in creating a new Extension.
    """

    from numpy import get_include
    kwargs.setdefault('include_dirs', []).append(get_include())

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
