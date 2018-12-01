from distutils.core import setup, Extension
import numpy

modu = Extension('prepare',
                    sources=['prepare.cpp'],
                    extra_compile_args=['-std=c++11'],
                    include_dirs=[numpy.get_include()])

setup(name='p', 
      version='1.0', 
      ext_modules=[modu])
