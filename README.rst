=====
bfast
=====

The bfast package provides a highly-efficient parallel implementation for the `Breaks For Additive Season and Trend (BFASTmonitor) <http://bfast.r-forge.r-project.org>`_ proposed by Verbesselt et al. The implementation is based on `OpenCL <https://www.khronos.org/opencl>`_. 

=============
Documentation
=============

See the `documentation <http://bfast.readthedocs.org>`_ for details and examples.

============
Dependencies
============

The bfast package has been tested under Python 3.*. The required Python dependencies are:

- numpy==1.16.3
- pandas==0.24.2
- pyopencl==2018.2.5
- scikit-learn==0.20.3
- scipy==1.2.1
- matplotlib==2.2.2
- wget==3.2
- Sphinx==2.2.0
- sphinx-bootstrap-theme==0.7.1

Further, `OpenCL <https://www.khronos.org/opencl>`_ needs to be available.

==========
Quickstart
==========

The package can easily be installed via pip via::

  pip install bfast

To install the package from the sources, first get the current stable release via::

  git clone https://github.com/gieseke/bfast.git

Afterwards, on Linux systems, you can install the package locally for the current user via::

  python setup.py install --user

==========
Disclaimer
==========

The source code is published under the GNU General Public License (GPLv3). The authors are not responsible for any implications that stem from the use of this software.

