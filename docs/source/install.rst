.. -*- rst -*-

Installation
============

.. warning::

    The authors are not responsible for any implications that stem from the use of this software!

Installation via PyPI
---------------------

The package is available on `PyPI <https://pypi.python.org/pypi>`_, but can also be installed directly from the source code. We recommend to use `virtualenv <https://pypi.python.org/pypi/virtualenv>`_ to install the package and all dependencies (see below). To install the package via `PyPI <https://pypi.python.org/pypi>`_, type::

  $ sudo pip install bfast

Installation From Sources
-------------------------

To install the package from the sources, first get the current stable release via::

  $ git clone https://github.com/gieseke/bfast.git

Subsequently, install the package locally via::

  $ cd bfast
  $ python setup.py install --user

or, globally for all users, via::

  $ sudo python setup.py build
  $ sudo python setup.py install

Dependencies
------------

The bfast package requires Python 3.*.`OpenCL <https://www.khronos.org/opencl>`_ (version 1.2 or higher) has to be installed correctly on the system. The installation of OpenCL depends on the particular system, see, e.g.,

- `Intel <https://software.intel.com/en-us/intel-opencl/download>`_
- `Nvidia <https://developer.nvidia.com/opencl>`_
- `AMD <http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/getting-started-with-opencl/>`_

We refer to Andreas Klöckner's `wiki <https://wiki.tiker.net/OpenCLHowTo>`_ page for an excellent description of the OpenCL installation process on Linux-based systems. OpenCL is installed on `macOS <https://developer.apple.com/opencl/>`_. For Windows, we refer to this `blog post <https://streamcomputing.eu/blog/2015-03-16/how-to-install-opencl-on-windows/>`_.

The bfast package depends on the following Python packages:

- numpy==1.16.3
- pandas==0.24.2
- pyopencl==2018.2.5
- scikit-learn==0.20.3
- scipy==1.2.1
- matplotlib==2.2.2
- wget==3.2

