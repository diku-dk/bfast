.. -*- rst -*-

Installation
============

.. warning::

    The authors are not responsible for any implications that stem from the use of this software!

We recommend to use `virtualenv <https://pypi.python.org/pypi/virtualenv>`_ to install the package and all dependencies (see below).

Create a new virtual environment and activate it::

  $ virtualenv -p python3 myenv
  $ source myenv/bin/activate

The package is available on `PyPI <https://pypi.python.org/pypi>`_, but can also be installed directly from the source code.

Installation via PyPI
---------------------
  
To install the package via `PyPI <https://pypi.python.org/pypi>`_, type::

  $ pip install bfast

Installation From Sources
-------------------------

To install the package from the sources, first get the current stable release via::

  $ git clone https://github.com/gieseke/bfast.git

Subsequently, install the package locally via::

  $ cd bfast
  $ python setup.py install --user

or, globally for all users, via::

  $ sudo python setup.py install
  
In case you would like to extend the package, type::

  $ python setup.py develop

Google Colab Installation 
-------------------------

A simple way to run the massively-parallel implementation using powerful GPUs is to resort to `Google Colab <https://colab.research.google.com>`_. To install and run the code, you can proceed as follows: Start a new notebook and change the runtime to GPU (Runtime->Change runtime type). Afterwards, type in::

  $ !pip install bfast

Execute the cell (hit Shift+Return). You might have to restart the runtime by clicking on the button that appeared at the bottom for the cell (Restart Runtime). Afterwards, you can follow the instructions provided in :ref:`Getting Started`.

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
- Sphinx==2.2.0
- sphinx-bootstrap-theme==0.7.1

