import ast
import os
import sys
import shutil
import setuptools
from distutils.command.clean import clean as Clean

DISTNAME = 'bfast'
DESCRIPTION = 'A Python library for Breaks For Additive Season and Trend (BFAST) that resorts to parallel computing for accelerating the computations.'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Dmitry Serykh'
MAINTAINER_EMAIL = 'dmitry.serykh@gmail.com'
URL = 'http://bfast.readthedocs.io'
LICENSE = 'GNU GENERAL PUBLIC LICENSE'
DOWNLOAD_URL = 'https://github.com/diku-dk/bfast'


# use setuptools for certain commands
if len(set(('develop', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist_wininst', 'install_egg_info', 'build_sphinx',
           'egg_info', 'easy_install', 'upload', 'bdist_wheel',
           '--single-version-externally-managed',
            )).intersection(sys.argv)) > 0:
    extra_setuptools_args = dict(
        zip_safe=False,
        include_package_data=True,
    )
else:
    extra_setuptools_args = dict()


# define new clean command
class CleanCommand(Clean):
    description = "Removes build directories and compiled files in the source tree."

    def run(self):

        Clean.run(self)

        if os.path.exists('build'):
            shutil.rmtree('build')

        for dirpath, dirnames, filenames in os.walk('bfast'):
            for filename in filenames:
                if (filename.endswith('.so') or \
                    filename.endswith('.pyd') or \
                    filename.endswith('.dll') or \
                    filename.endswith('.pyc') or \
                    filename.endswith('_wrap.c') or \
                    filename.startswith('wrapper_') or \
                    filename.endswith('~')):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == '__pycache__' or dirname == 'build':
                    shutil.rmtree(os.path.join(dirpath, dirname))


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('bfast')

    return config


def get_version(module_path, identifier="__version__"):
    """
    Get the version of the package without an import.

    Uses abstract syntax trees to avoid importing, and to not
    rely on regex. See a similar approach using line splitting:
    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/ (option 1).

    Parameters
    ----------
    module_path : str
        The relative path from the setup.py file to the module
        containing the version, given as a path.
        E.g. "bfast/__init__.py".
    identifier : str, default "__version__"
        The name of the variable containing the version.

    Returns
    -------
    str
        The version identifier as set in the module.

    Raises
    ------
    AttributeError
        If the version identifier is not set in the module.
    """
    setup_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(setup_file_path)
    module_path = os.path.join(root_dir, module_path)

    with open(module_path) as rf:
        module = ast.parse(rf.read())

    for node in ast.walk(module):
        if isinstance(node, ast.Assign) and node.targets[0].id == identifier:
            version = node.value.value  # no break in case it is set multiple times

    try:
        return version
    except UnboundLocalError:
        raise AttributeError("Module {} has no attribute {}".format(repr(module_path), repr(identifier)))


VERSION = get_version("bfast/__init__.py")


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    download_url=DOWNLOAD_URL,
                    long_description=LONG_DESCRIPTION,
                    packages=setuptools.find_packages(),
                    install_requires=[
                        'numpy>=1.11.0',
                        'pandas>=1.0.0',
                        'pyopencl>=2018.2.5',
                        'scikit-learn>=0.20.3',
                        'scipy>=1.2.1',
                        'matplotlib>=2.2.2',
                        'wget>=3.2',
                        'cupy>=9.2.0'
                    ],
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.3',
                                 'Programming Language :: Python :: 3.4',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 ],
                    cmdclass={'clean': CleanCommand},
                    setup_requires=["numpy>=1.11.0"],
                    **extra_setuptools_args)

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):

        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION

    else:

        from numpy.distutils.core import setup
        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
