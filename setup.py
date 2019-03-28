#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# adapted from https://github.com/kennethreitz/setup.py/blob/master/setup.py

# also change the TROVE CLASSIFIER if you change it
__license__ = "GPL"

import os
import sys

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'tf_crnn'
DESCRIPTION = 'TensorFlow Convolutional Recurrent Neural Network (CRNN)'
URL = 'https://github.com/cipri-tom/tf-crnn'
EMAIL = 'ciprian.tomoiaga@gmail.com'
AUTHOR = 'Ciprian Tomoiaga'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.1'

# What packages are required for this module to be executed?
REQUIRED = [
    # here should be tensorflow, but we can't decide for the user which
    # version (CPU/GPU) to install, so we force them below to install one
    # TF doesn't provide sane installation https://github.com/pypa/pip/issues/5887

    # also numpy, but having tensorflow means they have it already
]

EXTRAS = {
    # nothing yet
    # we could let our users decide whether they want us to install TF for them
    # by using these extras, like here [1] or here [2]:
    # [1] https://github.com/ragulpr/wtte-rnn/commit/8b289f1f1364adef33d2be03ff2fbfc63d15b1c8
    # [2] https://github.com/openai/baselines/pull/488/files
    #     https://github.com/fabito/baselines/blob/44af526f1018b3ab1ee19f138991fd5542dc6ef7/README.md#installation
}

# the user should have installed TensorFlow before
try:
    import tensorflow
except ImportError as import_ex:
    raise RuntimeError("""This package depends on a valid TensorFlow installation.
        You can install  it via `pip install tensorflow-gpu` if you have
        a CUDA-enabled GPU or with `pip install tensorflow` without
        GPU support.""") from import_ex


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # as the definition of an implicit namespace package is quite lenient (PEP 420),
    # you may need to define a few exclusions here:
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=__license__,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing',
    ],
)
