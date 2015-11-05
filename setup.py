#!/usr/bin/env python

from distutils.core import setup
import os

"""Utility function to read the README file.
Used for the long_description.  It's nice, because now 1) we have a top level
README file and 2) it's easier to type in the README file than to put a raw
string in below ..."""

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='metaArray',
	version='0.9.9',
	description='meta-info container for numpy ndarray, with associate libraries.',
	author='Charley (Yichao) Fan',
	author_email='y.fan@warwick.ac.uk',
	url='http://warwickultrasound.co.uk/',
	packages=['metaArray'],
	long_description=read('README'),
     )
