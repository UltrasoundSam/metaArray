Python metaArray

Wrapper for combining numpy ndarrays with meta information

- [INSTALLATION](#installation)
- [DESCRIPTION](#description)

# INSTALLATION

===================
Requirements
===================

Before installing, make sure that you have installed all of the prerequiste dependancies:

	Python 		- (Tested on 2.6 and 2.7)							(https://www.python.org/)
	Numpy		- For data processing and scientific computing		(http://www.numpy.org/)
	Matplotlib	- For plotting										(http://matplotlib.org/)
	H5Py		- For stable and cross platform data storage		(http://www.h5py.org/)

Also, whilst not required for metaArray, scipy (http://www.scipy.org/) is a very useful
module for scientific computing and data processing.

===================
Install
===================

-----------------------
PIP - Recommended
-----------------------

Very easy to install - once downloaded and extracted, simply use pip to install:

	pip install /path/to/download/metaArray

If already installed, and you want to update to a newer version of metaArray:

	pip install --upgrade /path/to/download/metaArray

To uninstall:

	pip uninstall metaArray

-----------------------

Can also use the setup file to install or create built distribution

	python setup.py install

or 

	python setup.py dist (--format=rpm/wininst/etc)

----------------------

Or finally, for Window users, can download window installer exe from [https://warwickultrasound.co.uk/~sam/]


# DESCRIPTION

Wrapper for combining numpy ndarrays with meta information. Also has useful utilities;
for example, reading in data files from PZFlex, binary data from oscilloscopes, talking
to impedance analysers, etc. 

Module comes with demonstration function to illustrate metaArray usage. In python shell:

> from metaArray import example
>
> example.demo()

