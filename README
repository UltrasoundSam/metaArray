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

Currently only supports linear sampling.
Ring buffer hehaviour (e.g. -ve index) is not supported, for it may cause confusion in xyz space.
Slice stepping is not supported, use metaFunc.metaResample instead.

object = metaArray(numpy.ndarray)

object.data is the Numpy ndarray storage container
object.info is the dict meta information container

If the requested index or slice object is int, no index translation is performed
    object[int].data == object.data[int]

If the requested index or slice object is float, then it is assumed to be in x-y-z space,
and will be rounded to the nearest i-j-k index before obtaining the data.
    object[float] === object[x,y,z]

The x-y-z space is defined in such a way that, if object.ndim == 3:
    object.info['range'] = {'begin':(x0,y0,z0), 'end':(x1,y1,z1)}

All mathmetical operations are x-y-z space aware, in a exclusion mode.
    A = metaArray(dataA)
    B = metaArray(dataB)
    c = A + B === A[x,y,z] + B[x,y,z]
    
    such that Cx,Cy,Cz will be the overlaping section between 
    Ax,Ay,Az and Bx,By,Bz. If they do not overlap, then c = None
    
Addition and subtraction requires both metaArray has the same unit 
    description string, else an UnitError exception will be raised.

Unit definitions are default to None, i.e. undefined/Arb. Unitless quantities
    should be assigned empty string '' for unit description.
    

Module comes with demonstration function to illustrate metaArray usage. In python shell:

>>> from metaArray import demo
>>>
>>> demo()

