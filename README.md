Python metaArray

Wrapper for combining numpy ndarrays with meta information

- [INSTALLATION](#installation)
- [DESCRIPTION](#description)

# INSTALLATION

### Requirements


The module has the following dependencies:

- [Python](https://www.python.org/) - (Tested on 3.11 and 3.12)
- [Numpy](http://www.numpy.org/) - For data processing and scientific computing
- [Matplotlib](http://matplotlib.org/)  - For plotting
- [H5Py](http://www.h5py.org/) - For stable and cross platform data storage
- [Scipy](http://www.scipy.org/) - For scientific computing
- [pySerial](https://pyserial.readthedocs.io/en/latest/pyserial.html) - For serial port communications

# Install

### PIP - Recommended

Very easy to install - once downloaded and extracted, simply use pip to install:

```
pip install /path/to/download/metaArray
```

If already installed, and you want to update to a newer version of metaArray:

```
pip install --upgrade /path/to/download/metaArray
```

To uninstall:

```
pip uninstall metaArray
```

-----------------------

Can also use the setup file to install or create built distribution

```
python setup.py install
```

or

```
python setup.py dist (--format=rpm/wininst/etc)
```

----------------------

Or finally, for Window users, can download window installer exe from [https://dwarfpalace.org/~sam/metaArray/]


# DESCRIPTION

Wrapper for combining numpy ndarrays with meta information. Also has useful utilities;
for example, reading in data files from PZFlex, binary data from oscilloscopes, talking
to impedance analysers, etc.

Module comes with demonstration function to illustrate metaArray usage. In python shell:

> from metaArray import demo
>
> demo()

![Tests](https://github.com/UltrasoundSam/metaArray/actions/workflows/tests.yml/badge.svg)
