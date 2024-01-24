# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 2024 15:46

@author: samhill

This module contains the core functionality of the metaArray library -
defining the metaArray object.
"""

import numpy as np
import numpy.typing as npt
import typing

from os import linesep
from copy import deepcopy

from .misc import linearFunc, logFunc, expFunc


class UnitError(ArithmeticError):
    """
    Unit error will be generated if trying to operate on metaArrays of
    different unit descriptions.
    """
    pass


class DimError(ArithmeticError):
    """
    Dimension error will be generated if trying to operate on metaArrays
    of different number of dimensions.
    """
    pass


class NoOverlapError(IndexError):
    """
    NoOverlapError is raised when testing for overlaping regions in xyz space,
    but none found.
    """
    pass


class range_info(dict):
    """
    Range info dictionary.

    In the event of an update (through __setitem__) it will notify its host
    objects via the host.update_range() method, so the host can carry out any
    necessary updates accordingly.

    A register of host objects are kept in a dict (self.register), with the
    host object id [id(obj)] as the dict index, and its range update method as
    the value.
    e.g. self.register[id(obj)] = obj.update_range

    """
    def __init__(self, values: dict = None) -> None:
        dict.__init__(self)
        if values is not None:
            self.update(values)
        self.register = {}

    def __getitem__(self, key: str) -> typing.Any:
        return dict.__getitem__(self, key)

    def __setitem__(self, key: str, value: typing.Any) -> None:
        dict.__setitem__(self, key, value)
        register = self.register
        for i in register:
            try:
                register[i]()
            except KeyError:
                print("Host object with id: " + str(i) +
                      "can not be updated, removing from range_info register")
                self.dereg(register[i])

    def reg(self, obj: 'metaArray', method: typing.Callable = None) -> None:
        """
        Register host object, with an optional update method arguement.
        Host.update_range() will be used to notify any changes unless otherwise
        specified.
        """
        if method is None:
            method = obj.update_range

        self.register[id(obj)] = method

    def dereg(self, obj: 'metaArray') -> None:
        """
        Deregister host object
        """
        dict.__delitem__(self, id(obj))
        return

    def __deepcopy__(self, memo) -> 'range_info':
        """
        Deep copy of self dictionary contents.

        self.register will not be copied however.
        """

        # from copy import deepcopy
        return range_info(deepcopy(dict.copy(self)))


class metaArray:
    """
    Numpy array with meta data, currently only supports linear sampling.
    Ring buffer hehaviour (e.g. -ve index) is not supported, for it
    may cause confusion in xyz space. Slice stepping is not supported,
    use metaFunc.metaResample instead.

    object = metaArray(numpy.ndarray)

    object.data is the Numpy ndarray storage container
    object.info is the dict meta information container

    If the requested index or slice object is int, no index translation
    is performed
        object[int].data == object.data[int]

    If the requested index or slice object is float, then it is assumed to be
    in x-y-z space, and will be rounded to the nearest i-j-k index
    before obtaining the data.
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

    Unit definitions are default to None, i.e. undefined/Arb. Unitless
        quantities should be assigned empty string '' for unit description.

    """
    def __init__(self, data: npt.ArrayLike, dtype: np.dtype = None,
                 copy: bool = False, info: dict = None,
                 debug: bool = False) -> None:
        """
        data    Array-like data, will be convert to numpy.ndarray
        dtype   Array element data type to aid convertion to numpy.ndarray
        copy    Whether to make a copy of data when convert to numpy.ndarray
        info    Optional meta data, can be updated later via object methods
        debug   Debug level/flag)

        If given data is a metaArray instance, the meta info can be overwritten
        by passing an new info, otherwise it will build as usual.
        """
        # Convert the data into ndarray
        if isinstance(data, metaArray):
            if info is None:
                self_data = np.array(data.data, dtype=dtype, copy=copy)
                info = data.copy_info()
            else:
                self_data = np.array(data.data, dtype=dtype, copy=copy)
        else:
            self_data = np.array(data, dtype=dtype, copy=copy)

        # Get information about size and dimension of data
        ndim = self_data.ndim
        shape = self_data.shape
        self.debug = debug

        # Generate default metainfo
        if debug:
            print("*** Generating the default meta_info")

        # These are global attributes
        self_info = {'name': None,
                     'unit': None,
                     'label': None,
                     'resample': False}

        # Range is the corresponding xyz index
        if debug:
            print("*** Generating default range descriptions")

        # These are per-axis attributes
        self_info['range'] = range_info()
        # Beginning and ending coordinates
        self_info['range']['begin'] = list(np.zeros(ndim, dtype=float))
        self_info['range']['end'] = list(np.array(shape, dtype=float))
        # Unit and labels
        self_info['range']['unit'] = [None] * ndim
        self_info['range']['label'] = [None] * ndim
        # Lin / log scale
        self_info['range']['log'] = [False] * ndim
        # Data order (see numpy.fft.fftshift)
        self_info['range']['fft'] = [False] * ndim

        # If info parameter is given, update them accordingly
        if info is not None:
            if debug:
                print("*** meta_info was supplied")

            # Update values
            keys = list(self_info.keys())
            for key in keys:
                # Check if defining range (which is another dict)
                if key != 'range':
                    # if not range, can simply try to update value
                    try:
                        self_info[key] = info[key]
                    except KeyError:
                        # Value not defined, continuing
                        continue
                else:
                    # Iterate over 'range' dict
                    for key2, val in info['range'].items():
                        self_info['range'][key2] = val

        # Assemble the metaArray
        self.data = self_data
        self.info = self_info

        # Duplicate some of the common ndarray attributes
        self.ndim = ndim
        self.shape = shape
        self.ctypes = self_data.ctypes
        self.dtype = self_data.dtype
        self.itemsize = self_data.itemsize
        self.nbytes = self_data.nbytes
        self.size = self_data.size

        # Register self with the range_info object and update
        # conversion functions
        self_info['range'].reg(self)
        self.update_range()

    def __repr__(self) -> str:
        '''Test representation of object
        '''
        # Make a copy first, because some of the entries will be
        # destroyed along the way
        nfo = self.copy_info()

        # Produce the range (axis) descriptions.
        range_nfo = nfo['range']
        range_desc = '=' * 72 + linesep
        range_desc += "Axis   \t"
        range_desc += " Begin  \t"
        range_desc += " End    \t"
        range_desc += " Scale  \t"
        range_desc += " Order  \t"
        range_desc += " Unit   \t"
        range_desc += " Label  \t"
        range_desc += linesep

        for i in range(self.ndim):
            range_desc += str(i) + " \t "
            range_desc += f"{range_nfo['begin'][i]:0.2e} \t "
            range_desc += f"{range_nfo['end'][i]:0.2e} \t "

            scale = range_nfo['log'][i]
            if not scale:
                scale = "Linear"
            elif isinstance(scale, int) or isinstance(scale, float):
                scale = "log" + str(scale)
            elif scale:
                scale = "log10"
            range_desc += f"{scale:8s} \t "

            try:
                order = range_nfo['fft'][i]
                if not order:
                    order = "Normal"
            except KeyError:
                order = "Normal"

            range_desc += f"{order:8} \t "
            range_desc += f"{str(range_nfo['unit'][i]):8} \t "
            range_desc += f"{str(range_nfo['label'][i]):8} \t "
            range_desc += linesep

        if not self.debug:
            del nfo['range']

        # Produce some of the default descriptions.
        desc = "array(" + linesep + str(self.data) + ")" + linesep
        desc += '=' * 72 + linesep
        desc += "Meta Info: " + linesep
        name = nfo.pop('name')
        desc += f"\t['name'] Title of this metaArray: {name} {linesep}"

        unit = nfo.pop('unit')
        if unit is None:
            desc += "\t['unit'] Array element quantity unit is not defined."
        elif unit == '':
            desc += "\t['unit'] Array element quantity is unitless."
        else:
            desc += f"\t['unit'] Array element quantity unit is: {unit}"
        desc += linesep

        label = nfo.pop('label')
        desc += f"\t['label'] Array element quantity label is: {label}"
        desc += linesep

        resample = nfo.pop('resample')
        if resample:
            desc += "\t['resample'] Automatic resampling of this array is allowed." + linesep  # noqa: E501
        else:
            desc += "\t['resample'] Automatic resampling of this array is prohibited." + linesep  # noqa: E501

        # FFT attribute needs be apply per axis
        ###################################
        # try:
        #  fft = nfo.pop('fft')
        #    if fft == True:
        #      desc += "\t['fft'] Array data in frequency domain representation."  # noqa: E501
        #      if self.debug = True:
        #          desc += linesep + "\t\t(i.e. See numpy.fft.fftshift)"
        #      desc += linesep
        #  else:
        #      desc += "\t['fft'] Array data in normal representation." + linesep  # noqa: E501
        # except:
        #    pass
        ####################################

        # Produce the rest of the descriptions
        ext_desc = ''
        for field, value in sorted(nfo.items()):
            ext_desc += f"\t['{field}'] = {value} {linesep}"

        if ext_desc != '':
            desc += '=' * 72 + linesep
            desc += ext_desc

        # Append the range descriptions
        desc += range_desc

        return desc

    def __copy__(self) -> 'metaArray':
        """All copies are deep copies"""
        return self.copy()

    def __deepcopy__(self) -> 'metaArray':
        """All copies are deep copies"""
        return self.copy()

    def copy(self) -> 'metaArray':
        """
        Retrun a duplicate copy (deep copy) of self
        """
        return metaArray(self.data, info=self.copy_info(), copy=True)

    def copy_info(self) -> dict:
        """
        Return a duplicate copy of own info

        Some of the sequence data types require explicit copy operations to
        aviod just duplicating object pointers.
        """
        if self.debug:
            print("*** Duplicating meta info")

        return deepcopy(self.info)

    def update(self, info_dict: dict) -> None:
        """
        Imitate the dict.update() method

        Will override existing values, use with care!
        """
        self.info.update(info_dict)

    def update_range(self) -> None:
        """
        Check if range info has been updated, update converstion ijk to xyz
        space conversion functions accordingly if so.
        """
        if self.debug:
            print("### Range info has changed, updating.")

        self.i2x = self.gen_i2x()
        self.x2i = self.gen_x2i()

    def set_range(self, axis: int, field: str,
                  value: typing.Union[str, float]) -> None:
        """
        Method to set range meta info

        Example: self.set_range(0, 'unit', 's')
        """

        nfo_range = self.info['range']

        if nfo_range.has_key(field):
            nfo_range[field][axis] = value
        else:
            raise ValueError(f"Requested field ({field}) name do not exist")

        self.update_range()

    def get_range(self, axis: int, field: str) -> typing.Union[str, float]:
        """
        Method to get range meta info

        Example: self.set_range(0, 'unit')
        """
        nfo_range = self.info['range']

        if nfo_range.has_key(field):
            return nfo_range[field][axis]
        else:
            raise ValueError(f"Requested field ({field}) name do not exist")

    def get_axis(self, axis: int = 0) -> npt.NDArray:
        """
        Return a 1D numpy ndarray representing the discretized real space
        indexies of the given axis

        Example: time = self.get_axis(0)
        Returns, for instance, time axis
        """

        assert type(axis) is int, f"Axis is not an integer: {axis}"

        begin = self.get_range(axis, 'begin')
        end = self.get_range(axis, 'end')
        n = self.data.shape[axis]

        if self.get_range(axis, 'log'):
            return np.logspace(begin, end, n)
        else:
            return np.linspace(begin, end, n)

    def get_smp_rate(self, axis: int = 0) -> float:
        """
        Return the sampling rate of the given axis

        Example: self.get_smp_rate(0)
        """

        assert type(axis) is int, f"Axis is not an integer:{axis}"

        x0 = self.get_range(axis, 'begin')
        x1 = self.get_range(axis, 'end')
        n = self.data.shape[axis]

        return float(n) / abs(x1 - x0)

    def gen_x2i(self) -> list[typing.Callable[..., float]]:
        """
        Regenerate the xyz to ijk space convertion functions
        """
        shape = self.shape = self.data.shape
        x0 = self.info['range']['begin']
        x1 = self.info['range']['end']
        lg = self.info['range']['log']

        lst = []
        for i in range(self.ndim):
            if not lg[i]:
                # Log scale is not applied, everything is linear
                if x0[i] == x1[i]:
                    # Start stop at the same point
                    lst.append(lambda x: 0.)
                else:
                    lst.append(linearFunc(x0[i], 0, x1[i], shape[i]))
            elif isinstance(lg[i], int) or isinstance(lg[i], float):
                # Log scale applied, given log base.
                lst.append(logFunc(x0[i], 0, x1[i], shape[i], base=lg[i]))
            elif lg[i]:
                # Log scale applied, use default base
                lst.append(logFunc(x0[i], 0, x1[i], shape[i]))
            else:
                raise ValueError(f"Log scale descriptor can only be int,\
                                 float, True, False or None, given: {lg[i]}")
        return lst

    def gen_i2x(self) -> list[typing.Callable[..., float]]:
        """
        Regenerate the ijk to xyz space convertion functions
        """
        shape = self.shape = self.data.shape
        x0 = self.info['range']['begin']
        x1 = self.info['range']['end']
        lg = self.info['range']['log']

        if self.debug:
            print("*** starting xyz:", x0, "ending xyz:", x1,
                  "shape:", shape, "ndim:", self.ndim)

        lst = []
        for i in range(self.ndim):
            if not lg[i]:
                # Log scale is not applied, everything is linear
                if shape[i] == 1:
                    # Start stop at the same point
                    lst.append(lambda x: x0[i])
                else:
                    lst.append(linearFunc(0, x0[i], shape[i], x1[i]))
            elif isinstance(lg[i], (int, float)):
                # Log scale applied, given log base.
                lst.append(expFunc(x0[i], 0, x1[i], shape[i], base=lg[i]))
            elif lg[i]:
                # Log scale applied, use default base
                lst.append(expFunc(x0[i], 0, x1[i], shape[i]))
            else:
                raise ValueError("Log scale descriptor can only be int,\
                    float, True, False or None, given: " + str(lg[i]))
        return lst

    def min(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        Min value of the data array
        """
        return self.data.min(axis)

    def max(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        Max value of the data array
        """
        return self.data.max(axis)

    def argmin(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        argMin value of the data array in x-y-z space
        """
        # return self._i2x(self.data.argmin(axis))
        pass

    def argmax(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        argMax value of the data array in x-y-z space
        """
        # return self._i2x(self.data.argmax(axis))
        pass

    def ptp(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        Peak to peak value of the data array
        """
        return self.data.ptp(axis)

    def sum(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        Returns the sum of the data values along the given axis
        """
        return self.data.sum(axis)

    def log10(self) -> 'metaArray':
        """
        Take the log10 of values and returns new metaArray
        """
        self.data = np.log10(self.data)

        try:
            self['unit'] = "log10(" + self['unit'] + ')'
        except TypeError:
            self['unit'] = None

        try:
            self['label'] = "log10 of " + self['label']
        except TypeError:
            self['label'] = 'log10()'

        return self
