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
import h5py

from os import linesep
from copy import deepcopy

from .misc import linear_func, log_func, exp_func, filePath


# Alias for common type
numeric = typing.Union[int, float, npt.NDArray, 'metaArray']


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
                        value = deepcopy(info[key])

                        if isinstance(value, bytes):
                            # Make sure values are strings not bytes
                            value = value.decode()

                        self_info[key] = value
                    except KeyError:
                        # Value not defined, continuing
                        continue
                else:
                    # Iterate over 'range' dict
                    for key2, val in info['range'].items():
                        value = deepcopy(val)

                        # Make sure values are strings not bytes
                        if isinstance(value, bytes):
                            value = value.decode()
                        elif isinstance(value, list):
                            try:
                                value = [entry.decode() for entry
                                         in value]
                            except AttributeError:
                                pass

                        self_info['range'][key2] = value

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

    def __contains__(self, b: 'metaArray') -> bool:
        """
        Retrun true if self xyz space contains the b xyz in entirety
        """
        if not ((isinstance(self, metaArray) and isinstance(b, metaArray))):
            raise TypeError("Both operand must be metaArray objects")

        ndim = self.ndim
        if ndim != b.ndim:
            msg = "Both operand must have the same number of dimensions"
            raise DimError(msg)

        abegin = self.info['range']['begin']
        aend = self.info['range']['end']

        bbegin = b.info['range']['begin']
        bend = b.info['range']['end']

        # Check common grounds per dimension
        for i in range(ndim):
            # Starting point must be less or equal to be true
            if abegin[i] > bbegin[i]:
                return False

            # Ending point must be larger of equal to be true
            if aend[i] < bend[i]:
                return False

        return True

    def __delitem__(self,
                    key: typing.Union[str, int, float, tuple[float]]) -> None:
        raise ValueError("Cannot delete array elements")

    def __delslice__(self, b: typing.Union[int, float],
                     c: typing.Union[int, float]) -> None:
        raise ValueError("Cannot delete array elements")

    def __setitem__(self, key: typing.Union[int, float, str, tuple[float]],
                    value: typing.Any) -> None:
        """
        Modified method

        if key is int, return self.data[key] with corresponding meta info
        if key is float, convert key to ijk space, return self.data[ikey]
                         with corresponding meta info
        if key is string, return self.info['key']
        if key is tuple, convert xyz indexes to ijk space first, then return
                         self.data[ikey] with corresponding meta info
        """
        if self.debug:
            print("*** setitem", key, value)

        # Only have to set the meta info value
        if isinstance(key, str):
            self.info[key] = value
            # The is really for backward compatibility.
            # Should use set_range method instead
            if key == 'range':
                self.update_range()
            return

        proc_key = self.__proc_key

        # Could be a tuple of int, float or slice
        if isinstance(key, tuple):
            ijk_key = []
            # Walk through each axis
            for i in range(len(key)):
                ijk_key.append(proc_key(key[i], axis=i)[0])

            self.data[tuple(ijk_key)] = value        # Write to the array slice
        else:                                        # must be int or float
            self.data[proc_key(key)[0]] = value
        return

    def __setslice__(self, begin: tuple[int, float], end: tuple[int, float],
                     value: typing.Any) -> None:
        """
        Modified slice method
        Do not support stepping
        """
        if self.debug:
            print("*** setslice", begin, end, value)

        proc_key = self.__proc_key

        ibegin = proc_key(begin)[0]
        iend = proc_key(end)[0]

        # Get the indexes right
        if ibegin is None:
            ibegin = 0
        if iend is None:
            iend = self.shape[0]

        self.data[ibegin:iend] = value

    def __getstate__(self) -> dict[str, typing.Any]:
        odict = {}
        # odict = self.__dict__.copy()
        odict['info'] = self.copy_info()
        odict['data'] = self.data
        odict['dtype'] = self.dtype
        odict['debug'] = self.debug
        return odict

    def __setstate__(self, dict: dict) -> None:
        # fh = open(dict['file'])      # reopen file
        # count = dict['lineno']       # read from file...
        # while count:                 # until line count is restored
        #     fh.readline()
        #     count = count - 1
        # self.__dict__.update(dict)   # update attributes
        # self.fh = fh                 # save the file object

        self.__init__(dict['data'], info=dict['info'],
                      dtype=dict['dtype'], debug=dict['debug'])
        # dtype=None, copy=False, debug=False
        self.update_range()

    def __proc_key(self,
                   key: typing.Union[tuple[typing.Union[int, float]], slice],
                   axis: int = 0) -> typing.Union[tuple[slice, slice],
                                                  tuple[int, float]]:
        """
        Work out the ijk space key based on the given key data type.
        The key could be a tuple of int, float or slice
        """
        if self.debug:
            print("*** __proc_key:" + str(key))

        key_pair = self._key_pair
        if isinstance(key, slice):
            start = key_pair(key.start, axis=axis)
            stop = key_pair(key.stop, axis=axis)
            step = key_pair(key.step, axis=axis)
            slice_ijk = slice(start[0], stop[0], step[0])
            slice_xyz = slice(start[1], stop[1], step[1])
            return slice_ijk, slice_xyz
        else:
            keys = key_pair(key, axis=axis)
            return keys[0], keys[1]

    def _key_pair(self,
                  key: typing.Union[int, float],
                  axis: int = 0) -> tuple[int, float]:
        """
        Identify the given key space, return key pair in ijk and xyz space
        """
        x2i = self._x2i
        i2x = self._i2x

        if isinstance(key, float):
            xkey = key
            ikey = x2i(key, axis=axis)
        elif isinstance(key, int):
            xkey = i2x(key, axis=axis)
            ikey = key
        elif key is None:
            ikey = None
            xkey = None
        else:
            raise ValueError(f"Indexes must be int or float {str(key)} given.")

        if self.debug:
            print(f"*** _key_pair({str(key)}) => ikey: {str(ikey)}, xkey: {str(xkey)}")  # noqa: E501

        return ikey, xkey

    def __getitem__(self, key: typing.Union[str, int, float, tuple[float]]):
        """
        Modified method

        if key is string, return self.info['key']
        if key is int, return self.data[key] with corresponding meta info
        if key is float, convert key to ijk space, return self.data[ikey]
                         with corresponding meta info
        if key is tuple, convert xyz indexes to ijk space first, then return
                         self.data[ikey] with corresponding meta info
        """
        if self.debug:
            print("*** getitem:", key)

        # >>> a[1]
        # __getitem__(1)
        # >>> a[1,2]
        # __getitem__((1, 2))
        # >>> a[1:2]
        # __getslice__(1, 2)
        # >>> a[1:2:1]
        # __getitem__(slice(1, 2, 1))
        # >>> a[1,2::-1]
        # __getitem__((1, slice(2, None, -1)))

        # Only have to return the meta info value
        if isinstance(key, str):
            if str == 'info':
                return self.info
            else:
                try:
                    return self.info[key]
                except KeyError:
                    return None

        elif isinstance(key, slice):
            if (key.step != 1) and (key.step is not None):
                raise NotImplementedError("Non unity slice stepping is not supported. Use resample() instead." + str(key))  # noqa: E501
            return self.__getslice__(key.start, key.stop)

        proc_key = self.__proc_key

        # Init the new nfo dict
        nfo = self.copy_info()
        nfo_range = nfo['range']

        # Could be a tuple of int, float and slice
        if isinstance(key, tuple):
            # Init a new key tuple for the ijk version of the key
            ijk_key = []
            # Expected shape of the array
            ijk_shape = []

            # Walk through each axis
            for i, key_val in enumerate(key):
                keys = proc_key(key_val, axis=i)
                ikey = keys[0]
                # This will be ijk index pass onto ndarray
                ijk_key.append(ikey)

                if self.debug:
                    print("*** getitem ijk_key component:", ikey)

                # Slice is given, update metainfo
                if isinstance(ikey, slice):
                    if ikey.start is None:
                        istart = 0
                    else:
                        istart = ikey.start

                    if ikey.stop is None:
                        istop = self.shape[i]
                    else:
                        istop = ikey.stop

                    ijk_shape.append(istop-istart)
                    start = keys[1].start
                    stop = keys[1].stop
                    if start is not None:
                        nfo_range['begin'][i] = start
                    if stop is not None:
                        nfo_range['end'][i] = stop

                # Individual index is given, update meta info accordingly
                else:
                    ijk_shape.append(1)
                    nfo_range['begin'][i] = keys[1]
                    nfo_range['end'][i] = keys[1]

            # Tidy up the empty dimensions
            # Start with highest index, so deletion will not affect
            # later index counts
            for i in range(len(ijk_shape))[::-1]:
                if ijk_shape[i] == 1:
                    del ijk_shape[i]
                    del nfo_range['begin'][i]
                    del nfo_range['end'][i]
                    del nfo_range['unit'][i]
                    del nfo_range['label'][i]
            try:
                dat = self.data[tuple(ijk_key)]     # Obtain the array slice
            except IndexError:
                print("### ijk_key: " + str(ijk_key))
                print("### ijk_shape: " + str(ijk_shape))
                raise

        elif isinstance(key, int) or isinstance(key, float):
            # Selecting a element of the first axis.
            nfo_range['begin'] = nfo_range['begin'][1:]
            nfo_range['end'] = nfo_range['end'][1:]
            del nfo_range['unit'][0]
            dat = self.data[proc_key(key)[0]]
        else:
            typs = '<int|float|tuple|string>'
            m = f"Expecting {typs} key data type, {str(type(key))} given."
            raise IndexError(m)

        # If only one element remaining, no need to return meta info.
        if isinstance(dat, int) or isinstance(dat, float) \
           or isinstance(dat, complex):
            return dat
        else:
            return metaArray(dat, info=nfo)

    def __getslice__(self, begin: typing.Union[int, float],
                     end: typing.Union[int, float]) -> 'metaArray':
        """
        Modified slice method
        Do not support stepping
        """
        if self.debug:
            print("*** getslice: ", begin, end)

        # Init the new nfo dict
        nfo = self.copy_info()
        nfo_range = nfo['range']

        proc_key = self.__proc_key
        new_shape = list(self.shape)

        begin = proc_key(begin)
        end = proc_key(end)
        ibegin = begin[0]
        iend = end[0]
        xbegin = begin[1]
        xend = end[1]

        # Get the indexes right
        if ibegin is None:
            ibegin = 0
        if iend is None:
            iend = new_shape[0]
        new_shape[0] = iend - ibegin    # Here is the new shape

        # Update the meta info
        if xbegin is not None:
            nfo_range['begin'][0] = xbegin
        if xend is not None:
            nfo_range['end'][0] = xend

        # Obtain the array slice
        dat = self.data[ibegin:iend].reshape(new_shape)

        return metaArray(dat, info=nfo)

    def __len__(self) -> int:
        return len(self.data)

    def __abs__(self) -> 'metaArray':
        """
        Return a absolute value copy of self
        """
        ary = metaArray(abs(self.data), info=self.copy_info())
        try:
            ary['name'] = 'abs(' + ary['name'] + ')'
        except TypeError:
            pass
        return ary

    def truth(self) -> bool:
        """
        Returns True value
        """
        return True

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

    def copy_root_info(self) -> dict:
        """
        Return a duplicate copy of own info, except for those
        keys with a '.' char
        """
        info = self.copy_info()

        for field in info.keys():
            if field.find('.') != -1:
                del info[field]

        return info

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

        if field in nfo_range:
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

        if field in nfo_range:
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
                    lst.append(linear_func(x0[i], 0, x1[i], shape[i]))
            elif isinstance(lg[i], int) or isinstance(lg[i], float):
                # Log scale applied, given log base.
                lst.append(log_func(x0[i], 0, x1[i], shape[i], base=lg[i]))
            elif lg[i]:
                # Log scale applied, use default base
                lst.append(log_func(x0[i], 0, x1[i], shape[i]))
            else:
                raise ValueError(f"Log scale descriptor can only be int,\
                                 float, True, False or None, given: {lg[i]}")
        return lst

    def _x2i(self, key: float, axis: int = 0) -> int:
        """
        Convert xyz indexes to their ijk counter parts
        """
        return int(np.round(self.x2i[axis](key)))

    def _i2x(self, key: int, axis: int = 0) -> float:
        """
        Convert ijk indexes to their xyz counter parts
        """
        return self.i2x[axis](key)

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
                    lst.append(linear_func(0, x0[i], shape[i], x1[i]))
            elif isinstance(lg[i], (int, float)):
                # Log scale applied, given log base.
                lst.append(exp_func(x0[i], 0, x1[i], shape[i], base=lg[i]))
            elif lg[i]:
                # Log scale applied, use default base
                lst.append(exp_func(x0[i], 0, x1[i], shape[i]))
            else:
                raise ValueError("Log scale descriptor can only be int,\
                    float, True, False or None, given: " + str(lg[i]))
        return lst

    def overlap(self, b: 'metaArray') -> tuple[slice]:
        """
        Find the overlapping area/volum between two metaArrays (in xyz space)

        The output boundaries are always in float, because this is the way
        metaArray recognise it as xyz space.

        The output boundaries can be direcly apply to __getitem__ or __slice__.
        i.e.  a[a.overlap(b)] is valid.

        """

        if not ((isinstance(self, metaArray) and isinstance(b, metaArray))):
            raise TypeError("Both operand must be metaArray objects")

        ndim = self.ndim

        if ndim != b.ndim:
            msg = "Both operand must have the same number of dimensions"
            raise DimError(msg)

        abegin = self.info['range']['begin']
        aend = self.info['range']['end']

        bbegin = b.info['range']['begin']
        bend = b.info['range']['end']

        output = []

        # Check common grounds per dimension
        for i in range(ndim):
            # take the highest starting point
            begin = max(float(abegin[i]), float(bbegin[i]))

            # take the lowest ending point
            end = min(float(aend[i]), float(bend[i]))

            # There must be at least some overlap in all dimensions
            if begin > end:
                raise NoOverlapError

            output.append(slice(begin, end))

        return tuple(output)

    def to_h5(self, dest: str) -> None:
        """
        Write the metAry into given file path in the HDF5 format

        dest is interpret as the destination file path.
        """
        # Writing file to path
        path = filePath(dest)

        if not path.write:
            raise ValueError("Unable to write to: " + str(path.full))

        with h5py.File(path.full, 'w') as f:
            # Write data array then metainfo
            f.create_dataset('ndarray', data=self.data)
            __dict_loop(self.info, f.create_group('info'))

    def unitChk(self, b: 'metaArray') -> bool:
        """
        Compare the unit descriptions between two metaArrays.
        Return true if they match, raise UnitError otherwise.
        """
        if self['unit'] != b['unit']:
            msg = f"{self['unit']} != {b['unit']}"
            raise UnitError(msg)

        if self['range']['unit'] != b['range']['unit']:
            msg = f"{str(self['range']['unit'])} != {str(b['range']['unit'])}"
            raise UnitError(msg)

        return True

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
        return self._i2x(self.data.argmin(axis))

    def argmax(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        argMax value of the data array in x-y-z space
        """
        return self._i2x(self.data.argmax(axis))

    def ptp(self, axis: int = None) -> typing.Union[float, npt.NDArray]:
        """
        Peak to peak value of the data array
        """
        return np.ptp(self.data, axis=axis)

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

    def __basic_op(self, b: numeric,
                   op: str) -> typing.Union[npt.NDArray, 'metaArray']:
        """
        Basic arithmetic operations
        """
        # Simple ops
        if isinstance(b, (int, float, np.ndarray)):
            ary = self.copy()
            return self.__non_meta_op(ary, b, op)

        # metaArray operations
        elif isinstance(b, metaArray):
            # work out the common xyz region
            region = self.overlap(b)

            if self.debug:
                print("*** Overlap region: " + str(region))
                print(self[region])

            # Perform the operation on the common region
            newArray = self[region].copy()
            info = newArray.info
            b_region = b[region]
            binfo = b_region.copy_info()

            # If the ijk shape do not agree, have to resample the
            # data before arithmetic operation
            if newArray.shape != b_region.shape:
                # See if resampling is allowed
                if newArray['resample']:
                    msg = f"Unable to operate, non-identical array shapes {str(newArray.shape)} vs {str(b_region.shape)}, but auto-resampling not yet implemented"  # noqa E501
                    raise NotImplementedError(msg)
                else:
                    msg = f"Unable to operate, non-identical array shapes {str(newArray.shape)} vs {str(b_region.shape)}, but auto-resampling not allowed"  # noqa E501
                    raise ValueError(msg)

            if op == '+':
                try:
                    newArray.unitChk(b_region)
                    newArray.data += b_region.data
                except UnitError:
                    raise UnitError("Axis unit description do no match")

            elif op == '-':
                try:
                    newArray.unitChk(b_region)
                    newArray.data -= b_region.data
                except UnitError:
                    raise UnitError("Axis unit description do no match")

            elif op == '*':
                newArray.data *= b_region.data
                try:
                    info['unit'] = f"{info['unit']} * {b_region['unit']}"
                except TypeError:
                    info['unit'] = None

            elif op == '/':
                newArray.data /= b_region.data
                try:
                    info['unit'] = f"{info['unit']} / {b_region['unit']}"
                except TypeError:
                    info['unit'] = None

            elif op == '//':
                newArray.data = newArray.data.__floordiv__(b_region)
                try:
                    info['unit'] = f"{info['unit']} / {b_region['unit']}"
                except TypeError:
                    info['unit'] = None

            elif op == 't/':
                newArray.data = newArray.data.__truediv__(b_region)
                try:
                    info['unit'] = f"{info['unit']} / {b_region['unit']}"
                except TypeError:
                    info['unit'] = None

            elif op == '^':
                newArray.data = newArray.data ** b_region
                try:
                    info['unit'] = f"{info['unit']} ^ {b_region['unit']}"
                except TypeError:
                    info['unit'] = None
            else:
                raise ValueError("unknown operator" + str(op))

            # Update the metainfo.
            # metainfo that require processing before the generic merge process
            if info['name'] is not None:
                if b['name'] is not None:
                    info_name = info['name'] + op + b['name']
            elif b['name'] is not None:
                info_name = b['name']

            info_unit = info['unit']

            # Remove grand parent info in
            for field in info.keys():
                if field.find('.') != -1:
                    del info[field]

            # Merge the two branches together
            for field in binfo.keys():
                if field.find('.') != -1:
                    continue
                elif field in info:
                    if info[field] != binfo[field]:
                        info[field] += '|' + op + '|' + binfo[field]
                else:
                    info[field] = binfo[field]

            # Apply the metainfo that require processing before
            # the generic merge process
            info['name'] = info_name
            info['unit'] = info_unit
            return newArray

        # Unknown type
        else:
            raise ValueError("Only numeric types can be operated on metaArray")

    def __basic_iop(self, b: numeric,
                    op: 'str') -> typing.Union[npt.NDArray, 'metaArray']:
        """
        Basic inplace arithmetic operations
        """
        # Simple ops
        if isinstance(b, int) or isinstance(b, float) \
           or isinstance(b, npt.NDArray):
            ary = self
            return self.__non_meta_op(ary, b, op)

        # metaArray operations
        elif isinstance(b, metaArray):
            region = self.overlap(b)

            if self.debug:
                print("*** Overlap region: " + str(region))
                print(self[region])

            # Perform the operation on the common region
            newArray = self[region]
            info = newArray.info
            b_region = b[region]
            binfo = b_region.copy_info()

            # Resampling to be implemented.
            if len(newArray) != len(b_region):
                msg = "Different Shapes, auto-resampling not yet implemented."
                raise NotImplementedError(msg)

            if op == '+':
                try:
                    newArray.unitChk(b_region)
                    newArray.data += b_region.data
                except UnitError:
                    raise UnitError("Axis unit description do no match")

            elif op == '-':
                try:
                    newArray.unitChk(b_region)
                    newArray.data -= b_region.data
                except UnitError:
                    raise UnitError("Axis unit description do no match")

            elif op == '*':
                newArray.data *= b_region.data
                info['unit'] = info['unit'] + '*' + b_region['unit']

            elif op == '/':
                newArray.data /= b_region.data
                info['unit'] = info['unit'] + '/' + b_region['unit']

            elif op == '//':
                newArray.data = newArray.data.__floordiv__(b_region)
                info['unit'] = info['unit'] + '/' + b_region['unit']

            elif op == 't/':
                newArray.data = newArray.data.__truediv__(b_region)
                info['unit'] = info['unit'] + '/' + b_region['unit']

            elif op == '^':
                newArray.data = newArray.data ** b_region
                info['unit'] = info['unit'] + '^' + b_region['unit']

            else:
                raise ValueError("unknown operator" + str(op))

            # Update the metainfo.
            # metainfo that require processing before the generic merge process
            if info['name'] is not None:
                if b['name'] is not None:
                    info_name = info['name'] + op + b['name']
            elif b['name'] is not None:
                info_name = b['name']

            info_unit = info['unit']

            # Remove grand parent info in
            for field in info.keys():
                if field.find('.') != -1:
                    del info[field]

            # Merge the two branches together
            for field in binfo.keys():
                if field.find('.') != -1:
                    continue
                elif field in info:
                    if info[field] != binfo[field]:
                        info[field] += '|' + op + '|' + binfo[field]
                else:
                    info[field] = binfo[field]

            # Apply the metainfo that require processing before
            # the generic merge process
            info['name'] = info_name
            info['unit'] = info_unit
            return newArray

        # Unknown type
        else:
            raise ValueError("Only numeric types can be operated on metaArray")

    def __non_meta_op(self, ary: 'metaArray',
                      b: numeric,
                      op: 'str') -> 'metaArray':
        """
        Simple arithmetic operations with non-metaArray data types
        """
        if op == '+':
            ary.data += b
        elif op == '-':
            ary.data -= b
        elif op == '*':
            ary.data *= b
        elif op == '/':
            ary.data /= b
        elif op == '//':
            ary.data = ary.data.__floordiv__(b)
        elif op == 't/':
            ary.data = ary.data.__truediv__(b)
        elif op == '^':
            ary.data = ary.data ** b
        else:
            raise ValueError("Unknown operator" + str(op))

        return ary

    def __iadd__(self, b):
        """
        Inplace add
        """
        return

    def __isub__(self, b):
        return

    def __imul__(self, b):
        return

    def __idiv__(self, b):
        return

    def __itruediv__(self, b):
        return

    def __ifloordiv__(self, b):
        return

    def __imod__(self, b):
        return

    def __ipow__(self, b):
        return

    def __add__(self, b: numeric) -> 'metaArray':
        """
        Return the sum of a and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '+')

    def __sub__(self, b: numeric) -> 'metaArray':
        """
        Return the defference between self and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '-')

    def __div__(self, b: numeric) -> 'metaArray':
        """
        Return the quotient between self and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '/')

    def __floordiv__(self, b: numeric) -> 'metaArray':
        return self.__basic_op(b, '//')

    def __truediv__(self, b: numeric) -> 'metaArray':
        return self.__basic_op(b, 't/')

    def __mul__(self, b: numeric) -> 'metaArray':
        """
        Return the product self and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '*')

    def __neg__(self) -> 'metaArray':
        """
        Return the copy of negated self.
        """
        negArray = self.__basic_op(-1, '*')

        info = negArray.info
        if info['name'] is not None:
            info['name'] = "-" + info['name']

        return negArray

    def __pow__(self, b: numeric) -> 'metaArray':
        """
        Return self ** b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(-1, '^')


def __dict_loop(dic: dict, h5: h5py.File) -> None:
    """
    Recursively store items in a dictionary object into given h5py object as
    h5py.Dataset.

    If the dictionary item is itself an dictionary object, create a h5py.Group,
    and store subsequent items within.
    """

    # List of keys to the items in the current dict is itself a dict
    dict_lst = []

    # First loop, store all the simple data, and build the dict_lst
    for key, val in dic.items():
        if isinstance(val, dict):
            dict_lst.append(key)
        else:
            h5.create_dataset(key, data=val)

    # Process the dic_lst
    for key in dict_lst:

        # Create a subgroup
        grp = h5.create_group(key)

        __dict_loop(dic[key], grp)
