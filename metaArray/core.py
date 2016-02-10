#       core.py
#
#       Copyright 2013 charley <y.fan@warwick.ac.uk>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#
#
'''
Version history:

   0.6 Added "Order" field to axis range description to allow fft data representation
   0.5 Enabled pickling


To be implement:

   Inplace operations
   Non-identical array shape operations
'''

import numpy as np
from os import linesep
from copy import deepcopy

from metaArray.misc import linearFunc
from metaArray.misc import logFunc
from metaArray.misc import expFunc

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
    def __init__(self, values=None):
        dict.__init__(self)
        if values is not None:
            self.update(values)
        self.register = {}
        return

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        register = self.register
        for i in register:
            try:
                register[i]()
            except:
                print("Host object with id: " + str(i) + \
                "can not be updated, removing from range_info register")
        return

    def reg(self, obj, method=None):
        """
        Register host object, with an optional update method arguement.
        Host.update_range() will be used to notify any changes unless otherwise
        specified.
        """
        if method is None:
            method = obj.update_range

        self.register[id(obj)] = method
        return

    def dereg(self, obj):
        """
        Deregister host object
        """
        dict.__delitem__(self, id(obj))
        return

    def __deepcopy__(self, memo):
        """
        Deep copy of self dictionary contents.

        self.register will not be copied however.
        """

        # from copy import deepcopy
        return range_info(deepcopy(dict.copy(self)))



class metaArray(object):
    """
    Numpy array with meta data, currently only supports linear sampling.
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

    """
    def __init__(self, data, dtype=None, copy=False, info=None, debug=False):

        """
        data    Array-like data, will be convert to numpy.ndarray
        dtype   Array element data type to aid convertion to numpy.ndarray
        copy    Whether to make a copy of data when convert to numpy.ndarray
        info    Optional meta data, can be updated later via object methods
        debug   Debug level/flag)

        If given data is a metaArray instance,
        """

        # self.data = numpy.ndarray         # Array storage container
        # self.info = {}                    # Meta information container
        # self.ndim = data.ndim             # Number of dimensions

        # Convert the data into ndarray
        if isinstance(data, metaArray):
            if info is None:
                self_data = np.array(data.data, dtype=dtype, copy=copy)
                info = data.copy_info()
            else:
                self_data = np.array(data.data, dtype=dtype, copy=copy)
        else:
            self_data = np.array(data, dtype=dtype, copy=copy)

        ndim = self_data.ndim
        shape = self_data.shape

        self.debug = debug



        # Generate the default meta_info
        if debug: print("*** Generating the default meta_info")

        # These are global attributes
        self_info = {'name':None, \
                     'unit':None, \
                    'label':None, \
                 'resample':False}

        # Range is the corresponding xyz index
        if debug: print("*** Generating default range descriptions")

        # These are per-axis attributes
        self_info['range'] = range_info()
        self_info['range']['begin'] = list(np.zeros(ndim, dtype=float))    # Beginning coordinates
        self_info['range']['end'] = list(np.array(shape, dtype=float))  # Ending coordinates
        self_info['range']['unit'] = [None] * ndim                    # Unit
        self_info['range']['label'] = [None] * ndim                   # Label
        self_info['range']['log'] = [False] * ndim                    # Lin / log scale
        self_info['range']['fft'] = [False] * ndim                    # Data order (see numpy.fft.fftshift)

        # If info parameter is given, update them accordingly
        if info is not None:
            if debug: print("*** meta_info was supplied")

            if info.has_key('name'): self_info['name'] = info['name']
            if info.has_key('unit'): self_info['unit'] = info['unit']
            if info.has_key('label'): self_info['label'] = info['label']
            if info.has_key('resample'): self_info['resample'] = info['resample']

            if info.has_key('range'):

                if info['range'].has_key('begin'): self_info['range']['begin'] = list(info['range']['begin'])
                if info['range'].has_key('end'): self_info['range']['end'] = list(info['range']['end'])
                if info['range'].has_key('unit'): self_info['range']['unit'] = list(info['range']['unit'])
                if info['range'].has_key('label'): self_info['range']['label'] = list(info['range']['label'])
                if info['range'].has_key('log'): self_info['range']['log'] = list(info['range']['log'])
                if info['range'].has_key('fft'): self_info['range']['fft'] = list(info['range']['fft'])


        ## Use the specified 'info' parameter if given
        ## These are global attributes
        #if info is not None:

            #if debug: print "*** meta_info was supplied"

            #self_info = info

        #else:
            #if debug: print "*** Generating the default meta_info"

            #self_info = {'name':None, \
                         #'unit':None, \
                        #'label':None, \
                     #'resample':False}

        ## Check if range info is given
        ## These are per-axis attributes
        #if self_info.has_key('range'):

            #if debug: print "*** Range description is available"

            #self_info['range'] = range_info(self_info['range'])

            ## Make sure they are in list type, they have to be mutable
            #self_info['range']['begin'] = list(self_info['range']['begin']) # Beginning coordinates
            #self_info['range']['end'] = list(self_info['range']['end'])     # Ending coordinates
            #self_info['range']['unit'] = list(self_info['range']['unit'])   # Unit
            #self_info['range']['label'] = list(self_info['range']['label']) # Label
            #self_info['range']['log'] = list(self_info['range']['log'])     # Lin / log scale
            #self_info['range']['fft'] = list(self_info['range']['fft'])     # Data order (see numpy.fft.fftshift)

        #else:
            ## Range is the corresponding xyz index
            #if debug: print "*** Generating default range descriptions"

            #self_info['range'] = range_info()
            #self_info['range']['begin'] = list(zeros(ndim, dtype=float))    # Beginning coordinates
            #self_info['range']['end'] = list(array(shape, dtype=float))     # Ending coordinates
            #self_info['range']['unit'] = [ None ] * ndim                    # Unit
            #self_info['range']['label'] = [ None ] * ndim                   # Label
            #self_info['range']['log'] = [ False ] * ndim                    # Lin / log scale
            #self_info['range']['fft'] = [ False ] * ndim                    # Data order (see numpy.fft.fftshift)

        # Assemble the metaArray
        self.data = self_data
        self.info = self_info

        # Duplicate some of the common ndarray attributes (instead of sub-classing)
        self.ndim = ndim
        self.shape = shape
        self.ctypes = self_data.ctypes
        self.dtype = self_data.dtype
        self.itemsize = self_data.itemsize
        self.nbytes = self_data.nbytes
        self.size = self_data.size

        # Register self with the range_info object and update conversion functions
        self_info['range'].reg(self)
        self.update_range()

        return

    def __repr__(self):
        """
        Text representation of the object
        """
        # Make a copy first, because some of the entries will be destroyed along the way
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
            range_desc += "%(Begin)0.2e \t " % {'Begin':range_nfo['begin'][i]}
            range_desc += "%(End)0.2e \t " % {'End':range_nfo['end'][i]}

            scale = range_nfo['log'][i]
            if scale is None or scale == False:
                scale = "Linear"
            elif isinstance(scale, int) or isinstance(scale, float):
                scale = "log" + str(scale)
            elif scale == True:
                scale = "log10"
            range_desc += "%(Scale)-8s \t " % {'Scale':scale}

            try:
                order = range_nfo['fft'][i]
                if order is None or order == False:
                    order = "Normal"
            except KeyError:
                order = "Normal"

            range_desc += "%(Order)-8s \t " % {'Order':order}
            range_desc += "%(Unit)-8s \t " % {'Unit':range_nfo['unit'][i]}
            range_desc += "%(Label)-8s \t " % {'Label':range_nfo['label'][i]}
            range_desc += linesep

        if self.debug != True:
            del nfo['range']

        # Produce some of the default descriptions.
        desc = "array(" + linesep + str(self.data) + ")" + linesep
        desc += '=' * 72 + linesep
        desc += "Meta Info: " + linesep
        try:
            desc += "\t['name'] Title of this metaArray: " + str(nfo.pop('name')) + linesep
        except:
            pass

        try:
            unit = nfo.pop('unit')
            if unit is None:
                desc += "\t['unit'] Array element quantity unit is not defined." + linesep
            elif unit == '':
                desc += "\t['unit'] Array element quantity is unitless." + linesep
            else:
                desc += "\t['unit'] Array element quantity unit is: " + unit + linesep
        except:
            pass

        try:
            desc += "\t['label'] Array element quantity label is: " + str(nfo.pop('label')) + linesep
        except:
            pass

        try:
            resample = nfo.pop('resample')
            if resample == True:
                desc += "\t['resample'] Automatic resampling of this array is allowed." + linesep
            else:
                desc += "\t['resample'] Automatic resampling of this array is prohibited." + linesep
        except:
            pass

        # FFT attribute needs be apply per axis
        ###################################
        #try:
        #    fft = nfo.pop('fft')
        #    if fft == True:
        #        desc += "\t['fft'] Array data in frequency domain representation."
        #        if self.debug = True:
        #            desc += linesep + "\t\t(i.e. See numpy.fft.fftshift)"
        #        desc += linesep
        #    else:
        #        desc += "\t['fft'] Array data in normal representation." + linesep
        #except:
        #    pass
        ####################################

        # Produce the rest of the descriptions
        ext_desc = ''
        for field, value in sorted(nfo.items()):
            ext_desc += "\t['" + field + "'] = " + str(value) + linesep

        if ext_desc != '':
            desc += '=' * 72 + linesep
            desc += ext_desc

        # Append the range descriptions
        desc += range_desc

        return desc

    def __proc_key(self, key, axis=0):
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
            return slice(start[0], stop[0], step[0]), slice(start[1], stop[1], step[1])
        else:
            keys = key_pair(key, axis=axis)
            return keys[0], keys[1]

    def __getitem__(self, key):
        """
        Modified method

        if key is string, return self.info['key']
        if key is int, return self.data[key] with corresponding meta info
        if key is floart, convert key to ijk space, return self.data[ikey] with corresponding meta info
        if key is tuple, convert xyz indexes to ijk space first, then return self.data[ikey] with corresponding meta info
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
                raise NotImplementedError("Non unity slice stepping is not supported. Use resample() instead." + str(key))
            return self.__getslice__(key.start, key.stop)

        key_pair = self._key_pair
        proc_key = self.__proc_key
        nfo = self.copy_info()                  # Init the new nfo dict
        nfo_range = nfo['range']

        if isinstance(key, tuple):            # Could be a tuple of int, float and slice
            ijk_key = []                        # Init a new key tuple for the ijk version of the key
            ijk_shape = []                      # Expected shape of the array
            # Walk through each axis
            for i in range(len(key)):
                keys = proc_key(key[i], axis=i)
                ikey = keys[0]
                ijk_key.append(ikey)            # This will be the ijk index pass on to ndarray
                if self.debug:
                    print("*** getitem ijk_key component:", ikey)

                if isinstance(ikey, slice):     # Slice are given, update meta info accordingly
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
                    ## ijk_shape.append(1)          # Can not do delete here, must keep dimension in order
                    # del nfo_range['begin'][i]     # not to confuse the i index for later dimensions fetching
                    # del nfo_range['end'][i]       # Do the tidy up later
                    # del nfo_range['unit'][i]

            # Tidy up the empty dimensions
            # Start with highest index, so deletion will not affect later index counts
            for i in range(len(ijk_shape))[::-1]:
                if ijk_shape[i] == 1:
                    del ijk_shape[i]
                    del nfo_range['begin'][i]
                    del nfo_range['end'][i]
                    del nfo_range['unit'][i]
                    del nfo_range['label'][i]
            try:
                # dat = self.data[tuple(ijk_key)].reshape(ijk_shape)     # Obtain the array slice
                dat = self.data[tuple(ijk_key)]     # Obtain the array slice
            except:
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
            raise IndexError("Expecting <int|float|tuple|string> key data type, " + str(type(key)) + " given.")


        if isinstance(dat, int) \
        or isinstance(dat, float) \
        or isinstance(dat, complex):  # If only one element remaining, no need to return meta info.
            return dat
        else:
            return metaArray(dat, info=nfo)

    def update_range(self):
        """
        Check if range info has been updated, update converstion ijk to xyz
        space conversion functions accordingly if so.
        """
        if self.debug:
            print("### Range info has changed, updating.")

        self.i2x = self.gen_i2x()
        self.x2i = self.gen_x2i()

        return

    def update(self, info_dict):
        """
        Imitate the dict.update() method

        Will override existing values, use with care!
        """

        self.info.update(info_dict)

        return


    def set_range(self, axis, field, value):
        """
        Method to set range meta info

        Example: self.set_range(0, 'unit', 's')
        """

        nfo_range = self.info['range']

        if nfo_range.has_key(field):
            nfo_range[field][axis] = value
        else:
            raise ValueError("Requested field (" + field + ") name do not exist")

        self.update_range()
        return

    def get_range(self, axis, field):
        """
        Method to get range meta info

        Example: self.set_range(0, 'unit')
        """
        nfo_range = self.info['range']

        if nfo_range.has_key(field):
            return nfo_range[field][axis]
        else:
            raise ValueError("Requested field (" + field + ") name do not exist")

        return False


    def get_smp_rate(self, axis=0):
        """
        Return the sampling rate of the given axis

        Example: self.get_smp_rate(0)
        """

        assert type(axis) is int, "Axis is not an integer: %r" % axis

        x0 = self.get_range(axis, 'begin')
        x1 = self.get_range(axis, 'end')
        n = self.data.shape[axis]

        return float(n) / abs(x1 - x0)


    def get_axis(self, axis=0):
        """
        Return a 1D numpy ndarray representing the discretized real space
        indexies of the given axis

        Example: time = self.get_axis(0)
        Returns, for instance, time axis
        """

        assert type(axis) is int, "Axis is not an integer: %r" % axis

        begin = self.get_range(axis, 'begin')
        end = self.get_range(axis, 'end')
        n = self.data.shape[axis]

        if self.get_range(axis, 'log') == True:
            return np.logspace(begin, end, n)
        else:
            return np.linspace(begin, end, n)


    def __getslice__(self, begin, end):
        """
        Modified slice method
        Do not support stepping
        """
        if self.debug:
            print("*** getslice: ", begin, end)

        nfo = self.copy_info()                  # Init the new nfo dict
        nfo_range = nfo['range']

        key_pair = self._key_pair
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
            iend = new_shape[0]         # Not new yet, still identical to list(self.shape)
        new_shape[0] = iend - ibegin    # Here is the new shape

        # Update the meta info
        if xbegin is not None:
            nfo_range['begin'][0] = xbegin
        if xend is not None:
            nfo_range['end'][0] = xend

        dat = self.data[ibegin:iend].reshape(new_shape)     # Obtain the array slice

        return metaArray(dat, info=nfo)

    def __len__(self):
        return len(self.data)

    def __abs__(self):
        """
        Return a absolute value copy of self
        """
        ary = metaArray(abs(self.data), info=self.copy_info())
        try:
            ary['name'] = 'abs(' + ary['name'] + ')'
        except TypeError:
            pass
        return ary

    def truth(self):
        """
        Returns True value
        """
        return True

    def __basic_op(self, b, op):
        """
        Basic arithmetic operations
        """
        # Simple ops
        if isinstance(b, (int, float, np.ndarray)):
            ary = self.copy()
            return self.__non_meta_op(ary, b, op)

        elif isinstance(b, metaArray):                      # metaArray operations
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

            # If the ijk shape do not agree, have to resample the data before arithmetic operation
            if newArray.shape != b_region.shape:
                # See if resampling is allowed
                if newArray['resample'] == True:
                    raise NotImplementedError("Unable to operate, non-identical array shapes " + \
                        str(newArray.shape) + " vs " + str(b_region.shape) + \
                        " , but auto-resampling is not yet implemented.")
                else:
                    raise ValueError("Unable to operate, non-identical array shapes " + \
                        str(newArray.shape) + " vs " + str(b_region.shape) + \
                        " , and auto-resampling is not allowed.")

            if op == '+':
                try:
                    newArray.unitChk(b_region)
                except UnitError:
                    raise UnitError("Axis unit description do no match")

                newArray.data += b_region.data
            elif op == '-':
                if not newArray.unitChk(b_region):
                    raise UnitError("Axis unit description do no match")
                else:
                    newArray.data -= b_region.data
            elif op == '*':
                newArray.data *= b_region.data
                try:
                    info['unit'] = info['unit'] + '*' + b_region['unit']
                except TypeError:
                    info['unit'] = None
            elif op == '/':
                newArray.data /= b_region.data
                try:
                    info['unit'] = info['unit'] + '/' + b_region['unit']
                except TypeError:
                    info['unit'] = None
            elif op == '//':
                newArray.data = newArray.data.__floordiv__(b_region)
                try:
                    info['unit'] = info['unit'] + '/' + b_region['unit']
                except TypeError:
                    info['unit'] = None
            elif op == 't/':
                newArray.data = newArray.data.__truediv__(b_region)
                try:
                    info['unit'] = info['unit'] + '/' + b_region['unit']
                except TypeError:
                    info['unit'] = None
            elif op == '^':
                newArray.data = newArray.data ** b_region
                try:
                    info['unit'] = info['unit'] + '^' + b_region['unit']
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
                elif info.has_key(field):
                    if info[field] != binfo[field]:
                        info[field] += '|' + op + '|' + binfo[field]
                else:
                    info[field] = binfo[field]

            # Apply the metainfo that require processing before the generic merge process
            info['name'] = info_name
            info['unit'] = info_unit
            return newArray

        # Unknown type
        else:
            raise ValueError("Only numeric types can be operated on metaArray")

    def __basic_iop(self, b, op):
        """
        Basic inplace arithmetic operations
        """
        # Simple ops
        if isinstance(b, int) or isinstance(b, float) or isinstance(b, np.ndarray):
            ary = self
            return self.__non_meta_op(ary, b, op)

        elif isinstance(b, metaArray):                      # metaArray operations
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
                raise NotImplementedError("Non-identical array length, auto-resampling is not yet implemented.")

            if op == '+':
                try:
                    newArray.unitChk(b_region)
                except UnitError:
                    raise UnitError("Axis unit description do no match")

                newArray.data += b_region.data
            elif op == '-':
                if not newArray.unitChk(b_region):
                    raise UnitError("Axis unit description do no match")
                else:
                    newArray.data -= b_region.data
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
                elif info.has_key(field):
                    if info[field] != binfo[field]:
                        info[field] += '|' + op + '|' + binfo[field]
                else:
                    info[field] = binfo[field]

            # Apply the metainfo that require processing before the generic merge process
            info['name'] = info_name
            info['unit'] = info_unit
            return newArray

        # Unknown type
        else:
            raise ValueError("Only numeric types can be operated on metaArray")

    def __non_meta_op(self, ary, b, op):
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

    def __add__(self, b):
        """
        Return the sum of a and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '+')

    def __sub__(self, b):
        """
        Return the defference between self and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '-')

    def __div__(self, b):
        """
        Return the quotient between self and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '/')

    def __floordiv__(self, b):
        return self.__basic_op(b, '//')

    def __truediv__(self, b):
        return self.__basic_op(b, 't/')

    def __mul__(self, b):
        """
        Return the product self and b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(b, '*')

    def __neg__(self):
        """
        Return the copy of negated self.
        """
        negArray = self.__basic_op(-1, '*')

        info = negArray.info
        if info['name'] is not None:
            info['name'] = "-" + info['name']

        return negArray

    def __pow__(self, b):
        """
        Return self ** b, aligned in xyz space.
        Only overlapping regions will be returned.
        """
        return self.__basic_op(-1, '^')

    def __contains__(self, b):
        """
        Retrun true if self xyz space contains the b xyz in entirety
        """
        if not ((isinstance(self, metaArray) and isinstance(b, metaArray))):
            raise TypeError("Both operand must be metaArray objects")

        ndim = self.ndim
        if ndim != b.ndim:
            raise DimError("Both operand must have the same number of dimensions")

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

    def __delitem__(self, key):
        raise ValueError("Cannot delete array elements")

    def __delslice__(self, b, c):
        raise ValueError("Cannot delete array elements")

    def __setitem__(self, key, value):
        """
        Modified method

        if key is int, return self.data[key] with corresponding meta info
        if key is floart, convert key to ijk space, return self.data[ikey] with corresponding meta info
        if key is string, return self.info['key']
        if key is tuple, convert xyz indexes to ijk space first, then return self.data[ikey] with corresponding meta info
        """
        if self.debug:
            print("*** setitem", key, value)

        if isinstance(key, str):                # Only have to set the meta info value
            self.info[key] = value
            if key == 'range':                  # The is really for backward compatibility. Should use set_range method instead
                self.update_range()
            return

        key_pair = self._key_pair
        proc_key = self.__proc_key

        if isinstance(key, tuple):              # Could be a tuple of int, float or slice
            ijk_key = []                        # Init a new key tuple for the ijk version of the key
            # Walk through each axis
            for i in range(len(key)):
                ijk_key.append(proc_key(key[i], axis=i)[0])

            self.data[tuple(ijk_key)] = value        # Write to the array slice
        else:                                        # must be int or float
            self.data[proc_key(key)[0]] = value
        return

    def __setslice__(self, begin, end, value):
        """
        Modified slice method
        Do not support stepping
        """
        if self.debug:
            print("*** setslice", begin, end, value)

        key_pair = self._key_pair
        proc_key = self.__proc_key

        ibegin = proc_key(begin)[0]
        iend = proc_key(end)[0]

        # Get the indexes right
        if ibegin is None:
            ibegin = 0
        if iend is None:
            iend = self.shape[0]

        self.data[ibegin:iend] = value

        return

    def __getstate__(self):
        odict = {}
        # odict = self.__dict__.copy()
        odict['info'] = self.copy_info()
        odict['data'] = self.data
        odict['dtype'] = self.dtype
        odict['debug'] = self.debug
        #del odict['i2x']
        #del odict['x2i']
        #del odict['ctypes']
        return odict

    def __setstate__(self, dict):
        #fh = open(dict['file'])      # reopen file
        #count = dict['lineno']       # read from file...
        #while count:                 # until line count is restored
        #    fh.readline()
        #    count = count - 1
        #self.__dict__.update(dict)   # update attributes
        #self.fh = fh                 # save the file object

        self.__init__(dict['data'], info=dict['info'], dtype=dict['dtype'], debug=dict['debug'])
        # dtype=None, copy=False, debug=False
        self.update_range()


    def __copy__(self):
        """All copies are deep copies"""
        return self.copy()

    def __deepcopy__(self):
        """All copies are deep copies"""
        return self.copy()

    def copy(self):
        """
        Retrun a duplicate copy (deep copy) of self
        """
        return metaArray(self.data, info=self.copy_info(), copy=True)

    def copy_info(self):
        """
        Return a duplicate copy of own info

        Some of the sequence data types require explicit copy operations to
        aviod just duplicating object pointers.
        """
        if self.debug: print("*** Duplicating meta info")

        # from copy import deepcopy
        #
        #info = self.info
        #range = info['range']
        #nfo = {}
        #nfo.update(info)
        #nfo['range'] = range.copy()
        #nfo['range']['end'] = list(range['end'])
        #nfo['range']['begin'] = list(range['begin'])
        #nfo['range']['unit'] = list(range['unit'])
        #nfo['range']['label'] = list(range['label'])
        return deepcopy(self.info)

    def copy_root_info(self):
        """
        Return a duplicate copy of own info, except for those keys with a '.' char
        """
        info = self.copy_info()

        for field in info.keys():
            if field.find('.') != -1:
                del info[field]

        return info


    def _key_pair(self, key, axis=0):
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
            raise ValueError("Indexes must be int or float, " + str(key) + " given.")

        if self.debug:
            print("*** _key_pair(" + str(key) + ") => ikey: " + str(ikey) + ", xkey:" + str(xkey))

        return ikey, xkey

    def _x2i(self, key, axis=0):
        """
        Convert xyz indexes to their ijk counter parts
        """
        return int(np.round(self.x2i[axis](key)))

    def _i2x(self, key, axis=0):
        """
        Convert ijk indexes to their xyz counter parts
        """
        return self.i2x[axis](key)

    def gen_x2i(self):
        """
        Regenerate the xyz to ijk space convertion functions
        """
        shape = self.shape = self.data.shape
        x0 = self.info['range']['begin']
        x1 = self.info['range']['end']
        lg = self.info['range']['log']
        # i0 = 0
        # i1 = len(self.data[i])
        lst = []
        for i in range(self.ndim):
            if lg[i] is None or lg[i] == False:
                # Log scale is not applied, everything is linear
                if x0[i] == x1[i]:
                    # Start stop at the same point
                    lst.append(lambda x: 0)
                else:
                    lst.append(linearFunc(x0[i], 0, x1[i], shape[i]))
            elif isinstance(lg[i], int) or isinstance(lg[i], float):
                # Log scale applied, given log base.
                lst.append(logFunc(x0[i], 0, x1[i], shape[i], base=lg[i]))
            elif lg[i] == True:
                # Log scale applied, use default base
                lst.append(logFunc(x0[i], 0, x1[i], shape[i]))
            else:
                raise(ValueError, "Log scale descriptor can only be int,\
                    float, True, False or None, given: " + str(lg[i]))
        return lst

    def gen_i2x(self):
        """
        Regenerate the ijk to xyz space convertion functions
        """
        shape = self.shape = self.data.shape
        x0 = self.info['range']['begin']
        x1 = self.info['range']['end']
        lg = self.info['range']['log']
        # i0 = 0
        # i1 = len(self.data[i])
        if self.debug:
            print("*** starting xyz:", x0, "ending xyz:", x1, "shape:", shape, "ndim:", self.ndim)
        lst = []
        for i in range(self.ndim):
            if lg[i] is None or lg[i] == False:
                # Log scale is not applied, everything is linear
                if shape[i] == 1:
                    # Start stop at the same point
                    lst.append(lambda x: x0[i])
                else:
                    lst.append(linearFunc(0, x0[i], shape[i], x1[i]))
            elif isinstance(lg[i], (int, float)):
                # Log scale applied, given log base.
                lst.append(expFunc(x0[i], 0, x1[i], shape[i], base=lg[i]))
            elif lg[i] == True:
                # Log scale applied, use default base
                lst.append(expFunc(x0[i], 0, x1[i], shape[i]))
            else:
                raise ValueError("Log scale descriptor can only be int,\
                    float, True, False or None, given: " + str(lg[i]))
        return lst

    def overlap(self, b):
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
            raise DimError("Both operand must have the same number of dimensions")

        abegin = self.info['range']['begin']
        aend = self.info['range']['end']

        bbegin = b.info['range']['begin']
        bend = b.info['range']['end']

        output = []

        # Check common grounds per dimension
        for i in range(ndim):
            # take the highest starting point
            if abegin[i] > bbegin[i]:
                begin = float(abegin[i])
            else:
                begin = float(bbegin[i])

            # take the lowest ending point
            if aend[i] < bend[i]:
                end = float(aend[i])
            else:
                end = float(bend[i])

            # There must be at least some overlap in all dimensions
            if begin > end:
                raise NoOverlapError

            output.append(slice(begin, end))

        return tuple(output)

    def unitChk(self, b):
        """
        Compare the unit descriptions between two metaArrays.
        Return true if they match, raise UnitError otherwise.
        """
        if self['unit'] != b['unit']:
            raise UnitError(self['unit'] + " != " + b['unit'])

        if self['range']['unit'] != b['range']['unit']:
            raise UnitError(str(self['range']['unit']) + " != " + str(b['range']['unit']))

        return True

    def min(self, axis=None):
        """
        Min value of the data array
        """
        return self.data.min(axis)

    def max(self, axis=None):
        """
        Max value of the data array
        """
        return self.data.max(axis)

    def argmin(self, axis=None):
        """
        argMin value of the data array in x-y-z space
        """
        return self._i2x(self.data.argmin(axis))

    def argmax(self, axis=None):
        """
        argMax value of the data array in x-y-z space
        """
        return self._i2x(self.data.argmax(axis))

    def ptp(self, axis=None):
        """
        Peak to peak value of the data array
        """
        return self.data.ptp(axis)


    def log10(self):
        """
        Take the log10 of values
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
