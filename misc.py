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


# This file contain a number a miscellaneous functions
#
# linearChk     Check the input array is linear
# unitPrefix    Calculate the unit prefix.
# 
# Package dependency:
#     PIL
#     numpy
#     scipy
#

from decimal import Decimal

from string import rfind
from string import zfill
from struct import unpack, calcsize

import time
import datetime

from os import access
from os import F_OK
from os import linesep
from os import listdir
from os import R_OK
from os import sep
from os import W_OK
from os import walk
from os import X_OK
from os.path import abspath
from os.path import basename
from os.path import isdir
from os.path import isfile
from os.path import join
from os.path import split

from numpy import append
from numpy import arange
from numpy import array
from numpy import ceil, floor
from numpy import cos
from numpy import diff
from numpy import linspace
from numpy import log, exp, log10
from numpy import mod, abs
from numpy import nonzero
from numpy import ones, zeros
from numpy import pi
from numpy import real, imag
from numpy import round
from numpy import sign
from numpy import sort
from numpy import squeeze
from numpy import tile
from numpy import where
from numpy.fft import rfft, irfft

from scipy.interpolate import splev
from scipy.interpolate import splrep
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import filtfilt as scipy_filtfilt
from scipy.signal import firwin
from scipy.signal import lfilter
from scipy.signal.wavelets import morlet

from constants import GaussFWHM


class QuantsationError(ValueError):
    """
    Unable to quantise the float array without significant error in value.
    """
    pass


class InsufficientInput(ValueError):
    """
    InsufficientInput is raised when insufficient input variables are found.
    Not all the necessary parameters can be calculated from the given inputs.
    """
    pass


class dirPath(object):
    """
    DirPath object, provide various information about the dir path
    
    currently provide the following:
    
    self.full       The full path, i.e. the input string
    self.name       The dir name
    self.read       Whether it is readable
    self.write      Whether it is writable
    self.exist      Whether it already exists
    """
    
    def __init__(self, path):
        
        self.full = path = abspath(path.strip())
        self.name = basename(path)
        self.read = access(path, R_OK)
        self.write = access(path, W_OK)
        self.exist = access(path, F_OK)
        return
    
    def __repr__(self):
        """
        Text representation of the object
        """
        desc = ''
        desc += 'Full: ' + self.full + linesep
        desc += 'Name: ' + self.name + linesep
        desc += 'Read: ' + str(self.read) + linesep
        desc += 'Write: ' + str(self.write) + linesep
        desc += 'exist: ' + str(self.exist) + linesep
        
        return desc


class filePath(object):
    """
    FilePath object, provide various information about the file path
    
    currently provide the following:
    
    self.full       The full path, i.e. the input string
    self.baseDir    The dir path to the file
    self.name       The file name
    self.ext        The file extension
    self.read
    self.write
    self.exist      Whether it already exists
    """
    
    def __init__(self, file_path):
        
        self.full = path = abspath(file_path.strip())
        self.read = access(path, R_OK)
        self.write = access(path, W_OK)
        self.execute = access(path, X_OK)
        self.exist = access(path, F_OK)
        
        # self.baseDir = basename(path[:path.rfind(sep)])
        self.baseDir, fname = split(path)
        
        # Modify the behaviour if the file does not exit, check if the directory 
        # is writable instead
        if not self.exist:
            self.write = access(self.baseDir, W_OK)
        
        self.name = fname[:fname.rfind('.')]
        self.ext = fname[fname.rfind('.')+1:]
        
        return
        
    
    def __repr__(self):
        """
        Text representation of the object
        """
        desc = ''
        desc += 'Full: ' + self.full + linesep
        desc += 'Name: ' + self.name + linesep
        desc += 'Read: ' + str(self.read) + linesep
        desc += 'Write: ' + str(self.write) + linesep
        desc += 'Execute: ' + str(self.execute) + linesep
        desc += 'Exist: ' + str(self.exist) + linesep
        desc += 'baseDir: ' + self.baseDir + linesep
        desc += 'extension: ' + self.ext + linesep
        
        return desc



class cplx_trig_func(object):
    """
    A complex trigonometic function object.

    Not all input variables needs be defined, it will try to work them out.
    """

    def __init__(self, nLambda = False, pts = False, debug = False, \
        length = False, freq = False, samp_rate = False, dt = False):
        
        # Externally defined variables
        # Tier 1    
        # Native set of parameters, will try to derive from higher tiers if
        # not given. 
        self.nLambda = nLambda              # Number of wavelengths
        self.pts = pts                      # Number of samples
        self.debug = debug                  
        # Tier 2
        self.length = length                # Duration = nLambda * freq
        self.freq = freq                    # Frequency of oscillation
        # Tier 3
        self.samp_rate = samp_rate          # Sampling rate
        self.dt = dt                        # Sampling interval = pts * dt
        
        # Localy generated variables
        self.dx = None
        self.x = None
        
        # Private vars
        self.flg_resolved = False
        
        return

    def __call__(self):
        """
        Retrun the complex trigonometic oscillation function
        """
        # Try to reolve all the parameters from given data
        self.resolve()
        
        if self.x is None:
            x = linspace(0, self.nLambda*2*pi, self.pts)
            self.x = x
            self.dx = x[1]
        
        return exp(1j*self.x)

    def get_radian_space(self):
        """
        Return the radian space array of the complex trigonometic oscillation object.
        """
        # Try to reolve all the parameters from given data
        self.resolve()
        
        if self.x is not None:
            x = linspace(0, self.nLambda*2*pi, self.pts)
            self.x = x
            self.dx = x[1]
        
        return x

    def resolve(self):
        """
        Try to reolve all the parameters from given data
        """
        
        if self.flg_resolved:
            return          # Already resolved
        
        # Tier 1
        nLambda = self.nLambda
        pts = self.pts
        debug = self.debug
        # Tier 2
        length = self.length
        freq = self.freq
        # Tier 3
        samp_rate = self.samp_rate
        dt = self.dt
        
        # Identify Tier 1 Variables
        if pts == False:
            # pts can be obtain from (length x samp_rate) or (length / dt)
            # Need length in either case
            if length == False:
                # length can be obtained from (nLambda * freq)
                if (nLambda == False) or (freq == False):
                    raise InsufficientInput, "nLambda (number of wavelengths) and freq (frequency) are needed to define length (duration)"
                else:
                    length = float(nLambda) / float(freq)
            
            # length is now identified
            if samp_rate is not False:
                pts = float(length) * float(samp_rate)
            elif dt is not False:
                pts = float(length) / float(dt)
            else:
                raise InsufficientInput, "Either samp_rate (sampling rate) or dt (sampling interval) is needed to define pts (number of points)"
        
        
        # Identify Tier 1 variables
        if nLambda == False:
            # nLambda can be obtained from (length*freq)
            if freq == False:
                raise InsufficientInput, "freq (frequency) is needed to define nLambda (number of wavelengths)"
            
            if length == False:
                # length can be obtained from (pts * dt) or (pts / samp_rate)
                # Need pts in either case
                if pts == False:
                    raise InsufficientInput, "pts (number of points) is needed to define length (duration)"
                
                if dt is not False:
                    length = int(round(pts)) * float(dt)
                elif samp_rate is not False:
                    length = int(round(pts)) / float(samp_rate)
                else:
                    raise InsufficientInput, "Either samp_rate (sampling rate) or dt (sampling interval) is needed to define length (duration)"
            
            nLambda = float(length) * float(freq)
        
        # All the Tier 1 variables are solved
        # Solve the rest of the variables
        if length == False:
            if freq is not False:
                length = nLambda / freq
            elif dt is not False:
                length = pts * dt
            elif samp_rate is not False:
                length = pts / samp_rate
            elif debug:
                print('*** Unable to resolve Tier 2 variable length')
        
        if freq == False:
            if length is not False:
                freq = nLambda / length
            elif debug:
                print('*** Unable to resolve Tier 2 variable freq')
        
        if samp_rate == False:
            if length is not False:
                samp_rate = pts / length
        
        if dt == False:
            if length is not False:
                dt = length / pts
        
        # Write the variables back to the object
        self.nLambda = nLambda
        self.freq = freq
        self.pts = pts
        self.length = length
        self.samp_rate = samp_rate
        self.dt = dt
        
        # Indicate all necessary variables has been resolved
        self.flg_resolved = True
        
        return

    def __repr__(self):
        """
        Text representation of the object
        """
        desc = 'This is a complex trigonometic oscillation function object.' + linesep
        desc += 'It has the following attributes:' + linesep
        desc += '\tnLambda is: ' + str(self.nLambda) + linesep
        desc += '\tpts is: ' + str(self.pts) + linesep
        desc += '\tlength is: ' + str(self.length) + linesep
        desc += '\tfreq is: ' + str(self.freq) + linesep
        desc += '\tsamp_rate is: ' + str(self.samp_rate) + linesep
        desc += '\tdt is: ' + str(self.dt) + linesep
        desc += '\tdx is: ' + str(self.dx) + linesep
        
        desc += '\tRadian space is '
        if self.x is None:
            desc += 'not defined.' + linesep
        else:
            desc += 'defined.' + linesep
        
        desc += linesep + '\tDebugging output is '
        if self.debug:
            desc += 'enabled.' + linesep
        else:
            desc += 'disabled.' + linesep
        
        desc += '\tParameter definitions has '
        if self.flg_resolved:
            desc += 'been resolved.' + linesep
        else:
            desc += 'not been resolved.' + linesep
            desc += '\t Use the resolve() method to resolve parameter definitions.' + linesep
        
        desc += linesep + '\tUse the get_radian_space() method to obtain the radian space values.' + linesep
        desc += '\tCall the object to obtain the complex trigonometic function array.'
        
        return desc



class mother_morlet(object):
    """
    A mother morlet object
    """

    def __init__(self, M, w = 5.0, s = 1.0, A = 1.0,\
        complete = True, name=None, same_len=False, unit='scale'):
        
        """
        M           Length of the mother wavelet
        w           Omega0
        s           Scaling factor, windowed from -s*2*pi to +s*2*pi
        A           Amplitude of the mother wavelet
        complete    Use the complete or standard version
        name        Name of the mother wavelet
        same_len    Should daughter wavelet have the same M (but different s) as the mother wavelet.
        unit        Unit description for the scale (e.g. self(10) === 10 Hz)
        """
        
        self.length = int(round(M))
        self.Omega0 = w
        self.window = s
        self.Amplitude = A
        self.unit = unit
        
        # FWHM of a Gaussian is 2 * ( 2 * ln(2) )**0.5 = 2.3548200450309493
        # The fundamental scale is twice of that.
        # Scale of a trig function is 2*pi / w
        twopi = 2 * pi
        
        # Characteristic scale (relative "wavelength" to w = 1)
        scale = (1 / (2 * GaussFWHM)) + (1 / (twopi/w))
        scale = 1 / scale
        self.scale = scale / twopi
        
        if name is not None:
            self.name = name
        else:
            # Generate a default name
            self.name = "Morlet" + str(w)
        
        self.flg_complete = complete
        self.flg_same_len = same_len
        
        return


    def __call__(self, scale):
        """
        Scale relative to the mother wavelet
        
        Scale in here is not the same as the 's' in morlet(M, w, s)
        """
        amp = self.Amplitude / ((abs(scale))**0.5)
        
        if self.flg_same_len:
            M = self.length
            s = self.window / scale
        else:
            M = int(round(self.length * scale))
            s = self.window
        
        return amp * morlet(M, w = self.Omega0, s = s, complete = self.flg_complete)


    def __repr__(self):
        if self.flg_complete:
            desc = "Morlet (low oscillation corrected) mother wavelet of Omega0 = %(Omega0)1.2f." + linesep
        else:
            desc = "Morlet mother wavelet of Omega0 = %(Omega0)1.2f." + linesep
        if self.name is not None:
            desc += "\t " + self.name + linesep
        desc += "\t Number of points: " + str(self.length) + linesep
        desc += "\t Window range: -%(s)1.2fpi to %(s)1.2fpi" + linesep
        desc += "\t Unit Amplitude: %(A)1.3f" + linesep
        desc += "\t Has a characteristic scale of: %(Scale)1.3f*2*pi" + linesep
        desc += "\t Daughter wavelets have a scale in the unit of: %(Unit)s" + linesep
        if self.flg_same_len:
            desc += "\t Constant number of points for daugter wavelets."
        else:
            desc += "\t Constant window range for daugter wavelets."
        
        return desc % {'Omega0':self.Omega0, \
                        's':self.window*2, \
                        'A':self.Amplitude, \
                        'Scale':self.scale, \
                        'Unit':self.unit}




def prettyunit(unit, v0, v1):
    """
    This is a more specialised function for scaling the axis when plotting.
    
    Given the value range from V0 to V1, return suitable unit prefix and 
    scaling factor if unit definition exists.
    
    Do nothing if unit is not defined. i.e. is None or is ''
    
    """
    scale = 1       # Default scale is not to modify anything
    
    if unit == None or unit == '': 
        return unit, v0, v1, scale
    
    # Apply unit prefix if unit is defined
    if abs(v0) > abs(v1):
        vref = v0
    else:
        vref = v1
    
    scaled_vref, long_hand, short_hand, exponent = unitPrefix(vref)
    scale = scaled_vref / vref
    v0 *= scale
    v1 *= scale
    
    if short_hand == 'u':
        unit = '$\mu$ ' + unit
        #unit = short_hand + unit
    else:
        unit = short_hand + unit
    
    return unit, v0, v1, scale


def buffered_search(f, string, start = 0, buffer_size = 4194304):
    """
    Given a file object [f], search for the location of [string], from the 
    given [start] offset position. 
    
    This function will only read [buffer_size] bytes into memory at a time.
    
    Return -1 if string not found.
    """
    str_len = len(string)
    
    if str_len > buffer_size:
        raise BufferError, "Target string longer than buffer length."
    
    f_pos = start
    
    while True:
        f.seek(f_pos)
        buf = f.read(buffer_size)
        
        if buf == '':
            return -1                           # Reached the end of the file
        
        str_pos = buf.find(string)
        
        if str_pos == -1:
            # Couldn't find the string yet, but the file is not finished
            # Roll back, and continue to search
            f_pos = f_pos + buffer_size - str_len
            continue
        else:
            return f_pos + str_pos


def smooth_angle(x, start = None, end = None):
    """
    Join up the phase angles in a numpy array
    
    Majority of the data between index start and end are to be within +_ pi,
    i.e. the first phase semi-circle.
    
    The input array can the about of numpy.angle() for example, it is expected 
    that the array values are between +_ 2*pi
    """
    
    pi2 = 2*pi
    dx = zeros(len(x)).astype(float)
    dx[1:] = diff(x)
    dx[dx > pi] -= pi2
    dx[dx < -pi] += pi2
    # Second pass to remove those with +2pi to - 2pi changes
    dx[dx > pi] -= pi2
    dx[dx < -pi] += pi2
    
    # Put back the first element
    dx[0] = x[0]
    
    if (start is None) and (end is None):
        return dx.cumsum()
    else:
        output = dx.cumsum()
        npi = round(output[start:end].mean() / pi)
        # print npi
        return output - (npi * pi)





def file_list(base_dir, ext = None, SubDir = False):
    """
    Generate a list containing the absolute path of all of the files matching 
    the selection criteria under the given directory.
    
    If base_dir is given as a file path, return as a single item list.
    """
    base_dir = abspath(base_dir.rstrip(sep))
    
    # Create a list of files
    flist = []
    if isdir(base_dir):
        if SubDir == True:
            for root, dirs, files in walk(base_dir):
                for name in files:
                    f = filePath(abspath(join(root, name)))
                    if ext is not None:
                        if f.ext != ext:
                            continue
                    flist.append(f.full)
        else:
            for fname in listdir(base_dir):
                fname = abspath(join(base_dir, fname))
                if isfile(fname):
                    f = filePath(fname)
                    if ext is not None:
                        if f.ext != ext:
                            continue
                    flist.append(f.full)
    elif isfile(base_dir):
        f = filePath(base_dir)
        if ext is not None:
            if f.ext != ext:
                return []
        return [f.full]
    else:
        raise IOError, "Given path is not a dir"
    
    flist.sort()
    
    return flist



def extrema(x, max = True, min = True, \
            strict = False, withend = True, \
            zero = False, flat = False):
    """
    This function will index the extrema of a given array x.
    
    Options:
        max     If true, will index maxima
        min     If true, will index minima
        strict  If true, will not idex changes to zero gradient
        withend If true, always include x[0] and x[-1]
        Zero    If true, will include points where f(x) = f'(x) = f''(x) = 0
        flat    If true, will include points where f'(x) = f''(x) = 0
    
    This function will return a tuple of extrema indexies and values
    """
    
    x_len = len(x)
    
    # This is the gradient
    
    # f'(x)
    # Clean up the gradient in order to pick out any change of sign
    # and allow the detection of changes to zero gradient
    dx = zeros(x_len)
    dx[1:] = sign(diff(x))
    dx[0] = dx[1]
    
    # f''(x) to pick out the spikes
    # abs(f''(x)) = 1 if +ve <-> 0 <-> -ve
    # abs(f''(x)) = 2 if +ve <-> -ve
    d2x = zeros(x_len)
    d2x[:-1] = diff(dx)
    
    # See ind = nonzero(d2x > threshold)[0]
    if max and min:
        d2x = abs(d2x)
    elif max:
        d2x = -d2x
    
    # Take care of the two ends
    if withend:
        d2x[0] = 2
        d2x[-1] = 2
    
    # define the threshold for whether to pick out changes to zero gradient
    threshold = 0
    if strict:
        threshold = 1
    
    # Sift out the list of extremas
    ind = nonzero(d2x > threshold)[0]
    
    return ind, x[ind]


def zerocrossings(x):
    """
    This function will return the number of zero crossings in the given
    1-D numpy array.
    
    This function will pick out both types of zero crossings:
        Type I: +ve -> -ve, or -ve -> +ve
        Type II: +ve -> zero(s) -> -ve, or -ve -> zero(s) -> +ve
    """
    
    dsx = diff(sign(x))
    
    # Pick out +ve <-> -ve type (I) zero crossings
    ind = nonzero(abs(dsx) > 1)[0]
    # zero crossing count
    count = len(ind)
    
    # Pick out +ve <- 0 -> -ve type (II) zero crossings
    idsx = dsx.copy()
    ## Remove type I zero crossings from list
    idsx[ind] = 0
    
    ## Need two consecutive sign change to make a zero crossing
    ### Remove zeros in the list
    idsx = idsx[idsx.nonzero()[0]]
    
    cidsx = abs(idsx.cumsum()) - 1
    ind = cidsx.nonzero()[0]
    count += len(ind)
    
    return count




def resample(time, data, rate=False):
    """
    Resample the data series into the given sampling rate, this is 
    implemented using the cubic spline interpolation. 
    
    No filtering is done in here, anti-aliasing filters maybe necessary if 
    down-sampling.
    
    Will try to find the next highest sampling rate by default. The 
    resampled data will always align at time 0, and never exceed the 
    duration of the given data.
    
    The sampling rate will come in multiples of 1, 2, or 5Hz
    
    N.B. This funtion was initially created to harmonise the sampling rates 
    between FE simulations, they often have different time steps depends on the
    simulation parameters.
    """
    
    # Get an idea of the input datatype.
    ttype = time.dtype
    dtype = data.dtype
    
    # Get an idea of time step
    t0 = time[0]
    t1 = time[-1]
    duration = t1 - t0
    step = duration / len(time)
    smp_rate = 1 / step
    
    if rate:
        # The sampling rate is supplied, no need to estimate.
        spl_rate = rate
        spl_step = 1 / spl_rate
    else:
        # Find out the exponent of the current sampling rate
        exponent = Decimal(str(smp_rate)).adjusted()
        
        # Remove the exponent
        scale = smp_rate * 10**(0 - exponent)
        
        # make the standard scale slightly larger (1e-5) so numerical 
        # error (rounding error) do not come in to play and force it up
        # to the next sampling scale
        if scale > 5.00005:
            scale = 10
        elif scale > 2.00002:
            scale = 5
        elif scale > 1.00001:
            scale = 2
        else:
            # This really shouldnt happen, but just in case the Decimal
            # function return numbers like 0.123e+45 instead of 1.23e+45 
            scale = 1
            print "Warning!! Unexpected values for scale evaluation!" + \
            'scale variable (' + str(scale) + ') should be greater than 1.'
        
        # This is what the sampling rate should be
        spl_rate = scale * 10**exponent
        spl_step = 1.0 / spl_rate
        
    # Make sure it always starts later
    spl_i0 = int(ceil(t0 / spl_step))
    
    # Make sure it always finish earlier
    spl_i1 = int(floor(t1 / spl_step))
    
    # Work out maximum number of data points at the given sampling rate
    spl_length = abs(spl_i1 - spl_i0)
    
    spl_time = linspace(spl_i0 * spl_step, spl_i1 * spl_step, spl_length)
    
    # Fit the cubic spline line
    spline = splrep(time, data, k=3)
    
    spl_data = splev(spl_time, spline)
    
    return [spl_time.astype(ttype), spl_data.astype(dtype)]



def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None):
    """
    Local substitution of the scipy.signal.filtfilt funtion.
    
    By defult, scipy.signal.filtfilt will pad the data with its end point 
    values. This can be problematic for 'noisy' data, where the end point 
    values can be significantly different from the local average values.
    
    This version will manually force the end points inplace to be the local 
    average values. The longest of the filter coefficients are taken as the 
    average length.
    """
    l = max((len(a), len(b)))
    
    # Modify the end points to force a specific padding value.
    # This will aviod spikes at end points for 'noisy' data
    x[0] = x[:l].mean()
    x[-1] = x[-l:].mean()
    
    return scipy_filtfilt(b, a, x, axis=axis, padtype=padtype, padlen=padlen)


def spline_resize(x, n, l = 0.005, window='hamming', order = 3):
    """
    Resize the ndarray x in to n elements long.
    
    Default order of spline interpolation is cubic.
    
    If upsampling (n > len(x)), (cubic) spline interpolation will be used.
    
    If downsampling, two pass anti-aliasing Type I FIR filter will be applied, 
    once forward and once reverse to null the group delay, then (cubic) spline 
    interpolation will be used to resample the data.
    
    l           Length of the FIR filter, default to len(x) / 200, mimimum 3
    window      Window method to generate the FIR filter
    
    Window options:
        boxcar
        triang
        blackman
        hamming
        hann
        bartlett
        flattop
        parzen
        bohman
        blackmanharris
        nuttall
        barthann
        kaiser (needs beta)
        gaussian (needs std)
        general_gaussian (needs power, width)
        slepian (needs width)
        chebwin (needs attenuation)
        
    """
    
    m = len(x)
    if m == n:
        return x            # Do nothing, no need to resize
    
    
    r = float(n) / m
    y = x.copy()
    if r > 1:               # Up sampling, no need to filter
        # y = x.copy()
        pass
    else:                   # r < 1, Down sampling, apply anti-aliasing filter
        
        if type(l) is float: l = int(round(len(x) * l))     # FIR filter length
        
        if l%2 == 0: l += 1 # l must be odd for Type I filter
        
        if l < 3: l = 3 # Make sure l is at least three
        
        # a = [1.]
        b = firwin(l, r, window=window)
        
        y = filtfilt(b, [1.], y)
    
    # Cubic spline interpolation
    s = InterpolatedUnivariateSpline(arange(m), y, k=order)
    return s(linspace(0, m-1, n))





def linearFunc(x0, y0, x1, y1):
    """
    Return a linear function given two points in space.
    """
    x0 = float(x0)
    y0 = float(y0)
    x1 = float(x1)
    y1 = float(y1)
    k = (y1 - y0) / (x1 - x0)
    b = ((y0 - k*x0) + (y1 - k*x1)) / 2
    return lambda x: k * x + b
    


def logFunc(x0, y0, x1, y1, base=10):
    """
    Return a logarithmic function mapped on a arb. linear scale, given two points in space.
    
    Useful for mapping logarithmic scale to array index, for example.
    """
    x0 = log(x0)/log(base)
    x1 = log(x1)/log(base)
    
    k = (y1 - y0) / (x1 - x0)
    b = ((y0 - k*x0) + (y1 - k*x1)) / 2
    
    return lambda x: k * log(x)/log(base) + b
    

def expFunc(x0, y0, x1, y1, base=10):
    """
    Return a exponential function mapped on a arb. linear scale, given two points in space.
    
    Useful for mapping array inedx to logarithmic scale, for example.
    """
    y0 = log(y0)/log(base)
    y1 = log(y1)/log(base)
    
    k = (y1 - y0) / (x1 - x0)
    b = ((y0 - k*x0) + (y1 - k*x1)) / 2
    return lambda x: base ** ( k * x + b )
    

def linearChk(ary, axis = -1, strict = False, debug = False):
    """
    This will check whether the input array, along the given axis 
    (default is a 1-D array) has a equal increments between elements.
    
    Return true if it is linear, else, return the maximum difference.
    
    Will compare to numerical accuracy if strict option == True, else it
    is assume linear if none of the element is deviate more than one 
    increment away. String representations of numbers often have lower 
    precision than binary representation.
    
    """
    
    # Select the correct axis
    if axis == -1:
        tst = ary
    else:
        tst = ary[axis]
        
    
    # Generate the linear reference array
    linear = linspace(tst[0], tst[-1], len(tst))
    
    # Compute the maximum deviation from linear expectation
    diff = max(abs(linear - tst))
    
    
    if diff == 0: # Excellent! Linear to within numerical accuracy
        return True
        
    elif strict == True:
        return diff # Strict rules apply
        
    else: # Relax the rules a little
        if diff < abs(linear[1] - linear[0]):
            return True # none of the elements overlap
        else:
            return diff # Even relaxed rules cant save this one



def logChk(ary, axis = -1, strict = False, debug = False):
    """
    This will check whether the input array, along the given axis 
    (default is a 1-D array) has a equal increments between elements in log scale.
    
    Return true if it is linear, else, return the maximum difference.
    
    Will compare to numerical accuracy if strict option == True, else it
    is assume linear if none of the element is deviate more than one 
    increment away. String representations of numbers often have lower 
    precision than binary representation.
    
    """
    
    return linearChk(log(ary[axis], strict=strict, debug=debug))


def unitPrefix(num):
    """
    Work out the appropriate unit prefix for a given number
    
    [fixed num, long hand, short hand, exponent] = unitPrefix(number)
    
    
    >>> unit = "eV"
    >>> number = 20e7
    >>> prefix = unitPrefix(number)
    >>> print str(number / ( 10 ** prefix[3] )) + prefix[2] + unit 
    200.0MeV
    >>> print str(prefix[0]) + prefix[2] + unit 
    200.0MeV
    
    
    Please note, the micron sign is not given in Unicode, but replaced by "u" instead
    
    If the given number is beyond the scale listed below, the same number will be return.
    
    yotta   Y   e+24
    zetta   Z   e+21
    exa     E   e+18
    peta    P   e+15
    tera    T   e+12
    giga    G   e+9
    mega    M   e+6
    kilo    k   e+3
    milli   m   e-3
    micro   u   e-6
    nano    n   e-9
    pico    p   e-12
    femto   f   e-15
    atto    a   e-18
    zepto   z   e-21
    yocto   y   e-24
    """
    
    prefixLst = []
    prefixLst.append(['', '', 0])
    prefixLst.append(['kilo', 'k', 3])
    prefixLst.append(['mega', 'M', 6])
    prefixLst.append(['giga', 'G', 9])
    prefixLst.append(['tera', 'T', 12])
    prefixLst.append(['peta', 'P', 15])
    prefixLst.append(['exa', 'E', 18])
    prefixLst.append(['zetta', 'Z', 21])
    prefixLst.append(['yotta', 'Y', 24])
    prefixLst.append(['yocto', 'y', -24])
    prefixLst.append(['zepto', 'z', -21])
    prefixLst.append(['atto', 'a', -18])
    prefixLst.append(['femto', 'f', -15])
    prefixLst.append(['pico', 'p', -12])
    prefixLst.append(['nano', 'n', -9])
    prefixLst.append(['micro', 'u', -6])
    prefixLst.append(['milli', 'm', -3])
    
    # Because scientific notations are decimal numbers
    # Get the exponent
    exponent = Decimal(str(num)).adjusted()
    
    # Get the nearest engineering scale
    scale = int((exponent - mod(exponent, 3))/3)
    
    # This is beyond the range
    if abs(scale) > 8:
        scale = 0
    
    # Prepare for the output
    unit = prefixLst[scale]
    num = num / ( 10 ** unit[2] )
    
    return [num, unit[0], unit[1], unit[2]]


def engUnit(num, unit="", sigfig=None):
    """
    Return a print string for num with approprate engineering unit
    
    """
    
    prefix = unitPrefix(num)
    
    if sigfig is None:
        return str(prefix[0]) + prefix[2] + unit
    
    num = prefix[0]
    scale = 10 ** Decimal(str(num)).adjusted()      # scaling to make 1.2345e67
    #scale = 10 ** floor(log10(num))                # log10 wont work for 0
    num = round((num / scale), sigfig -1) * scale   # round to the correct sigfig
    num = '%(num)- 0.15f' % {'num' : num}           # Limit to 18 sigfig for safety
    num = num[:sigfig+2].rstrip('.')                # 1 decimal point, 1 sign
    
    engstr = num + prefix[2] + unit
    
    return engstr.strip()


def zerostretches(x, zero_threshold = 1e-6, zero_runlength = 3, atol = 1e-12):
    """
    Running zero detection.
    
    Given numpy ndarray x, find stretches of zero values, and return 
    a tuple of indexes indicating the beginning and ending of the 
    stretches. 
    
    options:
        zero_threshold = 1e-6       Magnitude threshold (relative to the
                                    max signal magnitude) at which a 
                                    Least number of consecutive zeros 
                                    before it is considered a running 
                                    strip. 
        
        zero_runlength = 3          Minimum consecutive zeros before it 
                                    is consider as a stretch, minium 
                                    necessary for stability. Cubic 
                                    splines need 3pts to come to 
                                    complete stop. 
    
        atol = 1e-12                Absolute threshold for zero finding.
        
     This is NOT an attempt to find zero crossing.
    
    ###########################################################################
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21]   index  
    ###########################################################################
     [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]   z
     [0 -1, 1,-1, 0, 1, 0, 0, 0, 0,-1, 0, 0, 1, 0,-1, 0, 0, 0, 0, 1, 0]   dz
    ###########################################################################
     [   1,    3,                  10,            15                  ]   Begin
     [      2,       5,                     13,                  20   ]   End
    ###########################################################################
    
    Returns:
        (array([ 1,  3, 10, 15]), array([ 2,  5, 13, 20]))
    """
    #zero_threshold = 1e-6
    #zero_runlength = 0
    #from numpy import array
    #x = array([-0.6,0,-0.2,0,0,-0.1,-0.7,0.2,0.2,0.6,0,0,0,0.3,-0.8,0,0,0,0,0,0.1,0.8])
    # x = [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    #from numpy import where
    #from numpy import abs, ones, zeros
    
    
    # Do the abs, just incase x is complex
    m = abs(x)
    
    # Map out all the zeros
    z = ones(len(m), int)
    z[where(m < max(max(m)*zero_threshold, atol))[0]] = 0
    
    dz = zeros(len(m), int)
    dz[1:] = diff(z)
    
    # Manually add zero starts
    if z[0] == 0:
        dz[0] = -1
    
    start = where(dz == -1)[0]
    end = where(dz == 1)[0]
    
    # Manually add zero ends
    if z[-1] == 0:
        end = append(end, len(z))
    
    # Filter out those below the runlength threshold
    z_th = where((end - start) >= zero_runlength)[0]
    
    start = start[z_th]
    end = end[z_th]
    
    return (start, end)



def resolve_complex_collusion(knots, asint = True, threshold = 0):
    """
    This function will search for items with identical real part (as int)
    in a complex array, reduce them to an item with the same int real 
    part, and averaged imaginary value.
    
    This is useful when a complex array is used to define the knots for 
    interpolation, such that the real part indicates position, and the 
    imaginary part represent the corresponding value. Some interpolation
    algorithms don't like knots with identical positions, or even not 
    having the knots sorted by the order of their position.
    
    Option:
    asint       If true, the position is assumed to have int value, and 
                will be detected and tidied up as int. 
    
    threshold   How close should the positions be before they are 
                considered identical.
    
    #######################################################################
    # 0   1   2   3   4   5   6   7   8  15  16                 # Results #
    #######################################################################
    # 0   1   2   2   2   3   4   5   6   6   7   8  15  16     # idx     #
    #######################################################################
    """
    
    knots = sort(knots)
    
    if asint:
        # Cast all index into int
        idx = round(real(knots)).astype(int)        
        didx = ones(len(idx))
        # Knot index should at least advance by 1
        didx[1:] = diff(idx)
        pos = where(didx < 1)[0]
        # Create a list of colliding positions
        # Only select those with didx just became zero (i.e. the first collusion)
        collusions = idx[pos[nonzero(didx[pos-1])[0]]]
        # Walk through the list of collusions
        for i in collusions:
            # Select those with same collusion index
            pos = where(idx == i)[0]
            # Get the averaged value
            val = imag(knots[pos]).mean()
            # Write them as -1 to signify invalid data
            knots[pos] = -1
            # Insert the averaged 
            knots[pos[0]] = i + 1j*val
        
    else:
        idx = real(knots)
        didx = ones(len(idx))
        # Knot index should at least advance by the amount defined by threshold
        didx[1:] = (diff(idx) > threshold).astype(int)
        # Create a list of colliding positions
        collusions = zerostretches(didx, zero_runlength = 1)
        # Walk through the list of collusions
        for i in zip(collusions[0],collusions[1]):
            begin = i[0] - 1
            end = i[1]
            # calculate the avg position
            pos = idx[begin:end].mean()
            # calculate the avg value
            val = imag(knots)[idx[begin:end]].mean()
            # Write them as -1 to signify invalid data
            knots[begin:end] = -1
            # Insert the averaged 
            knots[begin] = pos + 1j*val
        
    return sort(knots)[len(where(knots == -1)[0]):]
    


def gettypecode(bytelen, dtype):
    """
    Given the bit length and desired data type, work out the approprate type code.
    
    bytelen     <int>  Number of bytes
    dtype       {'int'|'Uint'|'float'}
    """
    #                           
    #Format   C Type             Python
    #  x      pad byte           no value
    #  c      char               string of length 1
    #  b      signed char        integer
    #  B      unsigned char      integer
    #  h      short              integer
    #  H      unsigned short     integer
    #  i      int                integer
    #  I      unsigned int       long
    #  l      long               integer
    #  L      unsigned long      long
    #  q      long long          long  
    #  Q      unsigned long long long  
    #  f      float              float
    #  d      double             float
    #  s      char[]             string
    #  p      char[]             string
    #  P      void *             integer
    #                           
    # Numpy takes:                      
    # ['c', 'b', 'u', 'i', 'l', 'f', 'd', 'F', 'D', 'O']    
    #       
    
    if dtype == 'int' or dtype is type(int()):
        lst = ['b','h','i','l', 'q']
    elif dtype == 'Uint':
        lst = ['B','H','I','L', 'Q']
    elif dtype == 'float' or dtype is type(float()):
        lst = ['f','d']
    #elif dtype == 'complex':
    #    lst = ['F','D']
    else:
        raise ValueError('Specified dtype: ' + str(dtype) + ' is not recognised.')
    
    for typecode in lst:
        if calcsize(typecode) == bytelen:
            return typecode
    
    # Unable to find a match
    raise ValueError('Unable to find a suitable typecode for the speficied dtype (' + str(dtype) + ') and byte length (' + str(bytelen) + ').')
    


def quantise(ary, threshold = 1e-8):
    """
    It will attempt to find the quantisation step size of a float array
    
    Quantisation error should not be bigger than the fraction of quantisation step
    size given by the threshold.
    
    Return quantisation step if a reasonable value is found
    """
    
    # Find the smallest non-zero increment
    quantum = abs(diff(ary))
    quanter = quantum[nonzero(quantum)].min()
    
    # See if all increments are multiples of the quanter
    quantum /= quanter          # Attempt quantisation
    quantum -= floor(quantum)   # Remove the whole number multiples
    quantum /= threshold        # Inflate the fractional error by threshold amount
    offlimit = squeeze(nonzero(quantum > quanter))
    
    if len(offlimit) is not 0:
        raise QuantsationError
    else:
        return quanter
    


def decimate2(x, n=None, axis=-1, window='hamming'):
    """
    Similar to scipy.signal.decimate, but fixed decimation factor to q = 2, and
    force two pass FIR filter to ensure zero group delay.
    
    The FIR is generated by the windowing method. Type I filter is always used
    to have band-stop charasterics on the Nyquist
    
    If filter length n is not specified, then n is choosen to be two hundred 
    times smaller than len(x).
    
    Example:
    
        halved = decimate2(x)
    """
    
    if n is None:
        n = int(round(len(x) / 200.0))
    
    if n < 3:
        n = 3
    
    # n must be odd for Type I filter
    if n%2 == 0:
        n += 1
    
    b = firwin(n, 0.5, window = window)
    a = 1.
    
    y = lfilter(b, a, x, axis = axis)           # First pass
    y = lfilter(b, a, y[::-1], axis = axis)     # Reverse pass
    y = y[::-1]                                 # Reverse the result
    
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, 2)
    
    return y[sl]


def timestamp():
    """
    st = timestamp()
    
    '2014-12-15 01:21:05'
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def datestamp():
    """
    st = datestamp()
    
    '2014-12-15'
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')


########################################################################
# This morlet function has been accepted into the Scipy package.
#
# Use the following instead:
#    from scipy.signal.wavelets import morlet
#
########################################################################
#def morlet(M, w = 5.0, s = 1.0, cplt = True):
#   """
#    Returns the complex Morlet wavelet with length of M
#
#    Inputs:
#        M   Length of the wavelet
#        w   Omega0
#        s   Scaling factor, windowed from -s*2*pi to +s*2*pi
#        cplt    Use the complete or standard version
#
#    The standard version:
#    pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
#
#    Often referred to as simply the Morlet wavelet in many text,
#        also commonly use in practice. However, this simplified version
#        can cause admissibility problem at low w.
#
#    The complete version:
#    pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
#
#    Complete version of the Morlet wavelet, with the correction
#        term to improve admissibility. For w is greater than 5, the
#        correction term will be negligible.
#
#    The energy of the return wavelet is NOT normalised according to s.
#
#    NB: The fundamental frequency of this wavelet in Hz is given by:
#
#        f = 2*s*w*r / M
#
#        r - Sampling Rate
#
#    """
#
#    correction = exp(-0.5*(w**2))
#    c1 = 1j*w
#    s *= 2*pi
#
#    xlist = linspace(-s, s, M)
#    output = zeros(M).astype('F')
#
##   if cplt == True:
##       for i in range(M):
##           x = xlist[i]
##           output[i] = (exp(c1*x) - correction) * exp(-0.5*(x**2))
##   else:
##       for i in range(M):
##           x = xlist[i]
##           output[i] = exp(c1*x) * exp(-0.5*(x**2))
#
#    if cplt == True:
#        output = (exp(c1*xlist) - correction) * exp(-0.5*(xlist**2))
#    else:
#        output = exp(c1*xlist) * exp(-0.5*(xlist**2))
#
#    output *= pi**-0.25
#
#    return output
