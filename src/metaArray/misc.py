# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 2023 10:52

@author: samhill

File for controlling csv file reading
"""

import numpy as np
import scipy as sp
import numpy.typing as npt
import typing
import decimal
import struct
import datetime
import os

from scipy.signal import filtfilt as scipy_filtfilt

from metaArray.constants import GaussFWHM


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


class dirPath:
    """
    DirPath object, provide various information about the dir path

    currently provide the following:

    self.full       The full path, i.e. the input string
    self.name       The dir name
    self.read       Whether it is readable
    self.write      Whether it is writable
    self.exist      Whether it already exists
    """

    def __init__(self, path: str) -> None:

        self.full = path = os.path.abspath(path.strip())
        self.name = os.path.basename(path)
        self.read = os.access(path, os.R_OK)
        self.write = os.access(path, os.W_OK)
        self.exist = os.access(path, os.F_OK)

    def __repr__(self) -> str:
        """
        Text representation of the object
        """
        desc = ''
        desc += 'Full: ' + self.full + os.linesep
        desc += 'Name: ' + self.name + os.linesep
        desc += 'Read: ' + str(self.read) + os.linesep
        desc += 'Write: ' + str(self.write) + os.linesep
        desc += 'exist: ' + str(self.exist) + os.linesep

        return desc


class filePath:
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

    def __init__(self, file_path: str) -> None:

        self.full = path = os.path.abspath(file_path.strip())
        self.read = os.access(path, os.R_OK)
        self.write = os.access(path, os.W_OK)
        self.execute = os.access(path, os.X_OK)
        self.exist = os.access(path, os.F_OK)
        self.baseDir, fname = os.path.split(path)

        # Modify the behaviour if the file does not exit, check if the directory
        # is writable instead
        if not self.exist:
            self.write = os.access(self.baseDir, os.W_OK)

        self.name, self.ext = os.path.splitext(fname)

    def __repr__(self) -> str:
        """
        Text representation of the object
        """
        desc = ''
        desc += 'Full: ' + self.full + os.linesep
        desc += 'Name: ' + self.name + os.linesep
        desc += 'Read: ' + str(self.read) + os.linesep
        desc += 'Write: ' + str(self.write) + os.linesep
        desc += 'Execute: ' + str(self.execute) + os.linesep
        desc += 'Exist: ' + str(self.exist) + os.linesep
        desc += 'baseDir: ' + self.baseDir + os.linesep
        desc += 'extension: ' + self.ext + os.linesep

        return desc


class cplx_trig_func:
    """
    A complex trigonometic function object.

    Not all input variables needs be defined, it will try to work them out.
    """

    def __init__(self, nLambda: float = False, pts: int = False,
                 debug: bool = False, length: float = False,
                 freq: float = False, samp_rate: float = False,
                 dt: float = False) -> None:

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

    def __call__(self) -> npt.NDArray[np.complex_]:
        """
        Retrun the complex trigonometic oscillation function
        """
        # Try to reolve all the parameters from given data
        self.resolve()

        if self.x is None:
            x = np.linspace(0, self.nLambda*2*np.pi, self.pts)
            self.x = x
            self.dx = x[1]

        return np.exp(1j*self.x)

    def __repr__(self) -> str:
        """
        Text representation of the object
        """
        desc = 'This is a complex trigonometic oscillation function object.'
        desc += os.linesep
        desc += 'It has the following attributes:' + os.linesep
        desc += '\tnLambda is: ' + str(self.nLambda) + os.linesep
        desc += '\tpts is: ' + str(self.pts) + os.linesep
        desc += '\tlength is: ' + str(self.length) + os.linesep
        desc += '\tfreq is: ' + str(self.freq) + os.linesep
        desc += '\tsamp_rate is: ' + str(self.samp_rate) + os.linesep
        desc += '\tdt is: ' + str(self.dt) + os.linesep
        desc += '\tdx is: ' + str(self.dx) + os.linesep

        desc += '\tRadian space is '
        if self.x is None:
            desc += 'not defined.' + os.linesep
        else:
            desc += 'defined.' + os.linesep

        desc += os.linesep + '\tDebugging output is '
        if self.debug:
            desc += 'enabled.' + os.linesep
        else:
            desc += 'disabled.' + os.linesep

        desc += '\tParameter definitions has '
        if self.flg_resolved:
            desc += 'been resolved.' + os.linesep
        else:
            desc += 'not been resolved.' + os.linesep
            desc += '\t Use the resolve() method to resolve parameter definitions.' + os.linesep  # noqa: E501

        desc += os.linesep + '\tUse the get_radian_space() method to obtain the radian space values.' + os.linesep  # noqa: E501
        desc += '\tCall the object to obtain the complex trigonometic function array.'  # noqa: E501

        return desc

    def get_radian_space(self) -> npt.NDArray[np.float_]:
        """
        Return the radian space array of the complex trigonometic
        oscillation object.
        """
        # Try to reolve all the parameters from given data
        self.resolve()

        if self.x is not None:
            x = np.linspace(0, self.nLambda*2*np.pi, self.pts)
            self.x = x
            self.dx = x[1]

        return x

    def resolve(self) -> None:
        """
        Try to resolve all the parameters from given data
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
        if pts is False:
            # pts can be obtain from (length x samp_rate) or (length / dt)
            # Need length in either case
            if length is False:
                # length can be obtained from (nLambda * freq)
                if (nLambda is False) or (freq is False):
                    raise InsufficientInput("nLambda (number of wavelengths) and freq (frequency) are needed to define length (duration)")  # noqa: E501
                else:
                    length = float(nLambda) / float(freq)

            # length is now identified
            if samp_rate is not False:
                pts = float(length) * float(samp_rate)
            elif dt is not False:
                pts = float(length) / float(dt)
            else:
                raise InsufficientInput("Either samp_rate (sampling rate) or dt (sampling interval) is needed to define pts (number of points)")  # noqa: E501

        # Identify Tier 1 variables
        if nLambda is False:
            # nLambda can be obtained from (length*freq)
            if freq is False:
                raise InsufficientInput("freq (frequency) is needed to define nLambda (number of wavelengths)")  # noqa: E501

            if length is False:
                # length can be obtained from (pts * dt) or (pts / samp_rate)
                # Need pts in either case
                if pts is False:
                    raise InsufficientInput("pts (number of points) is needed to define length (duration)")  # noqa: E501

                if dt is not False:
                    length = int(np.round(pts)) * float(dt)
                elif samp_rate is not False:
                    length = int(np.round(pts)) / float(samp_rate)
                else:
                    raise InsufficientInput("Either samp_rate (sampling rate) or dt (sampling interval) is needed to define length (duration)")  # noqa: E501

            nLambda = float(length) * float(freq)

        # All the Tier 1 variables are solved
        # Solve the rest of the variables
        if length is False:
            if freq is not False:
                length = nLambda / freq
            elif dt is not False:
                length = pts * dt
            elif samp_rate is not False:
                length = pts / samp_rate
            elif debug:
                print('*** Unable to resolve Tier 2 variable length')

        if freq is False:
            if length is not False:
                freq = nLambda / length
            elif debug:
                print('*** Unable to resolve Tier 2 variable freq')

        if samp_rate is False:
            if length is not False:
                samp_rate = pts / length

        if dt is False:
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


class MotherMorlet:
    """
    A mother morlet object
    """

    def __init__(self, M: int, w: float = 5.0, s: float = 1.0,
                 A: float = 1.0, complete: bool = True,
                 name: str = None, same_len: bool = False,
                 unit: str = 'scale') -> None:

        """
        M           Length of the mother wavelet
        w           Omega0
        s           Scaling factor, windowed from -s*2*pi to +s*2*pi
        A           Amplitude of the mother wavelet
        complete    Use the complete or standard version
        name        Name of the mother wavelet
        same_len    Should daughter wavelet have the same M
                    (but different s) as the mother wavelet.
        unit        Unit description for the scale (e.g. self(10) === 10 Hz)
        """

        self.length = int(np.round(M))
        self.Omega0 = w
        self.window = s
        self.Amplitude = A
        self.unit = unit

        # FWHM of a Gaussian is 2 * ( 2 * ln(2) )**0.5 = 2.3548200450309493
        # The fundamental scale is twice of that.
        # Scale of a trig function is 2*pi / w
        twopi = 2 * np.pi

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

    def __call__(self, scale: float) -> npt.NDArray[np.complex_]:
        """
        Scale relative to the mother wavelet

        Scale in here is not the same as the 's' in morlet(M, w, s)
        """
        amp = self.Amplitude / ((np.abs(scale))**0.5)

        if self.flg_same_len:
            M = self.length
            s = self.window / scale
        else:
            M = int(np.round(self.length * scale))
            s = self.window

        return amp * sp.signal.wavelets.morlet(M, w=self.Omega0, s=s,
                                               complete=self.flg_complete)

    def __repr__(self) -> str:
        if self.flg_complete:
            desc = f"Morlet (low oscillation corrected) mother wavelet of Omega0 = {self.Omega0:1.2f}. {os.linesep}"  # noqa: E501
        else:
            desc = f"Morlet mother wavelet of Omega0 = {self.Omega0:1.2f}."
            desc += os.linesep

        if self.name is not None:
            desc += f"\t {self.name}{os.linesep}"

        desc += f"\t Number of points: {self.length}{os.linesep}"
        desc += f"\t Window range: -{self.s:1.2f}𝜋 to {self.s:1.2}𝜋{os.linesep}"
        desc += f"\t Unit Amplitude: {self.Amplitude}{os.linesep}"
        desc += f"\t Has a characteristic scale of: {self.scale}*2*𝜋"
        desc += os.linesep
        desc += f"\t Daughter wavelets have a scale in the unit of: {self.unit}s" + os.linesep  # noqa: E501
        if self.flg_same_len:
            desc += "\t Constant number of points for daugter wavelets."
        else:
            desc += "\t Constant window range for daugter wavelets."

        return desc


def file_list(base_dir: str, ext: str = None,
              sub_dir: bool = False) -> list[str]:
    """
    Generate a list containing the absolute path of all of the files matching
    the selection criteria under the given directory.

    If base_dir is given as a file path, return as a single item list.
    """
    base_dir = os.path.abspath(base_dir.rstrip(os.sep))

    # Create a list of files
    flist = []
    if os.path.isdir(base_dir):
        if sub_dir is True:
            for root, dirs, files in os.walk(base_dir):
                for name in files:
                    f = filePath(os.path.abspath(os.path.join(root, name)))
                    if ext is not None:
                        if f.ext != ext:
                            continue
                    flist.append(f.full)
        else:
            for fname in os.listdir(base_dir):
                fname = os.path.abspath(os.path.join(base_dir, fname))
                if os.path.isfile(fname):
                    f = filePath(fname)
                    if ext is not None:
                        if f.ext != ext:
                            continue
                    flist.append(f.full)
    elif os.path.isfile(base_dir):
        f = filePath(base_dir)
        if ext is not None:
            if f.ext != ext:
                return []
        return [f.full]
    else:
        raise IOError("Given path is not a dir")

    flist.sort()

    return flist


def extrema(x: npt.ArrayLike, max: bool = True, min: bool = True,
            strict: bool = False,
            withend: bool = True) -> tuple[int, float]:
    """
    This function will index the extrema of a given array x.

    Options:
        max     If true, will index maxima
        min     If true, will index minima
        strict  If true, will not idex changes to zero gradient
        withend If true, always include x[0] and x[-1]

    This function will return a tuple of extrema indexies and values
    """

    x_len = len(x)

    # This is the gradient

    # f'(x)
    # Clean up the gradient in order to pick out any change of sign
    # and allow the detection of changes to zero gradient
    dx = np.zeros(x_len)
    dx[1:] = np.sign(np.diff(x))
    dx[0] = dx[1]

    # f''(x) to pick out the spikes
    # abs(f''(x)) = 1 if +ve <-> 0 <-> -ve
    # abs(f''(x)) = 2 if +ve <-> -ve
    d2x = np.zeros(x_len)
    d2x[:-1] = np.diff(dx)

    # See ind = nonzero(d2x > threshold)[0]
    if max and min:
        d2x = np.abs(d2x)
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
    ind = np.nonzero(d2x > threshold)[0]

    return ind, x[ind]


def zerocrossings(x: npt.ArrayLike) -> int:
    """
    This function will return the number of zero crossings in the given
    1-D numpy array.

    This function will pick out both types of zero crossings:
        Type I: +ve -> -ve, or -ve -> +ve
        Type II: +ve -> zero(s) -> -ve, or -ve -> zero(s) -> +ve
    """

    dsx = np.diff(np.sign(x))

    # Pick out +ve <-> -ve type (I) zero crossings
    ind = np.nonzero(np.abs(dsx) > 1)[0]
    # zero crossing count
    count = len(ind)

    # Pick out +ve <- 0 -> -ve type (II) zero crossings
    idsx = dsx.copy()
    # Remove type I zero crossings from list
    idsx[ind] = 0

    # Need two consecutive sign change to make a zero crossing
    # Remove zeros in the list
    idsx = idsx[idsx.nonzero()[0]]

    cidsx = np.abs(idsx.cumsum()) - 1
    ind = cidsx.nonzero()[0]
    count += len(ind)

    return count


def resample(time: npt.ArrayLike, data: npt.ArrayLike,
             rate=False) -> tuple[npt.NDArray, npt.NDArray]:
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
        exponent = decimal.Decimal(str(smp_rate)).adjusted()

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
            print(f"Warning!! Unexpected values for scale evaluation! \
                  scale variable ({scale}) should be greater than 1.")

        # This is what the sampling rate should be
        spl_rate = scale * 10**exponent
        spl_step = 1.0 / spl_rate

    # Make sure it always starts later
    spl_i0 = int(np.ceil(t0 / spl_step))

    # Make sure it always finish earlier
    spl_i1 = int(np.floor(t1 / spl_step))

    # Work out maximum number of data points at the given sampling rate
    spl_length = np.abs(spl_i1 - spl_i0)

    spl_time = np.linspace(spl_i0 * spl_step, spl_i1 * spl_step, spl_length)

    # Fit the cubic spline line
    spline = sp.interpolate.splrep(time, data, k=3)

    spl_data = sp.interpolate.splev(spl_time, spline)

    return [spl_time.astype(ttype), spl_data.astype(dtype)]


def filtfilt(b: npt.ArrayLike, a: npt.ArrayLike, x: npt.ArrayLike,
             axis: int = -1, padtype: str = 'odd',
             padlen: int = None) -> npt.NDArray[np.float_]:
    """
    Local substitution of the scipy.signal.filtfilt funtion.

    By default, scipy.signal.filtfilt will pad the data with its end point
    values. This can be problematic for 'noisy' data, where the end point
    values can be significantly different from the local average values.

    This version will manually force the end points inplace to be the local
    average values. The longest of the filter coefficients are taken as the
    average length.
    """
    length = max((len(a), len(b)))

    # Modify the end points to force a specific padding value.
    # This will avoid spikes at end points for 'noisy' data
    x[0] = x[:length].mean()
    x[-1] = x[-length:].mean()

    return scipy_filtfilt(b, a, x, axis=axis, padtype=padtype, padlen=padlen)


def spline_resize(x: npt.ArrayLike, n: int,
                  length: typing.Union[float, int] = 0.005,
                  window: str = 'hamming',
                  order: int = 3) -> npt.NDArray:
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
        y = x.copy()
    else:                   # r < 1, Down sampling, apply anti-aliasing filter

        if type(length) is float:
            length = int(np.round(len(x) * length))     # FIR filter length

        if length % 2 == 0:
            length += 1  # l must be odd for Type I filter

        if length < 3:
            length = 3   # Make sure l is at least three

        # a = [1.]
        b = sp.signal.firwin(length, r, window=window)

        y = filtfilt(b, [1.], y)

    # Cubic spline interpolation
    s = sp.interpolate.InterpolatedUnivariateSpline(np.arange(m), y, k=order)
    return s(np.linspace(0, m-1, n))


def unit_prefix(num: typing.Union[int, float]) -> tuple[float, str, str, int]:
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

    If the given number is beyond the scale listed below,
    the same number will be return.

    yotta   Y   e+24
    zetta   Z   e+21
    exa     E   e+18
    peta    P   e+15
    tera    T   e+12
    giga    G   e+9
    mega    M   e+6
    kilo    k   e+3
    milli   m   e-3
    micro   μ   e-6
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
    prefixLst.append(['micro', 'μ', -6])
    prefixLst.append(['milli', 'm', -3])

    # Because scientific notations are decimal numbers
    # Get the exponent
    exponent = decimal.Decimal(str(num)).adjusted()

    # Get the nearest engineering scale
    scale = int((exponent - np.mod(exponent, 3))/3)

    # This is beyond the range
    if np.abs(scale) > 8:
        scale = 0

    # Prepare for the output
    unit = prefixLst[scale]
    num = num / (10 ** unit[2])

    return (num, *unit)


def pretty_unit(unit: str, v0: float,
                v1: float) -> tuple[str, float, float, float]:
    """
    This is a more specialised function for scaling the axis when plotting.

    Given the value range from V0 to V1, return suitable unit prefix and
    scaling factor if unit definition exists.

    Do nothing if unit is not defined. i.e. is None or is ''

    """
    scale = 1       # Default scale is not to modify anything

    if unit is None or unit == '':
        return unit, v0, v1, scale

    # Apply unit prefix if unit is defined
    if np.abs(v0) > np.abs(v1):
        vref = v0
    else:
        vref = v1

    scaled_vref, long_hand, short_hand, exponent = unit_prefix(vref)
    scale = scaled_vref / vref
    v0 *= scale
    v1 *= scale

    unit = ''.join((short_hand, unit))

    return (unit, v0, v1, scale)


def eng_unit(num: float, unit: str = "",
             sigfig: int = None) -> str:
    """
    Return a print string for num with approprate engineering unit

    """

    prefix = unit_prefix(num)

    if sigfig is None:
        return str(prefix[0]) + prefix[2] + unit

    num = prefix[0]
    scale = 10 ** decimal.Decimal(str(num)).adjusted()

    # Round to correct sig fig
    num = np.round((num / scale), sigfig - 1) * scale
    num = f'{num:0.15f}'
    # 1 decimal point, 1 sign
    num = num[:sigfig+2].rstrip('.')

    engstr = num + prefix[2] + unit

    return engstr.strip()


def buffered_search(f: typing.IO, string: typing.Union[str, bytes],
                    start: int = 0,
                    buffer_size: int = 4194304) -> int:
    """
    Given a file object [f], search for the location of [string], from the
    given [start] offset position.

    This function will only read [buffer_size] bytes into memory at a time.

    Return -1 if string not found.
    """
    str_len = len(string)

    if str_len > buffer_size:
        raise BufferError("Target string longer than buffer length.")

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


def smooth_angle(x: npt.ArrayLike, start: int = None,
                 end: int = None) -> npt.NDArray:
    """
    Join up the phase angles in a numpy array

    Majority of the data between index start and end are to be within +_ pi,
    i.e. the first phase semi-circle.

    The input array can the about of numpy.angle() for example, it is expected
    that the array values are between +_ 2*pi
    """

    pi2 = 2*np.pi
    dx = np.zeros(len(x), dtype='float')
    dx[1:] = np.diff(x)
    dx[dx > np.pi] -= pi2
    dx[dx < -np.pi] += pi2
    # Second pass to remove those with +2pi to - 2pi changes
    dx[dx > np.pi] -= pi2
    dx[dx < -np.pi] += pi2

    # Put back the first element
    dx[0] = x[0]

    if (start is None) and (end is None):
        return dx.cumsum()
    else:
        output = dx.cumsum()
        npi = np.round(output[start:end].mean() / np.pi)
        # print npi
        return output - (npi * np.pi)


def zero_stretches(x: npt.ArrayLike, zero_threshold: float = 1e-6,
                   zero_runlength: int = 3,
                   atol: float = 1e-12) -> tuple[npt.NDArray[np.int_],
                                                 npt.NDArray[np.int_]]:
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
    # Do the abs, just incase x is complex
    m = np.abs(x)

    # Map out all the zeros
    z = np.ones(len(m), dtype=int)
    z[np.where(m < max(max(m)*zero_threshold, atol))[0]] = 0

    dz = np.zeros(len(m), dtype=int)
    dz[1:] = np.diff(z)

    # Manually add zero starts
    if z[0] == 0:
        dz[0] = -1

    start = np.where(dz == -1)[0]
    end = np.where(dz == 1)[0]

    # Manually add zero ends
    if z[-1] == 0:
        end = np.append(end, len(z))

    # Filter out those below the runlength threshold
    z_th = np.where((end - start) >= zero_runlength)[0]

    start = start[z_th]
    end = end[z_th]

    return (start, end)


def resolve_complex_collusion(knots: npt.ArrayLike, asint: bool = True,
                              threshold: float = 0) -> npt.NDArray:
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

    knots = np.sort(knots)

    if asint:
        # Cast all index into int
        idx = np.round(np.real(knots)).astype(int)
        didx = np.ones(len(idx))
        # Knot index should at least advance by 1
        didx[1:] = np.diff(idx)
        pos = np.where(didx < 1)[0]
        # Create a list of colliding positions
        # Only select those with didx just became zero
        # (i.e. the first collusion)
        collusions = idx[pos[np.nonzero(didx[pos-1])[0]]]
        # Walk through the list of collusions
        for i in collusions:
            # Select those with same collusion index
            pos = np.where(idx == i)[0]
            # Get the averaged value
            val = np.imag(knots[pos]).mean()
            # Write them as -1 to signify invalid data
            knots[pos] = -1
            # Insert the averaged
            knots[pos[0]] = i + 1j*val

    else:
        idx = np.real(knots)
        didx = np.ones(len(idx))
        # Knot index should at least advance by the amount defined by threshold
        didx[1:] = (np.diff(idx) > threshold).astype(int)
        # Create a list of colliding positions
        collusions = zero_stretches(didx, zero_runlength=1)
        # Walk through the list of collusions
        for i in zip(collusions[0], collusions[1]):
            begin = i[0] - 1
            end = i[1]
            # calculate the avg position
            pos = idx[begin:end].mean()
            # calculate the avg value
            val = np.imag(knots)[idx[begin:end]].mean()
            # Write them as -1 to signify invalid data
            knots[begin:end] = -1
            # Insert the averaged
            knots[begin] = pos + 1j*val

    return np.sort(knots)[len(np.where(knots == -1)[0]):]


def gettypecode(bytelen: int,
                dtype: typing.Union[str, float, int]) -> str:
    """
    Given the bit length and desired data type, work out the
    approprate type code.

    bytelen     <int>  Number of bytes
    dtype       {'int'|'Uint'|'float'}
    """
    #
    # Format   C Type             Python
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

    if dtype == 'int' or dtype is type(int):
        lst = ['b', 'h', 'i', 'l', 'q']
    elif dtype == 'Uint':
        lst = ['B', 'H', 'I', 'L', 'Q']
    elif dtype == 'float' or dtype is type(float):
        lst = ['f', 'd']
    # elif dtype == 'complex':
    #     lst = ['F','D']
    else:
        raise ValueError(f'Specified dtype: {dtype} is not recognised.')

    for typecode in lst:
        if struct.calcsize(typecode) == bytelen:
            return typecode

    # Unable to find a match
    raise ValueError(f'Unable to find a suitable typecode for the speficied \
                     dtype ({dtype}) and byte length ({bytelen}).')


def quantise(ary: npt.ArrayLike, threshold: float = 1e-8) -> float:
    """
    It will attempt to find the quantisation step size of a float array

    Quantisation error should not be bigger than the fraction of
    quantisation step size given by the threshold.

    Return quantisation step if a reasonable value is found
    """

    # Find the smallest non-zero increment
    quantum = np.abs(np.diff(ary))
    quanter = quantum[np.nonzero(quantum)].min()

    # See if all increments are multiples of the quanter
    # Attempt quantisation
    quantum /= quanter
    # Remove the whole number multiples
    quantum -= np.floor(quantum)
    #  Inflate the fractional error by threshold amount
    quantum /= threshold
    offlimit = np.squeeze(np.nonzero(quantum > quanter))

    if len(offlimit) != 0:
        raise QuantsationError
    else:
        return quanter


def decimate2(x: npt.ArrayLike, n: int = None,
              axis: int = -1, window: str = 'hamming') -> npt.NDArray:
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
        n = int(np.round(len(x) / 200.0))

    if n < 3:
        n = 3

    # n must be odd for Type I filter
    if n % 2 == 0:
        n += 1

    b = sp.signal.firwin(n, 0.5, window=window)
    a = 1.

    y = sp.signal.lfilter(b, a, x, axis=axis)           # First pass
    y = sp.signal.lfilter(b, a, y[::-1], axis=axis)     # Reverse pass
    y = y[::-1]                               # Reverse the result

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, 2)

    return y[sl]


def timestamp():
    """
    st = timestamp()

    '2014-12-15 01:21:05'
    """
    return datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')


def datestamp():
    """
    st = datestamp()

    '2014-12-15'
    """
    return datetime.date.today().strftime('%Y-%m-%d')


def linear_func(x0: float, y0: float, x1: float,
                y1: float) -> typing.Callable[[float], float]:
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


def log_func(x0: float, y0: float, x1: float,
             y1: float, base: int = 10) -> typing.Callable[[float], float]:
    """
    Return a logarithmic function mapped on a arb. linear scale, given
    two points in space.

    Useful for mapping logarithmic scale to array index, for example.
    """
    x0 = np.log(x0)/np.log(base)
    x1 = np.log(x1)/np.log(base)

    k = (y1 - y0) / (x1 - x0)
    b = ((y0 - k*x0) + (y1 - k*x1)) / 2

    return lambda x: k * np.log(x)/np.log(base) + b


def exp_func(x0: float, y0: float, x1: float,
             y1: float, base: int = 10) -> typing.Callable[[float], float]:
    """
    Return a exponential function mapped on a arb. linear scale, given
    two points in space.

    Useful for mapping array inedx to logarithmic scale, for example.
    """
    y0 = np.log(y0)/np.log(base)
    y1 = np.log(y1)/np.log(base)

    k = (y1 - y0) / (x1 - x0)
    b = ((y0 - k*x0) + (y1 - k*x1)) / 2
    return lambda x: base ** (k * x + b)


def linearChk(ary: npt.ArrayLike, axis: int = -1,
              strict: bool = False) -> typing.Union[True, float]:
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
    linear = np.linspace(tst[0], tst[-1], len(tst))

    # Compute the maximum deviation from linear expectation
    diff = max(np.abs(linear - tst))

    if diff == 0:
        # Excellent! Linear to within numerical accuracy
        return True
    elif strict is True:
        return diff   # Strict rules apply
    else:   # Relax the rules a little
        if diff < np.abs(linear[1] - linear[0]):
            return True  # none of the elements overlap
        else:
            return diff  # Even relaxed rules cant save this one


def logChk(ary: npt.ArrayLike, axis: int = -1,
           strict: bool = False) -> typing.Union[True, float]:
    """
    This will check whether the input array, along the given axis
    (default is a 1-D array) has a equal increments between elements
    in log scale.

    Return true if it is linear, else, return the maximum difference.

    Will compare to numerical accuracy if strict option == True, else it
    is assume linear if none of the element is deviate more than one
    increment away. String representations of numbers often have lower
    precision than binary representation.

    """
    return linearChk(np.log(ary[axis], strict=strict))
