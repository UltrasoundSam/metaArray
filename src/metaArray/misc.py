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


class mother_morlet:
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


def linearFunc(x0: float, y0: float, x1: float,
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


def logFunc(x0: float, y0: float, x1: float,
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


def expFunc(x0: float, y0: float, x1: float,
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
