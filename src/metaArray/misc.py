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

'''
This file contain a number a miscellaneous functions

linearChk     Check the input array is linear
unitPrefix    Calculate the unit prefix.

Package dependency:
 PIL
 numpy
 scipy
'''
import numpy as np
import numpy.typing as npt
import typing
import os

from scipy.signal import filtfilt as scipy_filtfilt


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
