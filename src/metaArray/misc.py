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
import typing
import os


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
