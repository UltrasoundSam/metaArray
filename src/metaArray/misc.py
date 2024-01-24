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
