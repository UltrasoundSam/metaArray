#       constants.py
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
'''
Tests the core functionality of the metaArray class
'''

import numpy as np
import numpy.typing as npt
import pytest

import metaArray as ma


@pytest.fixture
def magic_square() -> npt.NDArray[np.float_]:
    '''Create a magic square for data to pass into metaArray

    Creates a 2d array of with length 5 that each column/row sums
    to the same value (65).
    '''
    N = 5
    magic_square = np.zeros((N, N), dtype=int)

    n = 1
    i, j = 0, N//2

    while n <= N**2:
        magic_square[i, j] = n
        n += 1
        newi, newj = (i-1) % N, (j+1) % N
        if magic_square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj

    return magic_square


@pytest.fixture
def meta_square(magic_square) -> ma.metaArray:
    '''Create a magic square for data to pass into metaArray

    Creates a 2d array of with length 5 that each column/row sums
    to the same value (65).
    '''
    # Define meta info
    metainfo = {'name': 'Magic Square',
                'unit': 'V',
                'label': 'Amplitude',
                'resample': False,
                }

    # Define ranges now
    range = {'begin': [0., -5e-3],
             'end': [15., 25e-3],
             'unit': ['s', 'm'],
             'label': ['Time', 'Distance']
             }

    metainfo['range'] = range

    return ma.metaArray(magic_square, dtype=int, info=metainfo)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_sum(meta_square: ma.metaArray, axis: int):
    '''Tests sum method'''
    sum = meta_square.sum(axis=axis)

    if axis is not None:
        # If we define an axis - as it is a magic square, we know
        # that all the rows/columns sum to 65.
        assert np.array_equal(sum, 65*np.ones(5, dtype='int'))
    else:
        # Haven't passed any value so want to sum the whole array
        assert sum == 65 * 5


