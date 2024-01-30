# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 2024 15:46

@author: samhill

Tests the core functionality of the metaArray class
"""

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
def meta_info() -> dict:
    '''Defines metainfo fixture
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
    return metainfo


@pytest.fixture
def meta_square(magic_square, meta_info) -> ma.metaArray:
    '''Create a magic square for data to pass into metaArray

    Creates a 2d array of with length 5 that each column/row sums
    to the same value (65).
    '''
    return ma.metaArray(magic_square, dtype=int, info=meta_info)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_sum(meta_square: ma.metaArray, axis: int) -> None:
    '''Tests sum method'''
    sum = meta_square.sum(axis=axis)

    if axis is not None:
        # If we define an axis - as it is a magic square, we know
        # that all the rows/columns sum to 65.
        assert np.array_equal(sum, 65*np.ones(5, dtype='int'))
    else:
        # Haven't passed any value so want to sum the whole array
        assert sum == 65 * 5


@pytest.mark.parametrize("axis", [0, 1, None])
def test_max(meta_square: ma.metaArray, magic_square: npt.NDArray[np.int_],
             axis: int) -> None:
    '''Tests to see whether metaArray behaves the same as numpy'''
    meta_max = meta_square.max(axis=axis)
    np_max = magic_square.max(axis=axis)
    assert np.array_equal(meta_max, np_max)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_min(meta_square: ma.metaArray, magic_square: npt.NDArray[np.int_],
             axis: int) -> None:
    '''Tests to see whether metaArray behaves the same as numpy'''
    meta_max = meta_square.min(axis=axis)
    np_max = magic_square.min(axis=axis)
    assert np.array_equal(meta_max, np_max)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_ptp(meta_square: ma.metaArray, magic_square: npt.NDArray[np.int_],
             axis: int) -> None:
    '''Tests to see whether metaArray behaves the same as numpy'''
    meta_max = meta_square.ptp(axis=axis)
    np_max = magic_square.ptp(axis=axis)
    assert np.array_equal(meta_max, np_max)


def test_metainfo(meta_square: ma.metaArray, meta_info):
    '''Tests to see whether the metainfo is passed without corruption.'''
    # Get meta info from metaArray
    metaInfo = meta_square.copy_info()

    # Remove range from metainfo
    range_info = meta_info.pop('range')

    # Got through each key/val in meta_info and check to see whether it
    # matches with metaInfo
    main_comparision = all((val == metaInfo[key]
                            for (key, val) in meta_info.items()))

    # Test range comparision
    range_comparison = all((val == metaInfo['range'][key]
                            for (key, val) in range_info.items()))

    comparison = (main_comparision, range_comparison)
    assert all(comparison)
