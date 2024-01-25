# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 2024 11:29

@author: samhill

metaArray I/O to HDF5 files via the h5py module
"""
import numpy as np
import typing
import h5py

from .core import metaArray, __dict_loop
from .misc import filePath


def to_h5(metAry: metaArray, dest: typing.Union[str, h5py.Group]) -> None:
    """
    Write the metAry into given file path in the HDF5 format

    dest can be string or h5py.Group object.

    If dest is a string, it is interpret as the destination file path.

    If dest is a h5py.Group object, the content will be store
    under the given group.
    """

    # Writing to existing group
    if isinstance(dest, h5py.Group):
        # Write the array
        dest.create_dataset('ndarray', data=metAry.data)
        # Write the meta info
        __dict_loop(metAry.info, dest.create_group('info'))
        return

    # Writing file to path
    path = filePath(dest)

    if not path.write:
        raise ValueError("Unable to write to: " + str(path.full))

    with h5py.File(path.full, 'w') as f:
        f.create_dataset('ndarray', data=metAry.data)     # Write the array
        __dict_loop(metAry.info, f.create_group('info'))  # Write the meta info


def __read_info(h5: h5py.File) -> dict:
    """
    Recursively read items the given h5py object into dict

    If the h5py object is dictionary item is itself an dictrionay object,
    create a HDF5 Datagroup, and store subsequent items within.
    """
    info = {}

    # List of Datagroups in the current h5
    grp_lst = []

    # First loop, store all the simply data, and build the grp_lst
    for key in h5:
        if isinstance(h5[key], h5py.Group):
            grp_lst.append(key)
        else:
            content = h5[key][()]

            if isinstance(content, np.ndarray):
                content = list(content)
            elif isinstance(content, tuple):
                content = list(content)

            info[key] = content

    # Process the grp_lst
    for key in grp_lst:
        info[key] = __read_info(h5[key])

    return info


def from_h5(src: typing.Union[str, h5py.Group]) -> metaArray:
    """
    Read the HDF5 file in the given path, build the metAry accordingly.

    dest can be string or h5py.Group object.

    If src is a string, it is interpret as the destination file path.

    If src is a h5py.Group object, the content will be read from
    the given group.
    """
    # Reading from existing group
    if isinstance(src, h5py.Group):
        ary = src['ndarray'][()]          # Read the array
        info = __read_info(src['info'])   # Read the meta info
        return metaArray(ary, info=info)

    # Reading from file path
    path = filePath(src)

    if not path.exist:
        raise IOError('File ' + str(path.full) + ' does not exist')

    if not path.read:
        raise IOError("Unable to read from: " + str(path.full))

    with h5py.File(path.full, 'r') as f:
        ary = f['ndarray'][()]          # Read the array
        info = __read_info(f['info'])   # Read the meta info

    return metaArray(ary, info=info)
