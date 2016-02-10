#  drv_h5py.py
#
#  Copyright 2015 Unknown <charley@utc2d>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#

'''
metaArray I/O to HDF5 files via the h5py module

 This is hardly the most storage efficient way of storing small metaArray,
 due to the structrual overhead, but it is an attractive option for data
 exchange.


 Package dependency:

       h5py
'''

import h5py

from metaArray.core import metaArray
from metaArray.misc import filePath

import numpy as np

def __dict_loop(dic, h5):
    """
    Recursively store items in a dictionary object into given h5py object as
    h5py.Dataset.

    If the dictionary item is itself an dictrionay object, create a h5py.Group,
    and store subsequent items within.
    """

    # List of keys to the items in the current dict is itself a dict
    dict_lst = []

    # First loop, store all the simple data, and build the dict_lst
    for key in dic:
        if isinstance(dic[key], dict):
            dict_lst.append(key)
        else:
            h5.create_dataset(key, data=dic[key])

    # Process the dic_lst
    for key in dict_lst:

        # Create a subgroup
        grp = h5.create_group(key)

        __dict_loop(dic[key], grp)

    return


def to_h5(metAry, dest, debug=False):
    """
    Write the metAry into given file path in the HDF5 format

    dest can be string or h5py.Group object.

    If dest is a string, it is interpret as the destination file path.

    If dest is a h5py.Group object, the content will be store under the given group.
    """

    # Writing to existing group
    if isinstance(dest, h5py.Group):
        dest.create_dataset('ndarray', data=metAry.data)    # Write the array
        __dict_loop(metAry.info, dest.create_group('info')) # Write the meta info
        return


    # Writing file to path
    path = filePath(dest)

    if not path.write:
        raise ValueError("Unable to write to: " + str(path.full))

    with h5py.File(path.full, 'w') as f:
        f.create_dataset('ndarray', data=metAry.data)     # Write the array
        __dict_loop(metAry.info, f.create_group('info'))  # Write the meta info

    return



def __read_info(h5):
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



def from_h5(src, debug=False):
    """
    Read the HDF5 file in the given path, build the metAry accordingly.

    dest can be string or h5py.Group object.

    If dest is a string, it is interpret as the destination file path.

    If dest is a h5py.Group object, the content will be read from the given group.

    """

    # Reading from existing group
    if isinstance(src, h5py.Group):
        ary = src['ndarray'][()]          # Read the array
        info = __read_info(src['info'])   # Read the meta info

    # Reading from file path
    path = filePath(src)

    if not path.read:
        raise ValueError("Unable to read from: " + str(path.full))

    with h5py.File(path.full, 'r') as f:
        ary = f['ndarray'][()]          # Read the array
        info = __read_info(f['info'])   # Read the meta info



    return metaArray(ary, info=info)
