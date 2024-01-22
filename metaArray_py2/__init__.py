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
"""
Init function for metaArray library
"""

__all__ = ["metaArray", "dirPath", "filePath", "file_list",\
"buffered_search", "gettypecode", "unitPrefix", "cplx_trig_func",\
"mother_morlet", "resample", "spline_resize", "meta_resample",\
"meta_resample", "padding_calc", "meta_lowpass", "rfft", "stfft", "cwt",\
"isf", "DPO2000_csv", "TDS2000_csv", "DPO2000_isf", "pout_hist",\
"data_out1", "plot1d", "plot2d", "plotcomplex", "plotcomplexpolar",\
 "drv_h5py", "Ritec4000"]

## Mathematical and Physical constants definition
# import constants

# Independent helper function/objects
from metaArray.misc import dirPath, filePath, file_list, buffered_search
from metaArray.misc import gettypecode, unitPrefix, resample
from metaArray.misc import cplx_trig_func, mother_morlet, spline_resize

# Core metaArray module
from metaArray.core import metaArray
# metaArray aware functions
from metaArray.metaFunc import meta_resample, meta_resample
from metaArray.metaFunc import padding_calc, meta_lowpass
# metaArray aware numerical transforms
from metaArray.metaTrans import rfft, stfft, cwt

# metaArray I/O, drivers
# Basic file driver classes
# from drv_fortran import binrecord
# from drv_csv import csv_file
from metaArray.drv_h5py import to_h5, from_h5       # Require h5py
from metaArray.drv_ritec import Ritec4000
# Higher level driver classes
from metaArray.drv_Tek import isf, DPO2000_csv, TDS2000_csv, DPO2000_isf
from metaArray.drv_flex import pout_hist, data_out1

# metaArray aware Plotting lib (based on matplotlib)
from metaArray.drv_pylab import plot1d, plot2d, plotcomplex, plotcomplexpolar

# Examples
from metaArray.example import demo

from metaArray.version import version
