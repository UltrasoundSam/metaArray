# Setting out version number
__version__ = '2.0.0'

# Import required functionality
from .core import metaArray

# Examples
from .example import demo

# Helper functions from misc
from .misc import dirPath, filePath, file_list, buffered_search
from .misc import gettypecode, unit_prefix, resample
from .misc import cplx_trig_func, MotherMorlet, spline_resize

# metaArray aware Plotting lib (based on matplotlib)
from .drv_pylab import plot1d, plot2d, plotcomplex, plotcomplexpolar

# metaArray aware numerical transforms
from .metaTrans import rfft, stfft, cwt

# metaArray aware functions
from .metaFunc import meta_resample, padding_calc, meta_lowpass

# File I/O
from .drv_h5py import to_h5, from_h5
from .drv_Tek import isf, DPO2000_csv, TDS2000_csv, DPO2000_isf
from .drv_flex import pout_hist, data_out1

# External devices
from .drv_ritec import Ritec4000
from .drv_hp4294 import HP4294A
