# Setting out version number
__version__ = '2.0.0'


# Import required functionality
from .core import metaArray

# Examples
from .example import demo

# Helper functions from misc
from .misc import dirPath, filePath, file_list, buffered_search
from .misc import gettypecode, unit_prefix, resample
from .misc import cplx_trig_func, mother_morlet, spline_resize

# metaArray aware Plotting lib (based on matplotlib)
from .drv_pylab import plot1d, plot2d, plotcomplex, plotcomplexpolar

# File I/O
from .drv_h5py import to_h5, from_h5
from metaArray.drv_Tek import isf, DPO2000_csv, TDS2000_csv, DPO2000_isf

# External devices
from .drv_ritec import Ritec4000
