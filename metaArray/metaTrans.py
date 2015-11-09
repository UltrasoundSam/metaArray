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

####################################################################
# This file contain numerical transforms that can handle metaArray #
####################################################################

# import numpy as np
from numpy import zeros
from numpy import arange
from numpy import floor
from numpy.fft import rfft as np_rfft

from scipy import linspace, logspace
from scipy.signal import convolve
from scipy.signal import get_window

from core import metaArray
from misc import resample
from misc import spline_resize

def rfft(metAry, n=None, axes=-1):
    """
    RFFT function wraper for numpy.fft to be used on metaArray objects, returns 
    scaled metaArray object.
    """
    
    nyquist = len(metAry) * 0.5 / (metAry.get_range(axes, 'end') - metAry.get_range(axes, 'begin'))
    
    fary = metaArray(np_rfft(metAry.data, n, axes))
    
    fary.set_range(0, 'begin', 0)
    fary.set_range(0, 'end', nyquist)
    fary.set_range(0, 'unit', 'Hz')
    fary.set_range(0, 'label', 'Frequency')
    
    fary['unit'] = ''
    fary['label'] = 'Amplitude'
    try:
        fary['name'] = 'FFT{ ' + metAry['name'] + ' }'
    except TypeError:
        fary['name'] = 'FFT{ }'
    
    return fary




def stfft(metAry, tres=100, fres=None, window='blackmanharris', \
            fmin=None, fmax=None, mag=True, debug=False):
    """
    Simple implementation of short time fast Fourier transform on metaArray
    object.
    
    metAry      Input metaArray
    tres        Temporal resolution
    fres        Frequency resolution
    window      Window function or None
    fmin        Cut-off frequency for the return data, default to the 0
    fmax        Cut-off frequency for the return data, default to the Nyquist
    mag         The default is to return the abs() value, return complex array if false
    
    Each window will overlap 50% with its immediate neighbours. e.g.:
    
    |_____________________________________________|
    | | 1 | 3 | 5 | 7 | 9 | 11| 13| 15| 17| 19| 21|
    | 0 | 2 | 4 | 6 | 8 | 10| 12| 14| 16| 18| 20| |
    
    
    
    """
    f1 = fmax
    
    # Length of the (short) time window
    l = int(round(2 * len(metAry) / float(tres)))
    
    # List of (short) time window starting points
    winlst = linspace(0, len(metAry) - l, tres).round().astype(int)
    
    # Native RFFT frequency resolution to Nyquist
    lfres = int(floor(l/2.0)+1)   
    Nyquist = 0.5 * len(metAry) / (metAry.get_range(0, 'end') - metAry.get_range(0, 'begin')) 
    
    # RFFT len, use native resolution by default
    n = None
    
    # Generate the (short) time window function
    if window is None:
        win = 1
    else:
        win = get_window(window, l)
    
    # Decide where to slice the rfft output as a ratio to Nyquist
    # Make fmax < 1 or None
    if fmax is not None:
        if fmax < Nyquist:
            fmax = fmax / Nyquist
        elif fmax >= Nyquist:
            fmax = None
            if debug:
                print("*** Warning, spec frequency range beyond Nyquist limit")
    
    # Check whether padding is needed
    # If fres is not specified, use the native resolution
    if fres is None:
        if fmax is None:
            # No freq limit, use native resolution
            fres = lfres
        else:
            # Still on native resolution, but truncated to fmax
            fres = int(round(fmax * lfres))
    else:
        # fres is specified
        if fmax is not None:
            # freq limit specified, work out global freq resolution
            gfres = int(round(fres / fmax))
        else:
            # No freq limit, global freq resolution is same as fres
            gfres = fres
        
        # Global freq resolution is greater than native freq resolution
        # Need padding for rfft
        if gfres > lfres:
            n = (gfres - 1) * 2
        elif gfres < lfres:
            # No need for padding, but throwing away freq resolution for nothing
            if debug:
                print("*** Warning, frequency resolution is artificially limited")
        # else gfres = lfres, no need for padding, native fres is just right
    
    
    # Convert fmax to array length if specified
    if fmax is not None:
        # If rfft is padded
        if n is not None:
            fmax = int(round(int(floor(n/2.0)+1) * fmax))
        else:
            # Otherwise just truncate from native output
            fmax = int(round(lfres * fmax))
    
    if debug:
        src_len = len(metAry.data[:l]*win)
        rfft_len = len(np_rfft(metAry.data[:l]*win, n=n))
        print("*** l: " + str(l))
        print("*** lfres: " + str(lfres))
        print("*** Nyquist: " + str(Nyquist))
        print("*** n: " + str(n))
        print("*** fmax: " + str(fmax))
        print("*** fres: " + str(fres))
        print("*** src_len: " + str(src_len))
        print("*** rfft_len: " + str(rfft_len))
        
    
    if mag:
        # Construct a place holder of the 2D time-freq output
        tfary = zeros((tres, fres)).astype(float)
        for i in range(len(winlst)):
            t = winlst[i]                # Where the (short) time window starts
            # Do the rfft to length n, and slice to fmax, then take abs()
            tfary[i] = spline_resize(abs(np_rfft(metAry.data[t:t+l]*win, n=n)[:fmax]), fres)
    else:
        # Construct a place holder of the 2D time-freq output
        tfary = zeros((tres,fres)).astype(complex)
        for i in range(len(winlst)):
            t = winlst[i]
            # Do the rfft to length n, and slice to fmax
            tfary[i] = spline_resize(np_rfft(metAry.data[t:t+l]*win, n=n)[:fmax], fres)
    
    tfary = metaArray(tfary)
    
    try:
        tfary['name'] = 'STFFT{ ' + metAry['name'] + ' }'
    except:
        tfary['name'] = 'STFFT{ }'
    
    tfary['unit'] = metAry['unit']
    tfary['label'] = metAry['label']
    
    # Per axis definitions
    tfary.set_range(0, 'begin', metAry.get_range(0, 'begin')) 
    tfary.set_range(0, 'end', metAry.get_range(0, 'end'))
    tfary.set_range(0, 'unit', metAry.get_range(0, 'unit'))
    tfary.set_range(0, 'label', metAry.get_range(0, 'label'))
    tfary.set_range(1, 'begin', 0)
    if f1 is None:
        tfary.set_range(1, 'end', Nyquist)
    else:
        tfary.set_range(1, 'end', f1)
    tfary.set_range(1, 'unit', 'Hz')
    tfary.set_range(1, 'label', 'Frequency')
    
    return tfary


def cwt(x, wavelet, scale0, scale1, res, scale=10, tres = None, debug = False):
    """
    This function will return continous wavelet transform of x, 
    calculated by the convolution method.
    
    x is a 1-D metaArray
    
    Inputs:
        x           The data as an metaArray object.
        wavelet     Mother wavelet function (e.g. wavelet(scale0) should
                    provide the largest scale daughter wavelet)
        scale0      Starting scale length
        scale1      Stoping scale length
        res         Resolution in the scale space (i.e. number of daughter wavelets)
        tres        Resolution in the time space (Default to be same as x)
    
    Options:
        scale       [int|float|True|'linscal'|'linfreq']
                    
                    int float True
                    Logarithmic scale (based 10) is used by default for the 
                    production of daughter wavelets. If a number is given
                    the log space will be generated with that base.
                    
                    'linscal'
                    Scale length is step linearly (i.e. freq in 1/x space)
                    
                    'linfreq'
                    Frequency is step linearly (i.e. scale length in 1/x space)
                    
    
    Output:
        A 2D metaArray in the time-sacle space.
    """
    
    data = x.data
    d_len = len(data)
    
    if tres is None:
        tres = len(x)
        flag_resample = False
    else:
        # Sanity check
        flag_resample = True
        resmp_time = arange(len(x))       # i.e. pretend 1Hz sample
        resmp_rate = float(tres) / len(x)
    
    # Generate a blank time-sacle space array
    page = zeros((tres,res)).astype('complex128')
    page = metaArray(page)
    
    prange = page['range']
    xrange = x['range']
    prange['label'][0] = xrange['label'][0]
    prange['begin'][0] = xrange['begin'][0]
    prange['end'][0] = xrange['end'][0]
    prange['unit'][0] = xrange['unit'][0]
    
    prange['label'][1] = "Scale"
    prange['begin'][1] = scale0
    prange['end'][1] = scale1
    
    try:
        prange['unit'][1] = wavelet.unit
    except:
        pass
    
    if x['name'] is not None:
        page['name'] = 'cwt{' + x['name'] + '}'
    else:
        page['name'] = 'cwt'
    
    
    # Generate a list of scales
    if isinstance(scale, int) or isinstance(scale, float):
        # Log scale applied, given log base.
        scale0 = log(scale0)/log(scale)
        scale1 = log(scale1)/log(scale)
        # print "*** num", scale0, scale1, res, scale
        scl_lst = logspace(scale0, scale1, res, base=scale)
        prange['log'][1] = scale
    elif scale == True:
        # Log scale applied, use default base
        scale0 = log(scale0)/log(10)
        scale1 = log(scale1)/log(10)
        # print "*** log", scale0, scale1, res, scale
        scl_lst = logspace(scale0, scale1, res, base=10)
        prange['log'][1] = 10
    elif scale == 'linscal':
        # print "*** lin", scale0, scale1, res, scale
        # Log scale is not applied, everything is linear
        scl_lst = linspace(scale0, scale1, res)
    elif scale == 'linfreq':
        scale0 = 1 / scale0
        scale1 = 1 / scale1
        scl_lst = linspace(scale0, scale1, res)
        scl_lst = 1 / scl_lst
    else:
        raise ValueError, "Log scale descriptor can only be int,\
            float, True, False or None, given: " + str(lg[i])
    
    # return scl_lst, page
    
    if debug:
        print "There are a total number of " + str(len(scl_lst)) + " scales to be processed:" 
    
    for i in range(len(scl_lst)):
        
        d_wavelet = wavelet(scl_lst[i])
        
        ###if debug:
        ###   print "\t line number: " + str(i) + "\t scale: " + str(scl_lst[i]) + "\t x.data: " + str(len(x.data)) + "\t wavelet: " + str(len(d_wavelet)
        
        if len(d_wavelet) > d_len:
            # It probably shouldnt happen, because the daughter wavelet is longer than
            # the signal itself now
            print "\t line number: " + str(i) + "\t scale: " + str(scl_lst[i]) + "\t data length: " + str(len(x.data)) + "\t wavelet length: " + str(len(d_wavelet))
            raise "Warning!!"
        else:
            line = convolve(data, d_wavelet, mode = 'same')
        
        if flag_resample:
            ###if debug:
            ###   print "\t resmp_time: " + str(len(resmp_time)) + "\t line: " + str(len(line)) + "\t resmp_rate: " + str(resmp_rate)
            
            line = resample(resmp_time, line, resmp_rate)[1]
            
            if debug:
                print "\t line number: " + str(i) + "\t scale: " + str(scl_lst[i])
            
        page.data[:len(line),i] = line
        
    return page
    



