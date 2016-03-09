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
metaArray Helper functions
'''
import numpy as np
import scipy.signal as sp

from metaArray.core import metaArray
from metaArray.misc import spline_resize
from metaArray.misc import quantise
from metaArray.misc import filtfilt
from metaArray.misc import engUnit

####################
# Helper functions #
####################

def padding_calc(metAry, min_freq=0, max_freq=1e6, resolution=2048, \
                debug=False):
    """
    For a given 1D metaArray, work out the overall length of array necessary
    for the specified resolution between the frequency limits

    Padding ratio is always >= len(metAry)

    Example:
    rfft(ary, n = padding_calc(ary))

    """

    n = len(metAry)

    t0 = metAry.get_range(0, 'begin')
    t1 = metAry.get_range(0, 'end')
    f = n / float(t1-t0)                        # Sampling freq
    # f = abs(f) / 2                            # Nyquist
    N = n * abs(max_freq - min_freq) / abs(f)   # Unpadded resolution

    if N < resolution:
        return int(np.round((resolution / N) * n))          # Scale up accordingly
    else:                                     # Already at or better resolution
        return int(np.round(n))


def meta_fir_len(metAry, length=0.005):
    """
    Simple helper function to work out the approprate number of taps for type I
    FIR for a given metaArray.

    Default to 0.5% of the input metAry duration, minimum 3.

    Input:
        metAry      Targed metaArray
        length      Desire length/duration of the filter as ratio to len(metAry)

    Output:
        length      Length of the desire FIR filter (Int)
    """

    length = int(np.round(len(metAry) * length))             # Round to nearest ratio

    if length < 3: length = 3

    # l must be odd for Type I filter
    if length%2 == 0: length += 1

    return length

def meta_lowpass(metAry, freq, length=0.005, window='hann', copy=True):
    """
    Perform a two pass Type I FIR filter of cut-off freq(uency) on the given
    1D metaArray, once forward and once backward.

    Inputs:
        metAry      Target metaArray
        freq        Cut-off frequency (float, in metAry unit)
        length      Length of the FIR filter (See notes below)
        window      Window function for the FIR filter
        copy        Whether to return a copy or modify inplace

    Length
        If given as float type, it will be interpreted as percentage length
        (duration) of the input metaArray.

        If given as int type, it will be interpreted as the desire number of
        taps for FIR filter.

        The default FIR length is 0.5% of that in the input metaArray, mimimum 3.
        The exact number of taps is rounded to the next odd number, in order to
        meet the type I conditions.

    Scipy.signal.firwin support the following window options:
        boxcar
        triang
        blackman
        hamming
        hann
        bartlett
        flattop
        parzen
        bohman
        blackmanharris
        nuttall
        barthann
        kaiser (needs beta)
        gaussian (needs std)
        general_gaussian (needs power, width)
        slepian (needs width)
        chebwin (needs attenuation)
    """

    assert metAry.ndim is 1, "Only 1D metaArray accepted, there are {0:d} dimemsions in the given data.".format(metAry.ndim)

    if copy:
        ary = metAry.copy()
    else:
        ary = metAry

    # Work out the Nyquist frequency
    Nyquist = ary.get_smp_rate() / 2

    # Normalise frequency
    name_str = 'Low pass filtered at ' + engUnit(freq, unit='Hz', sigfig=3)
    freq = float(freq) / Nyquist

    # Number of taps
    if type(length) is float:
        length = meta_fir_len(ary, length=length)
    elif type(length) is int:
        pass
    else:
        raise ValueError('Unexpected variable type for length: ' + str(type(length)))

    # a = [1.]
    b = sp.firwin(length, freq, window=window)

    ary.data = filtfilt(b, [1.], ary.data)

    if type(ary['name']) is str:
        ary['name'] += ' (' + name_str + ')'
    else:
        ary['name'] = name_str

    if copy:
        return ary
    else:
        return


def meta_highpass(metAry, freq, length=0.005, window='hann', copy=True):
    """
    Perform a two pass Type I FIR filter of cut-off freq(uency) on the given
    1D metaArray, once forward and once backward.

    meta_highpass(metAry) === metAry - meta_lowpass(metAry)

    Inputs:
        metAry      Target metaArray
        freq        Cut-off frequency (float, in metAry unit)
        length      Length of the FIR filter (See notes below)
        window      Window function for the FIR filter
        copy        Whether to return a copy or modify inplace

    See meta_lowpass for details
    """
    loary = meta_lowpass(metAry, freq, length=length, window=window, copy=True)

    name_str = 'High pass filtered at ' + engUnit(freq, unit='Hz', sigfig=3)

    if copy:
        ary = metAry.copy()
    else:
        ary = metAry

    ary.data -= loary.data

    if type(metAry['name']) is str:
        ary['name'] = metAry['name'] + ' (' + name_str + ')'
    else:
        ary['name'] = name_str

    if copy:
        return ary
    else:
        return


def meta_resample(metAry, rate=False, l=0.005, window='hamming', order=5):
    """
    Resample 1D metaArray data into the given sampling rate, this is
    implemented using misc.spline_resize()

    This function distinct from the scipy.signal.resample function that, it
    uses spline for resampling, instead of FFT based method. Periodicity of the
    metAry content is not implied, or required.

    Inputs:
        metAry      Input metaArray
        rate        Sampling rate (float, in metaArray unit)
        l           Length of the FIR filter, default to 0.5% len(metAry) mimimum 3
        window      Window method to generate the FIR filter
        order       Order of spline polynomial, default to 5

    Output:
        metaArray   A resampled copy of the input metAry

    If upsampling, quintic spline interpolation will be used.

    If downsampling, two pass anti-aliasing FIR filter will be applied, once
    forward and once reverse to null the group delay, then quintic spline
    interpolation will be used.

    If target sampling rate is not given, it will try to find the next highest
    sampling rate by default. The resampled data will always align at time 0,
    and never exceed the duration of the given data.

    The sampling rate will come in multiples of 1, 2, or 5Hz, this function
    will modify the input array in place.
    """

    assert metAry.ndim is 1, "Only 1D metaArray accepted, there are {0:d} dimemsions in the given data.".format(metAry.ndim)

    ary = metAry.copy()

    if rate == False:
        # Target sampling rate is not specified
        r = len(ary) / float(abs(ary.get_range(0, 'end') - ary.get_range(0, 'begin')))

        # Find out the exponent of the current sampling rate
        exponent = decimal.Decimal(str(r)).adjusted()

        # Remove the exponent
        scale = r * 10**(0 - exponent)

        # make the standard scale slightly larger (1e-5) so numerical
        # error (rounding error) do not come in to play and force it up
        # to the next sampling scale
        if scale > 5.00005:
            scale = 10
        elif scale > 2.00002:
            scale = 5
        elif scale > 1.00001:
            scale = 2
        else:
            # This really shouldnt happen, but just in case the Decimal
            # function return numbers like 0.123e+45 instead of 1.23e+45
            scale = 1
            print("Warning!! Unexpected values for scale evaluation!" + \
            'scale variable (' + str(scale) + ') should be greater than 1.')

        # This is what the sampling rate should be
        rate = scale * 10**exponent

    # Target size of the ary
    n = float(abs(ary.get_range(0, 'end') - ary.get_range(0, 'begin'))) * rate

    if type(l) is float: l = meta_fir_len(ary, l)

    # resize the data
    ary.data = spline_resize(ary.data, n, l=l, window=window, order=order)

    # Update the meta info
    ary.update_range()

    return ary

metaResample = meta_resample


def meta_histogram(metAry, bins=False):
    """
    Compute a histogram of the given 1D metaArray.

    It will try to work out the maximum number of bins (i.e. minimum
    quantisation from the data) by default.

    Will raise QuantsationError if unable to determin number of bins.
    """

    assert metAry.ndim is 1, "Only 1D metaArray accepted, there are {0:d} dimemsions in the given data.".format(metAry.ndim)

    # Flatten the data to 1D array
    data = metAry.data.ravel()

    if bins is not False:
        quanter = data.ptp() / bins
    else:
        # Try to quantise the array data
        quanter = quantise(data)

    # Quantise the data, and offset to the +ve side of value, bincount requires +ve
    # int arrays
    quantum = np.round(data / quanter).astype(int)

    quantum -= quantum.min()

    # Do the bincount for histogram
    hist = np.bincount(quantum)

    # Update the metaInfo
    hist = metaArray(hist)
    hist.set_range(0, 'begin', metAry.min())
    hist.set_range(0, 'end', metAry.max())
    hist.set_range(0, 'unit', metAry['unit'])
    hist.set_range(0, 'label', metAry['label'])

    hist['name'] = 'Histogram of ' + metAry['name']
    hist['unit'] = ''
    hist['label'] = 'Counts'

    return hist

histogram = meta_histogram
