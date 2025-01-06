# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 2024 19:09

@author: samhill

metaArray I/O to HDF5 files via the h5py module
"""

import numpy as np
import scipy.signal as ss
import typing
import decimal

from .core import metaArray
from .misc import spline_resize
from .misc import quantise
from .misc import filtfilt
from .misc import eng_unit

####################
# Helper functions #
####################


def padding_calc(metAry: metaArray, min_freq: float = 0.,
                 max_freq: float = 1e6,
                 resolution: float = 2048) -> int:
    """
    For a given 1D metaArray, work out the overall length of array necessary
    for the specified resolution between the frequency limits

    Padding ratio is always >= len(metAry)

    Example:
    rfft(ary, n = padding_calc(ary))

    """

    # Length of data
    n = len(metAry)

    t0 = metAry.get_range(0, 'begin')
    t1 = metAry.get_range(0, 'end')
    f = n / float(t1-t0)                        # Sampling freq
    # f = abs(f) / 2                            # Nyquist
    N = n * abs(max_freq - min_freq) / abs(f)   # Unpadded resolution

    if N < resolution:
        # Scale up accordingly
        return int(np.round((resolution / N) * n))
    else:
        # Already at or better resolution
        return int(np.round(n))


def meta_fir_len(metAry: metaArray, length: float = 0.005) -> int:
    """
    Simple helper function to work out the approprate number of taps for type I
    FIR for a given metaArray.

    Default to 0.5% of the input metAry duration, minimum 3.

    Input:
        metAry      Targed metaArray
        length      Desire length/duration of filter as ratio to len(metAry)

    Output:
        length      Length of the desire FIR filter (Int)
    """

    # Round to nearest ratio
    length = int(np.round(len(metAry) * length))

    if length < 3:
        length = 3

    # l must be odd for Type I filter
    if length % 2 == 0:
        length += 1

    return length


def meta_lowpass(metAry: metaArray, cut_freq: float,
                 length: typing.Union[int, float] = 0.005,
                 window: str = 'hann',
                 copy: bool = True) -> typing.Union[metaArray, None]:
    """
    Perform a two pass Type I FIR filter of cut-off freq(uency) on the given
    1D metaArray, once forward and once backward.

    Inputs:
        metAry      Target metaArray
        cut_freq    Cut-off frequency (float, in metAry unit)
        length      Length of the FIR filter (See notes below)
        window      Window function for the FIR filter
        copy        Whether to return a copy or modify inplace

    Length
        If given as float type, it will be interpreted as percentage length
        (duration) of the input metaArray.

        If given as int type, it will be interpreted as the desire number of
        taps for FIR filter.

        The default FIR length is 0.5% of that in the input metaArray-mimimum 3
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

    msg = f"Only 1D metaArray accepted, there are {metAry.ndim} dimensions in the given data."  # noqa: E501
    assert metAry.ndim == 1, msg

    if copy:
        ary = metAry.copy()
    else:
        ary = metAry

    # Work out the Nyquist frequency
    Nyquist = ary.get_smp_rate() / 2

    # Normalise frequency
    name_str = 'Low pass filtered at ' + eng_unit(cut_freq, unit='Hz',
                                                  sigfig=3)
    freq = float(cut_freq) / Nyquist

    # Number of taps
    if type(length) is float:
        length = meta_fir_len(ary, length=length)
    elif type(length) is int:
        pass
    else:
        msg = f'Unexpected variable type for length: {type(length)}'
        raise ValueError(msg)

    # a = [1.]
    b = ss.firwin(length, freq, window=window)

    ary.data = filtfilt(b, [1.], ary.data)

    if type(ary['name']) is str:
        ary['name'] += f"{metAry['name']} ({name_str})"
    else:
        ary['name'] = name_str

    if copy:
        return ary
    else:
        return


def meta_highpass(metAry: metaArray, cut_freq: float,
                  length: typing.Union[int, float] = 0.005,
                  window: str = 'hann',
                  copy: bool = True) -> typing.Union[metaArray, None]:
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
    loary = meta_lowpass(metAry, cut_freq, length=length,
                         window=window, copy=True)

    name_str = 'High pass filtered at ' + eng_unit(cut_freq,
                                                   unit='Hz', sigfig=3)

    if copy:
        ary = metAry.copy()
    else:
        ary = metAry

    ary.data -= loary.data

    if type(metAry['name']) is str:
        ary['name'] = f"{metAry['name']} ({name_str})"
    else:
        ary['name'] = name_str

    if copy:
        return ary
    else:
        return


def meta_resample(metAry: metaArray, rate: float = False,
                  length: typing.Union[int, float] = 0.005,
                  window: str = 'hamming',
                  order: int = 5) -> metaArray:
    """
    Resample 1D metaArray data into the given sampling rate, this is
    implemented using misc.spline_resize()

    This function distinct from the scipy.signal.resample function that, it
    uses spline for resampling, instead of FFT based method. Periodicity of the
    metAry content is not implied, or required.

    Inputs:
        metAry      Input metaArray
        rate        Sampling rate (float, in metaArray unit)
        l           Length of the FIR filter,
                    default to 0.5% len(metAry) minimum 3
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

    msg = f"Only 1D metaArray accepted, there are {metAry.ndim} dimensions in the given data."  # noqa: E501
    assert metAry.ndim == 1, msg

    ary = metAry.copy()

    if rate is False:
        # Target sampling rate is not specified
        rang = abs(ary.get_range(0, 'end') - ary.get_range(0, 'begin'))
        r = len(ary) / float(rang)

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
            print("Warning!! Unexpected values for scale evaluation!" +
                  f'scale variable ({scale}) should be greater than 1.')

        # This is what the sampling rate should be
        rate = scale * 10**exponent

    # Target size of the ary
    n = float(abs(ary.get_range(0, 'end') - ary.get_range(0, 'begin'))) * rate

    if type(length) is float:
        length = meta_fir_len(ary, length)

    # resize the data
    ary.data = spline_resize(ary.data, int(n), length=length,
                             window=window, order=order)

    # Update the meta info
    ary.update_range()

    return ary


metaResample = meta_resample


def meta_histogram(metAry: metaArray, bins: int = False) -> metaArray:
    """
    Compute a histogram of the given 1D metaArray.

    It will try to work out the maximum number of bins (i.e. minimum
    quantisation from the data) by default.

    Will raise QuantsationError if unable to determin number of bins.
    """

    msg = f"Only 1D metaArray accepted, there are {metAry.ndim} dimensions in the given data."  # noqa: E501
    assert metAry.ndim == 1, msg

    # Flatten the data to 1D array
    data = metAry.data.ravel()

    if bins is not False:
        quanter = data.ptp() / bins
    else:
        # Try to quantise the array data
        quanter = quantise(data)

    # Quantise the data, and offset to the +ve side of value, bincount
    # requires +ve int arrays
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
