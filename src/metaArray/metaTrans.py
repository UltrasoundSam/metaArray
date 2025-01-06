# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 2024 14:05

@author: samhill

This file contain numerical transforms that can handle metaArray
"""

import numpy as np
import scipy as sp
import typing

from .core import metaArray
from .misc import resample, MotherMorlet
from .misc import spline_resize


def rfft(metAry: metaArray, n: int = None,
         axes: int = -1) -> metaArray:
    """
    RFFT function wraper for numpy.fft to be used on metaArray objects, returns
    scaled metaArray object.
    """

    span = metAry.get_range(axes, 'end') - metAry.get_range(axes, 'begin')
    nyquist = len(metAry) * 0.5 / (span)

    fary = metaArray(np.fft.rfft(metAry.data, n, axes))

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


def stfft(metAry: metaArray, tres: int = 100, fres: int = None,
          window: str = 'blackmanharris', fmin: float = None,
          fmax: float = None, mag: bool = True,
          debug: bool = False) -> metaArray:
    """
    Simple implementation of short time fast Fourier transform on metaArray
    object.

    metAry      Input metaArray
    tres        Temporal resolution
    fres        Frequency resolution
    window      Window function or None
    fmin        Cut-off frequency for the return data, default to the 0
    fmax        Cut-off frequency for the return data, default to the Nyquist
    mag         The default is to return the abs() value, return complex
                array if false

    Each window will overlap 50% with its immediate neighbours. e.g.:

    |_____________________________________________|
    | | 1 | 3 | 5 | 7 | 9 | 11| 13| 15| 17| 19| 21|
    | 0 | 2 | 4 | 6 | 8 | 10| 12| 14| 16| 18| 20| |

    """
    f1 = fmax

    # Length of the (short) time window
    length = int(round(2 * len(metAry) / float(tres)))

    # List of (short) time window starting points
    winlst = np.linspace(0, len(metAry) - length, tres).round().astype(int)

    # Native RFFT frequency resolution to Nyquist
    lfres = int(np.floor(length/2.0)+1)
    span = metAry.get_range(0, 'end') - metAry.get_range(0, 'begin')
    Nyquist = 0.5 * len(metAry) / (span)

    # RFFT len, use native resolution by default
    n = None

    # Generate the (short) time window function
    if window is None:
        win = 1
    else:
        win = sp.signal.get_window(window, length)

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
            # No need for padding, but throwing away freq resolution for nada
            if debug:
                print("Warning, frequency resolution is artificially limited")
        # else gfres = lfres, no need for padding, native fres is just right

    # Convert fmax to array length if specified
    if fmax is not None:
        # If rfft is padded
        if n is not None:
            fmax = int(round(int(np.floor(n/2.0)+1) * fmax))
        else:
            # Otherwise just truncate from native output
            fmax = int(round(lfres * fmax))

    if debug:
        src_len = len(metAry.data[:length]*win)
        rfft_len = len(np.fft.rfft(metAry.data[:length]*win, n=n))
        print("*** l: " + str(length))
        print("*** lfres: " + str(lfres))
        print("*** Nyquist: " + str(Nyquist))
        print("*** n: " + str(n))
        print("*** fmax: " + str(fmax))
        print("*** fres: " + str(fres))
        print("*** src_len: " + str(src_len))
        print("*** rfft_len: " + str(rfft_len))

    if mag:
        # Construct a place holder of the 2D time-freq output
        tfary = np.zeros((tres, fres)).astype(float)
        for i in range(len(winlst)):
            t = winlst[i]                # Where the (short) time window starts
            # Do the rfft to length n, and slice to fmax, then take abs()
            fft = np.fft.rfft(metAry.data[t:t+length]*win, n=n)[:fmax]
            tfary[i] = spline_resize(abs(fft), fres)
    else:
        # Construct a place holder of the 2D time-freq output
        tfary = np.zeros((tres, fres)).astype(complex)
        for i in range(len(winlst)):
            t = winlst[i]
            # Do the rfft to length n, and slice to fmax
            fft = np.fft.rfft(metAry.data[t:t+length]*win, n=n)[:fmax]
            tfary[i] = spline_resize(fft, fres)

    tfary = metaArray(tfary)

    try:
        tfary['name'] = 'STFFT{ ' + metAry['name'] + ' }'
    except TypeError:
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


def cwt(x: metaArray, wavelet: MotherMorlet, scale0: typing.Union[int, float],
        scale1: typing.Union[int, float], res: int,
        scale: typing.Union[int, float, bool, str] = 10,
        tres: float = None, debug: bool = False) -> metaArray:
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
        res         Resolution in the scale space (i.e. number of
                    daughter wavelets)
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
        resmp_time = np.arange(len(x))       # i.e. pretend 1Hz sample
        resmp_rate = float(tres) / len(x)

    # Generate a blank time-sacle space array
    page = np.zeros((tres, res)).astype('complex128')
    page = metaArray(page)

    prange = page['range']
    x_range = x['range']
    prange['label'][0] = x_range['label'][0]
    prange['begin'][0] = x_range['begin'][0]
    prange['end'][0] = x_range['end'][0]
    prange['unit'][0] = x_range['unit'][0]

    prange['label'][1] = "Scale"
    prange['begin'][1] = scale0
    prange['end'][1] = scale1

    try:
        prange['unit'][1] = wavelet.unit
    except AttributeError:
        pass

    if x['name'] is not None:
        page['name'] = 'cwt{' + x['name'] + '}'
    else:
        page['name'] = 'cwt'

    # Generate a list of scales
    if isinstance(scale, int) or isinstance(scale, float):
        # Log scale applied, given log base.
        scale0 = np.log(scale0)/np.log(scale)
        scale1 = np.log(scale1)/np.log(scale)
        # print "*** num", scale0, scale1, res, scale
        scl_lst = np.logspace(scale0, scale1, res, base=scale)
        prange['log'][1] = scale
    elif scale is True:
        # Log scale applied, use default base
        scale0 = np.log(scale0)/np.log(10)
        scale1 = np.log(scale1)/np.log(10)
        # print "*** log", scale0, scale1, res, scale
        scl_lst = np.logspace(scale0, scale1, res, base=10)
        prange['log'][1] = 10
    elif scale == 'linscal':
        # print "*** lin", scale0, scale1, res, scale
        # Log scale is not applied, everything is linear
        scl_lst = np.linspace(scale0, scale1, res)
    elif scale == 'linfreq':
        scale0 = 1 / scale0
        scale1 = 1 / scale1
        scl_lst = np.linspace(scale0, scale1, res)
        scl_lst = 1 / scl_lst
    else:
        raise ValueError(f"Log scale descriptor can only be int, \
                         float, True, False or None, given: {scale}")

    # return scl_lst, page

    if debug:
        print(f"There are {len(scl_lst)} scales to be processed")

    for i in range(len(scl_lst)):

        d_wavelet = wavelet(scl_lst[i])

        if len(d_wavelet) > d_len:
            # It probably shouldnt happen, because the daughter wavelet
            # is longer than the signal itself now
            print(f"\t line number: {i}\t scale: {str(scl_lst[i])}" +
                  f"\t data length: {len(x.data)}\t wavelet length: " +
                  str(len(d_wavelet)))
            msg = "Warning: Daughter wavelet is longer than itself!"
            raise ValueError(msg)
        else:
            line = sp.signal.convolve(data, d_wavelet, mode='same')

        if flag_resample:
            line = resample(resmp_time, line, resmp_rate)[1]

            if debug:
                print(f"\t line number: {i}\t scale: {str(scl_lst[i])}")

        page.data[:len(line), i] = line

    return page
