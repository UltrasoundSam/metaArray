# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 2024 15:46

@author: samhill

This file contain a number of drivers to read the Tek scope output files.
"""

import typing
import numpy.typing as npt
from operator import itemgetter
from itertools import groupby
from numpy import array, float64
from struct import unpack
from decimal import Decimal

from os import linesep

from .core import metaArray
from .drv_csv import CSVFile
from .misc import linear_chk
from .misc import gettypecode
from .misc import filePath
from .misc import buffered_search


class isf:
    """
    Generic isf file/stream interpreter

    #
    # wfm -> isf file generator example
    ####################################
    #   :WFMPRE:BYT_NR 1;
    #   BIT_NR 8;
    #   ENCDG BIN;
    #   BN_FMT RI;
    #   BYT_OR LSB;
    #   CH2:WFID "Ch2, AC coupling, 20mVolts/div, 50ns/div, 50000 points, Sample mode";  # noqa: E501
    #   NR_PT 50000;
    #   PT_FMT Y;
    #   XUNIT "s";
    #   XINCR 1e-09;
    #   PT_OFF 15000;
    #   YUNIT "Volts";
    #   YMULT 0.0008;
    #   YOFF -50;
    #   YZERO 0;
    #   :CURVE #550000\xcd\xcb\
    #
    ################################
    #
    # DPO2000 series scopes example
    ################################
    #   :WFMP:NR_P      100000
    #   :WFMP:BYT_N     1
    #   BIT_N   8
    #   ENC     BIN
    #   BN_F    RI
    #   BYT_O   MSB
    #   WFI     "Ch1, AC coupling, 200.0mV/div, 40.00us/div, 100000 points, Sample mode"
    #   NR_P    100000
    #   PT_F    Y
    #   XUN     "s"
    #   XIN     4.0000E-9
    #   XZE     -200.0000E-6
    #   PT_O    0
    #   YUN     "V"
    #   YMU     8.0000E-3
    #   YOF     50.5000
    #   YZE     0.0E+0
    #   VSCALE  200.0000E-3
    #   HSCALE  40.0000E-6
    #   VPOS    2.0200
    #   VOFFSET 0.0E+0
    #   HDELAY  0.0E+0
    #   COMP    COMPOSITE_YT
    #   FILTERF 100000000
    #   :CURV   #6100000333333333
    #
    # DPO4000 series scopes example
    ################################
    #   :WFMO:BYT_N 1
    #   BIT_N       8
    #   ENC         BIN
    #   BN_F        RI
    #   BYT_O       MSB
    #   WFI         "Ch1, DC coupling, 50.00mV/div, 400.0us/div, 10000 points, Sample mode"
    #   NR_P        10000
    #   PT_F        Y
    #   PT_OR       LINEA
    #   XUN         "s"
    #   XIN         400.0000E-9
    #   XZE         -400.0000E-6
    #   PT_O        0
    #   YUN         "V"
    #   YMU         2.0000E-3
    #   YOF         0.0E+0
    #   YZE         0.0E+0
    #   DOM         TIM
    #   WFMTYP      ANALOG
    #   CENTERFREQ  0.0E+0
    #   SPAN        0.0E+0
    #   REFLE       0.0E+0
    #   :CURV       #510000
    #
    ###########################################################################
    #    NR_P
    #        The number of data points in the waveform record.
    #
    #    BYT_N <{1|2}>
    #        1 The number of bytes per incoming waveform data is 1, which is the default setting.
    #        2 indicating that there are 2 bytes per waveform data point.
    #
    #    BIT_N
    #        The number of bits per binary waveform point.
    #
    #    ENC {ASC|BIN}
    #        ASC specifies that the incoming data is in ASCII format.
    #        BIN specifies that the incoming data is in a binary format whose
    #            further interpretation requires knowledge of BYT_NR, BIT_NR,
    #            BN_FMT, and BYT_OR.
    #
    #    BN_F {RI|RP}
    #        RI specifies signed integer data point representation
    #        RP specifies positive integer data point representation
    #
    #    BYT_O {LSB|MSB}
    #        LSB specifies that the least significant byte will be expected first.
    #        MSB specifies that the most significant byte will be expected first.
    #
    #    WFI
    #        A string describing several aspects of the acquisition parameters for the
    #            waveform specified by the DATa:SOUrce command.
    #
    #    PT_F {ENV|Y}
    #        ENV specifies that the waveform is transmitted in envelope mode as maximum
    #            and minimum point pairs. Only Y values are explicitly transmitted.
    #            Absolute coordinates are given by:
    #                  Xn = XZEro + XINcr (n - PT_Off)
    #                  Ynmax = YZEro + YMUlt (ynmax - YOFf)
    #                  Ynmin = YZEro + YMUlt (ynmin - YOFf)
    #
    #        Y specifies a normal waveform where one ASCII or binary data point is
    #            transmitted for each point in the waveform record. Only Y values are
    #            explicitly transmitted. Absolute coordinates are given by:
    #                  Xn = XZEro + XINcr (n - PT_Off)
    #                  Yn = YZEro + YMUlt (yn - YOFf)
    #
    #    XUN
    #        The horizontal units of the incoming waveform.
    #
    #    XIN
    #        Horizontal point spacing in units of WFMOutpre:XUNit.
    #
    #    XZE
    #        The time coordinate of the first point in the waveform.
    #
    #    PT_O
    #        The query form always returns a 0, if the waveform specified by DATA:SOUrce
    #            is on or displayed. If the waveform is not displayed, the query form
    #            generates an error and returns event code 2244. This command is for
    #            compatibility with other Tektronix oscilloscopes.
    #
    #    YUN
    #        The vertical units for the waveform
    #
    #    YMU
    #        The vertical scale factor of the waveform, expressed in YUNits per waveform
    #            data point level. For one byte waveform data, there are 256 data point
    #            levels. For two byte waveform data there are 65,536 data point levels.
    #            YMUlt, YOFf, and YZEro are used to convert waveform record values to
    #            YUNit values using the following formula (where dl is the data level;
    #            curve_in_dl is a data point in CURVe):
    #                value_in_units = ((curve_in_dl - YOFf_in_dl) * YMUlt) + YZEro_in_units
    #
    #    YOF
    #        The vertical position in digitizing levels for the waveform.
    #
    #    YZE
    #        Returns the vertical offset in units specified by WFMOutpre:YUNit? for the
    #            waveform.
    #
    #    VSCALE: 200.0000E-3
    #    HSCALE: 40.0000E-6
    #    VPOS:   2.0200
    #    VOFFSET:        0.0E+0
    #    HDELAY: 0.0E+0
    #    COMP {COMPOSITE_YT|COMPOSITE_ENV|SINGULAR_YT}
    #        The type of data used by the CURVe? query.
    #        COMPOSITE_YT uses the sample part of the composite waveform.
    #        COMPOSITE_ENV uses the peak-detect part of the composite waveform.
    #        SINGULAR_YT uses the sample part of the singular waveform.
    #
    #    FILTERF
    #        The FilterVu low pass filter frequency applied to the LRL waveform record of
    #            the source waveform specified by DATa:SOUrce. If the DATa:RESOlution is
    #            set to FULL, then this is the filter frequency applied to the full
    #            resolution (LRL) waveform. If the DATa:RESOlution is set to REDUced,
    #            then this is the filter frequency applied to the reduced resolution
    #            (thumbnail) waveform.
    #
    #    CURV {<Block>|<asc curve>}
    #        Waveform in binary or ASCII format.
    #        <Block> is the waveform data in binary format. The waveform is
    #        formatted as:
    #           #<x><yyy><data><newline>
    #
    #        <x> is the number of y bytes. For example, if <yyy>=500, then <x>=3)
    #        <yyy> is the number of bytes to transfer if samples are one or two
    #                bytes wide.
    #        <data> is the curve data.
    #        <newline> is a single byte new line character at the end of the data.
    #
    ###########################################################################
    """
    def __init__(self, path: str, debug: bool = False,
                 buffer_size: int = 4096) -> None:
        """
        4kB buffer size
        """
        self.file_path = filePath(path)
        self.debug = debug
        self.buffer_size = buffer_size = buffer_size * 1024

        # Index of records in the file. Each item should have the format:
        #      [hdr_pos, hdr_len, data_pos, data_len, header_dict, unpack_str]
        self.idx = idx = []

        # Known header fields and value type.
        # This is only used for data type conversion from the byte stream
        # String values are left as is
        hdr_format = self.hdr_format = []
        hdr_format.append([':WFMP:NR_P', int])
        hdr_format.append([':WFMPRE:NR_PT', int])
        hdr_format.append([':WFMP:BYT_N', int])
        hdr_format.append([':WFMPRE:BYT_NR', int])
        hdr_format.append([':WFMO:BYT_N', int])
        hdr_format.append(['BIT_N', int])
        hdr_format.append(['BIT_NR', int])
        # ENC or ENCDG -> BIN
        # BN_F or BN_FMT-> RI
        # BYT_O or BYT_OR -> LSB or MSB
        # CH2:WFID "Ch2, AC coupling, 20mVolts/div, 50ns/div, 50000 points, Sample mode";  # noqa: E501
        # WFI "Ch2, DC coupling, 1.000mV/div, 40.00us/div, 1000000 points, Average mode";  # noqa: E501
        # PT_F or PT_FMT -> Y
        # XUN or XUNIT -> "s"
        hdr_format.append(['NR_P', int])
        hdr_format.append(['NR_PT', int])
        hdr_format.append(['PT_O', int])
        hdr_format.append(['PT_OFF', int])
        # PT_OR       LINEA                 # Always LINEAr
        hdr_format.append(['XIN', float])
        hdr_format.append(['XINCR', float])
        hdr_format.append(['XZE', float])
        # YUN or YUNIT -> "V" or  "Volts"
        hdr_format.append(['YMU', float])
        hdr_format.append(['YMULT', float])
        hdr_format.append(['YOF', float])
        hdr_format.append(['YOFF', float])
        hdr_format.append(['YZE', float])
        hdr_format.append(['YZERO', float])
        hdr_format.append(['VSCALE', float])
        hdr_format.append(['HSCALE', float])
        hdr_format.append(['VPOS', float])
        hdr_format.append(['VOFFSET', float])
        hdr_format.append(['HDELAY', float])
        # COMP -> COMPOSITE_YT
        hdr_format.append(['FILTERF', int])
        hdr_format.append(['CENTERFREQUENCY', float])
        # DOMAIN -> TIME
        hdr_format.append(['REFLEVEL', float])
        hdr_format.append(['SPAN', float])
        # WFMTYPE -> ANALOG
        # :CURV -> #72000000

        debug_str = ''

        # Identify the header and data byte positions in the byte streams
        #
        # The data string can take the following forms:
        #
        # ":WFMPRE:" --- Header info ---- ":CURVE #" ------ Binary Data -----
        # ":WFMP:" ----- Header info ---- ":CURV #" ------- Binary Data -----

        # END structure INIT

        f = self.open()
        f_pos = 0

        # Identify key locations in the data stream, write into header_rcd
        # There maybe multiple header-data streams
        while True:

            # Find the beginning of the header
            hdr_start = buffered_search(f, ':WFMP:'.encode(), start=f_pos,
                                        buffer_size=buffer_size)
            if hdr_start == -1:
                hdr_start = buffered_search(f, b':WFMPRE:', start=f_pos,
                                            buffer_size=buffer_size)
                if hdr_start == -1:
                    if debug:
                        debug_str += 'Neither header descriptors (":WFMP:" or ":WFMPRE:") is found.'  # noqa: E501
                        print(debug_str)
                    break  # no more header found
                elif debug:
                    debug_str += f'Header ":WFMPRE:" found at {hdr_start}'
                    debug_str += linesep
            elif debug:
                debug_str += f'Header":WFMP:" found at {hdr_start}'
                debug_str += linesep

            # Find the following data stream
            data_start = buffered_search(f, b':CURV #', start=hdr_start,
                                         buffer_size=buffer_size)
            if data_start == -1:
                data_start = buffered_search(f, b':CURVE #', start=hdr_start,
                                             buffer_size=buffer_size)

                if data_start == -1:
                    # Very bad, found header but no data!
                    # Show debug info if requested, ignore otherwise
                    if debug:
                        msg = 'Failed to find the following data stream!'
                        debug_str += msg
                        print(debug_str)

                    f_pos += 6
                    continue

                elif debug:
                    debug_str += f'Data descriptor ":CURVE #" found at {data_start}'  # noqa: E501
                    debug_str += linesep
            elif debug > 0:
                debug_str += f'Data descriptor ":CURV #" found at {data_start}'
                debug_str += linesep

            # Work out the length of the data stream is
            # :CURVE #
            # :CURV #<x><yyy><data><newline>
            f.seek(data_start)
            buf = f.read(20)

            pos_x = buf.find(b'#') + 1
            pos_yyy = pos_x + 1
            desc_len = int(buf[pos_x:pos_yyy])                  # <x>

            pos_data = pos_yyy + desc_len
            data_byte_len = int(buf[pos_yyy:pos_data])          # <yyy>

            # Parse the headers
            # Include ":CURV #<x><yyy>" into the header
            data_start += pos_data
            hdr_len = data_start-hdr_start

            # Read the header byte stream
            f.seek(hdr_start)
            hdr_dict = self.proc_header(f.read(hdr_len).decode())

            # Have a quick guess on the unpack string, this is the most
            # rudimentary information necessary to decode the binary data
            # unpack_str == None if unable to work out from here
            unpack_str = self.get_unpackstr(hdr_dict)
            if debug:
                if unpack_str is None:
                    debug_str += 'Unable to guess how to unpack binary data.'
                    debug_str += linesep
                    debug_str += 'Require the knowledge of at least the following header fields:' + linesep  # noqa: E501
                    debug_str += '\t BYT_O(R)' + linesep
                    debug_str += '\t NR_P(T)' + linesep
                    debug_str += '\t BIT_N(R)' + linesep
                    debug_str += '\t BN_F(MT)' + linesep
                else:
                    debug_str += 'Binary data is thought to be packed as: {unpack_str}'  # noqa: E501
                    debug_str += linesep

            # Assemble into index list
            # idx [hdr_pos, hdr_len, data_pos, data_len, hdr_dict, unpack_str]
            idx.append([hdr_start, hdr_len, data_start, data_byte_len,
                        hdr_dict, unpack_str])

            # Advance to the end of the data stream, prepare for
            # the next search loop
            f_pos = data_start + data_byte_len

            if debug:
                debug_str += f'Continue searching for the next record from {f_pos} byte......'  # noqa: E501
                print(debug_str)
                debug_str = ''

        f.close()

    def __call__(self) -> metaArray:
        return self.__getitem__(0)

    def __len__(self) -> int:
        """
        Return the number of data records in the file
        """
        return int(len(self.idx))

    def __repr__(self) -> str:
        """
        Text representation of the object
        """
        idx = self.idx

        desc = "This is a isf file object." + linesep
        desc += f"\tIt has {len(idx)} record(s):" + linesep

        # [header_pos, header_len, data_pos, data_len, unpack_str, header_dir]
        desc += "\tIndex \tType \tdesc " + linesep
        for i, entry in enumerate(idx):
            # [hdr_pos, hdr_len, data_pos, data_len, hdr_dict, unpack_str]
            hdr_dict = entry[4]

            # PT_F
            PT_F = '\t'
            if 'PT_F' in hdr_dict:
                PT_F = hdr_dict['PT_F']
            elif 'PT_FMT' in hdr_dict:
                PT_F = hdr_dict['PT_FMT']

            # WFI
            WFI = '\t'
            if 'WFI' in hdr_dict:
                WFI = hdr_dict['WFI']
            elif 'WFID' in hdr_dict:
                WFI = hdr_dict['WFID']
            else:
                for key in hdr_dict:
                    if key.find(':WFI') != -1:
                        WFI = hdr_dict[key]
                        break

            desc += "\t" + str(i)
            desc += "\t" + PT_F
            desc += "\t" + WFI + linesep

        if self.debug > 0:
            desc += '=' * 72 + linesep
            desc += 'Detailed record information:' + linesep

            for i, entry in enumerate(idx):
                index_str = "\t" + str(i) + " : "
                hdr_pos, hdr_len, data_pos, data_len, hdr_dict, unpack_str = entry  # noqa: E501

                desc += index_str + "Header location: " + str(hdr_pos) + linesep  # noqa: E501
                desc += index_str + "Header length: " + str(hdr_len) + linesep
                desc += index_str + "Data location: " + str(data_pos) + linesep
                desc += index_str + "Data length: " + str(data_len) + linesep
                desc += index_str + "Unpack string: " + unpack_str + linesep

                for key, val in hdr_dict.items():
                    desc += index_str + '[' + key + '] ' + str(val) + linesep

                desc += linesep

        return desc.rstrip()

    def open(self) -> typing.BinaryIO:
        """
        Open the file for read
        """
        # Check if the file is readable
        try:
            f = open(self.file_path.full, 'rb')
            return f
        except IOError:
            raise IOError("Unable to read file: " + str(self.file_path.full))

    def get_unpackstr(self, hdr_dict: dict) -> str:
        """
        Try to do basic guess of the unpack string from the given header
        dictionary.

        Return None if failed guess the unpack string

        BYT_O, NR_P, BYT_N, BN_F
        """

        if self.debug > 0:
            if 'ENCDG' in hdr_dict:
                ENC = hdr_dict['ENCDG']
            elif 'ENC' in hdr_dict:
                ENC = hdr_dict['ENC']
            else:
                ENC = None

            if ENC != 'BIN' and ENC != 'BINARY':
                print('ENC/ENCDG value is expected to be "BIN" or "BINARY", got "{ENC}" instead.')  # noqa: E501

        # Check byte order / endianness
        if 'BYT_O' in hdr_dict:
            BYT_O = hdr_dict['BYT_O']
        elif 'BYT_OR' in hdr_dict:
            BYT_O = hdr_dict['BYT_OR']
        else:
            return None

        if BYT_O == 'MSB':
            endian = '>'
        elif BYT_O == 'LSB':
            endian = '<'
        else:
            return None

        # Check number of elements in the array
        if 'NR_P' in hdr_dict:
            NR_P = hdr_dict['NR_P']
        elif 'NR_PT' in hdr_dict:
            NR_P = hdr_dict['NR_PT']
        elif ':WFMP:NR_P' in hdr_dict:
            NR_P = hdr_dict[':WFMP:NR_P']
        elif ':WFMPRE:NR_PT' in hdr_dict:
            NR_P = hdr_dict[':WFMPRE:NR_PT']
        else:
            return None

        # Work out bit depth of the array elements
        if 'BIT_N' in hdr_dict:
            BYT_N = hdr_dict['BIT_N'] / 8
        elif 'BIT_NR' in hdr_dict:
            BYT_N = hdr_dict['BIT_NR'] / 8
        elif ':WFMPRE:BYT_NR' in hdr_dict:
            BYT_N = hdr_dict[':WFMPRE:BYT_NR']
        elif ':WFMP:BYT_N' in hdr_dict:
            BYT_N = hdr_dict[':WFMP:BYT_N']
        else:
            return None

        # Work out the data type
        if 'BN_F' in hdr_dict:
            BN_F = hdr_dict['BN_F']
        elif 'BN_FMT' in hdr_dict:
            BN_F = hdr_dict['BN_FMT']
        else:
            return None

        if BN_F == 'RI':
            BN_F = 'int'
        elif BN_F == 'RP':
            BN_F = 'Uint'
        else:
            return None

        try:
            typecode = gettypecode(BYT_N, BN_F)
        except ValueError:
            return None

        return endian + str(NR_P) + typecode

    def proc_header(self, hdr_str: str) -> typing.Union[int, float]:
        """
        Given the header binary read out as string, return header info as dict
        object.

        Attempt to convert string values into numeric values according to
        descriptions in self.hdr_format
        """
        header = hdr_str.strip().rstrip(';').split(';')

        for i in range(len(header)):
            pair = header[i].split(' ', 1)
            pair[1] = pair[1].strip('"' + "'")
            header[i] = pair

        header = dict(header)

        # Convert a list of ASCII text values into numbers
        # It is always safer to convert to float first.
        # int('10000.0') will throw an error
        hdr_format = self.hdr_format

        for key, dtype in hdr_format:
            if key in header:
                header[key] = dtype(Decimal(header[key]))
        return header

    def __getitem__(self, index: int) -> metaArray:
        """
        Return metaArray object of the given index item in the file
        """
        # sanity check
        assert type(index) is int, f"Given index is not int type: {index}"

        idx = self.idx

        # [hdr_pos, hdr_len, data_pos, data_len, hdr_dict, unpack_str]
        hdr_pos, hdr_len, data_pos, data_len, hdr_dict, unpack_str = idx[index]

        if unpack_str is None:
            raise ValueError("Do not know how to decode the data byte stream.")

        # Read in the binary
        with open(self.file_path.full, 'rb') as f:
            f.seek(data_pos)
            data = f.read(data_len)

        data = array(unpack(unpack_str, data))

        # Attempt to scale the data
        # YMU
        if 'YMU' in hdr_dict:
            YMU = hdr_dict['YMU']
        elif 'YMULT' in hdr_dict:
            YMU = hdr_dict['YMULT']
        else:
            YMU = 1

        # YOF
        if 'YOF' in hdr_dict:
            YOF = hdr_dict['YOF']
        elif 'YOFF' in hdr_dict:
            YOF = hdr_dict['YOFF']
        else:
            YOF = 0

        # YZE
        if 'YZE' in hdr_dict:
            YZE = hdr_dict['YZE']
        elif 'YZERO' in hdr_dict:
            YZE = hdr_dict['YZERO']
        else:
            YZE = 0

        data = YZE + YMU * (data - YOF)

        # Attempt to label the data
        data = metaArray(data)

        # data['unit']
        if 'YUN' in hdr_dict:
            data['unit'] = hdr_dict['YUN']
        elif 'YUNIT' in hdr_dict:
            data['unit'] = hdr_dict['YUNIT']

        # data['label']
        if 'COMP' in hdr_dict:
            data['label'] = hdr_dict['COMP']
        elif 'PT_F' in hdr_dict:
            data['label'] = hdr_dict['PT_F']
        elif 'PT_FMT' in hdr_dict:
            data['label'] = hdr_dict['PT_FMT']

        # XUN
        if 'XUN' in hdr_dict:
            data.set_range(0, 'unit', hdr_dict['XUN'])
        elif 'XUNIT' in hdr_dict:
            data.set_range(0, 'unit', hdr_dict['XUNIT'])

        # WFI
        WFI = None
        if 'WFI' in hdr_dict:
            WFI = hdr_dict['WFI']
        elif 'WFID' in hdr_dict:
            WFI = hdr_dict['WFID']
        else:
            for key in hdr_dict:
                if key.find(':WFI') != -1:
                    WFI = hdr_dict[key]
                    break

        # data['name']
        data['name'] = self.file_path.name + '[' + str(index) + ']'

        if WFI is not None:
            chansep = WFI.find(',')
            data.set_range(0, 'label', WFI[:chansep])
            # data['name'] += WFI[chansep:]

        # scale the x-axis

        # XIN
        if 'XIN' in hdr_dict:
            XIN = hdr_dict['XIN']
        elif 'XINCR' in hdr_dict:
            XIN = hdr_dict['XINCR']
        else:
            XIN = 1

        # PT_O
        if 'PT_O' in hdr_dict:
            PT_O = hdr_dict['PT_O']
        elif 'PT_OFF' in hdr_dict:
            PT_O = hdr_dict['PT_OFF']
        else:
            PT_O = 0

        # XZE
        if 'XZE' in hdr_dict:
            XZE = hdr_dict['XZE']
        else:
            XZE = PT_O * -XIN

        data.set_range(0, 'begin', XZE)
        data.set_range(0, 'end', XZE + XIN * len(data))

        # Include the rest of the metainfo into metaArray
        for field, value in hdr_dict.items():
            data["isf."+field] = value

        data.update_range()

        return data


class TDS2000_csv(CSVFile):
    """
    The class define the csv file object saved from the Tek TDS2000
    series scopes

    #
    # TDS2000 series scopes
    #########################
    # ASCII file formats
    #
    # Record Length,    2.500000e+03,           ,   0.000041500000,   0.00000,
    # Sample Interval,  1.000000e-08,           ,   0.000041510000,   0.00146,
    # Trigger Point,    -4.150000000000e+03,    ,   0.000041520000,   0.00146,
    # ,                 ,                       ,   0.000041530000,   0.00146,
    # ,                 ,                       ,   0.000041540000,   0.00146,
    # ,                 ,                       ,   0.000041550000,   0.00146,
    # Source,           CH1,                    ,   0.000041560000,   0.00146,
    # Vertical Units,   V,                      ,   0.000041570000,   0.00146,
    # Vertical Scale,   3.640000e-02,           ,   0.000041580000,   0.00146,
    # Vertical Offset,  0.000000e+00,           ,   0.000041590000,   0.00146,
    # Horizontal Units, s,                      ,   0.000041600000,   0.00146,
    # Horizontal Scale, 2.500000e-06,           ,   0.000041610000,   0.00000,
    # Pt Fmt,           Y,                      ,   0.000041620000,   0.00146,
    # Yzero,            0.000000e+00,           ,   0.000041630000,   0.00146,
    # Probe Atten,      1.000000e+00,           ,   0.000041640000,   0.00146,
    # Model Number,     TDS2024B,               ,   0.000041650000,   0.00146,
    # Serial Number,    C031482,                ,   0.000041660000,   0.00000,
    # Firmware Version, FV:v22.01,              ,   0.000041670000,   0.00146,
    # ,                 ,                       ,   0.000041680000,   0.00000,
    # ,                 ,                       ,   0.000041690000,   0.00146,
    # ,                 ,                       ,   0.000041700000,   0.00146,
    # ,                 ,                       ,   0.000041710000,   0.00000,
    # ,                 ,                       ,   0.000041720000,   0.00146,
    ###########################################################################
    """
    def __init__(self, path: str, debug: bool = False) -> None:

        CSVFile.__init__(self, path=path, debug=debug, analyse=False,
                         field_delimiter=',', text_delimiter='')

        if debug:
            if self.cols != 6:
                print("\t*** Warning, the file does not contain exactly 6 \
                        data columns, it may not be a valid TDS2000 csv file.")

            model = self.getrow(15)[:2]
            if (model[0] != 'Model Number') or (model[1][:4] != 'TDS2'):
                print("\t*** Warning, row 15 of the file does not match the \
                      expected model number description." + linesep +
                      "\tExpecting 'Model':'DPO2', got '" +
                      model[0] + "':'" + model[1][:4] + "' instead.")

        # Create the metainfosv_file
        metainfo = {}
        info_pair = list(zip(self.getcolumn(0), self.getcolumn(1)))

        info_pair.sort(key=itemgetter(0))
        for field, value in groupby(info_pair, key=itemgetter(0)):
            val = list(map(itemgetter(1), value))[0]
            if field == '':
                if val == '':
                    # Blank lines
                    continue
                # Orphan values
                metainfo['Unknown'] = val
            metainfo[field] = val

        # Update the object's metainfo
        self.metainfo.update(metainfo)

        # Convert the ASCII representation into numbers
        Pformat = []
        Pformat.append('Horizontal Scale')
        Pformat.append('Vertical Scale')
        Pformat.append('Sample Interval')
        Pformat.append('Record Length')
        Pformat.append('Trigger Point')
        Pformat.append('Yzero')
        Pformat.append('Vertical Offset')
        Pformat.append('Probe Atten')
        CSVFile.update_metainfo(self, Pformat, float)

        # Although some of the numbers are saved in float representation
        # they really ought to be int instead.
        #
        # Converting a float ASCII repr directly will caused an error.
        #    >>> int('2.500000e+03')
        #    Traceback (most recent call last):
        #      File "<stdin>", line 1, in ?
        #    ValueError: invalid literal for int(): 2.500000e+03
        #
        # Have to convert to float first:
        #   >>> int(float('2.500000e+03'))
        #   2500
        #
        format_cust = []
        format_cust.append('Record Length')
        format_cust.append('Trigger Point')
        CSVFile.update_metainfo(self, format_cust, int)

        if debug:
            if self.metainfo['Record Length'] != self.rows:
                print("\t*** Warning, Record Length description (" +
                      str(self.metainfo['Record Length']) +
                      ") do not match the number of rows counted (" +
                      str(self.rows) +
                      ") in this file. This file maybe corrupted")

    def __call__(self) -> metaArray:
        """
        Return a metaArray when called
        """
        metainfo = self.metainfo
        index = array(self.getcolumn(3), dtype=float)
        data = array(self.getcolumn(4), dtype=float)

        if linear_chk(index) is not True:
            raise ValueError("The index array is not linear")

        # Write the data array as metaArray
        ary = metaArray(data)

        # Update the basic metaArray info
        ary['name'] = self.name
        ary['unit'] = metainfo['Vertical Units']

        # Update the scaling info
        ary['range']['begin'][0] = index[0]
        ary['range']['end'][0] = index[-1]
        ary['range']['unit'][0] = metainfo['Horizontal Units']
        ary['range']['label'][0] = metainfo['Source']

        # Include the rest of the metainfo into metaArray
        for field, value in metainfo.items():
            ary["TDS2.csv."+field] = value

        ary.update_range()
        return ary

    def __getitem__(self, key: int) -> npt.NDArray[float64]:
        """
        Return the requested data as numpy array.

        key = 0     Will return the index array ()
        key = 1     Will return the data array ()
        """

        if key == 0:
            return array(self.getcolumn(3), dtype=float)
        elif key == 1:
            return array(self.getcolumn(4), dtype=float)
        else:
            raise IndexError(f"The only acceptable key values are 0 and 1, given: {key}")  # noqa: E501

    def __repr__(self) -> str:
        """
        Text representation of the object
        """

        desc = 'This is a TDS2000_csv file object.' + linesep
        desc += CSVFile.__repr__(self)

        return desc


class DPO2000_csv(CSVFile):
    """
    The class define the csv file object saved from the Tek DPO2000
    series scopes
    #
    # DPO2000 series scopes
    #########################
    # ASCII file formats
    #
    #    Model,              DPO2014
    #    Firmware Version,   1.25
    #
    #    Point Format,       Y
    #    Horizontal Units,   S
    #    Horizontal Scale,   1e-05
    #    Sample Interval,    1e-09
    #    Filter Frequency,   1e+08
    #    Record Length,      100000
    #    Gating,             0.0% to 100.0%
    #    Probe Attenuation,  1
    #    Vertical Units,     V
    #    Vertical Offset,    0
    #    Vertical Scale,     0.01
    #    Label,
    #    TIME,               CH1
    #    -9.88000e-06,       0.0136
    #    -9.87900e-06,       0.0168
    #    -9.87800e-06,       0.0196
    #    -9.87700e-06,       0.0216
    #    -9.87600e-06,       0.0224
    #    -9.87500e-06,       0.0236
    #    -9.87400e-06,       0.0232
    #    -9.87300e-06,       0.022
    ###########################################################################
    """

    def __init__(self, path: str, data_col: int = 1,
                 debug: bool = False) -> None:
        """
        data_col option gives the option to choose the data column
        from csv data file if more than one oscilloscope channel is used.
        Defaults to prior behaviour (i.e. chooses first data column),
        but this can be changed by calling like so:
        DPO2000_csv('/foo/bar/filename.csv', data_col=2)
        """
        CSVFile.__init__(self, path=path, debug=debug, analyse=True,
                         field_delimiter=',', text_delimiter='')

        if debug:
            if self.cols != 2:
                print("\t*** Warning, the file does not contain exactly 2 \
                        data columns, it may not be a valid DPO2000 csv file.")

            model = self.getrow(0)[:2]
            if (model[0] != 'Model') or (model[1][:4] != 'DPO2'):
                print("\t*** Warning, row 0 of the file does not match the \
                      expected model number description." + linesep +
                      "\tExpecting 'Model':'DPO2', got '" +
                      model[0] + "':'" + model[1][:4] + "' instead.")

            if self.data_start != 16:
                print("\t*** Warning, Data stream is thought to start at row index {self.data_start}, instead of the expected 15.")  # noqa: E501

        self.data_col = data_col
        # File headers should be understood by drv_csv already
        metainfo = self.get_meta_col(self.data_col)

        # Convert the ASCII representation into numbers
        Pformat = []
        Pformat.append('Horizontal Scale')
        Pformat.append('Vertical Scale')
        Pformat.append('Sample Interval')
        Pformat.append('Record Length')
        Pformat.append('Filter Frequency')
        Pformat.append('Vertical Offset')
        Pformat.append('Probe Attenuation')
        CSVFile.update_metainfo(self, Pformat, float)

        # Although some of the numbers are saved in float representation they
        # really ought to be int instead.
        #
        # Converting a float ASCII repr directly will caused an error.
        #    >>> int('2.500000e+03')
        #    Traceback (most recent call last):
        #      File "<stdin>", line 1, in ?
        #    ValueError: invalid literal for int(): 2.500000e+03
        #
        # Have to convert to float first:
        #   >>> int(float('2.500000e+03'))
        #   2500
        #
        format_cust = []
        format_cust.append('Record Length')
        format_cust.append('Probe Attenuation')
        CSVFile.update_metainfo(self, format_cust, int)

        if debug:
            if metainfo['Record Length'] != self.rows - self.data_start:
                print("\t*** Warning, Record Length description (" +
                      str(metainfo['Record Length']) +
                      ") do not match the number of rows counted (" +
                      str(self.rows - self.data_start) +
                      ") in this file. This file maybe corrupted")

        self.metainfo = metainfo

    def __call__(self) -> metaArray:
        """
        Return a metaArray when called
        """
        metainfo = self.metainfo
        rcd_len = metainfo['Record Length']

        index = CSVFile.getcolumn(self, 0)[self.label_row:]
        index_name = index[0]
        index = array(index[1:rcd_len+1], dtype=float)

        try:
            data = CSVFile.getcolumn(self, self.data_col)[self.label_row:]
            metainfo['Source'] = data[0]
            data = array(data[1:rcd_len+1], dtype=float)
        except IndexError as err:
            print("Data column doesn't exist:", err)
            print("Defaulting to first data column")
            data = CSVFile.getcolumn(self, 1)[self.label_row:]
            metainfo['Source'] = data[0]
            data = array(data[1:rcd_len+1], dtype=float)

        if linear_chk(index) is not True:
            raise ValueError("The index array is not linear")

        # Write the data array as metaArray
        ary = metaArray(data)

        # Update the basic metaArray info
        ary['unit'] = metainfo['Vertical Units']
        if metainfo['Label'] == '':
            ary['name'] = self.name
        else:
            ary['name'] = metainfo['Label']

        # Update the scaling info
        ary['range']['begin'][0] = index[0]
        ary['range']['end'][0] = index[-1]
        ary['range']['unit'][0] = metainfo['Horizontal Units']
        ary['range']['label'][0] = index_name

        # Include the rest of the metainfo into metaArray
        for field, value in metainfo.items():
            ary["DPO2.csv."+field] = value

        ary.update_range()
        return ary

    def __repr__(self) -> str:
        """
        Text representation of the object
        """

        desc = 'This is a DPO2000_csv file object.' + linesep
        desc += CSVFile.__repr__(self)

        return desc


# This is provided for compatibility
DPO2000_isf = isf
