# -*- coding: utf-8 -*-

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
This file contain a number of drivers to communicate with the
HEWLETT-PACKARD Agilent 4294A Precision Impedance Analyzer

Package dependency:
        numpy
        scipy
'''
import warnings

import socket   # for sockets
import time     # for sleep
import select   # select.slect to wait for socket to be ready

from struct import unpack

from numpy import array, isclose

from metaArray.core import metaArray
from metaArray.misc import gettypecode, timestamp

class hp4294a:
    """
    ######################
    # MEAS {IMPH|IRIM|LSR|LSQ|CSR|CSQ|CSD|AMPH|ARIM|LPG|LPQ|CPG|CPQ|CPD|COMP
    #       |IMLS|IMCS|IMLP|IMCP|IMRS|IMQ|IMD|LPR|CPR}
    #
    # Command |   Trace A |   Trace B
    # --------------------------------
    # IMPH    |   |Z|     |   θ
    # IRIM    |   R       |   X
    # LSR     |   Ls      |   Rs
    # LSQ     |   Ls      |   Q
    # CSR     |   Cs      |   Rs
    # CSQ     |   Cs      |   Q
    # CSD     |   Cs      |   D
    # AMPH    |   |Y|     |   θ
    # ARIM    |   G       |   B
    # LPG     |   Lp      |   G
    # LPQ     |   Lp      |   Q
    # CPG     |   Cp      |   G
    # CPQ     |   Cp      |   Q
    # CPD     |   Cp      |   D
    # COMP    |   Z       |   Y
    # IMLS    |   |Z|     |   Ls
    # IMCS    |   |Z|     |   Cs
    # IMLP    |   |Z|     |   Lp
    # IMCP    |   |Z|     |   Cp
    # IMRS    |   |Z|     |   Rs
    # IMQ     |   |Z|     |   Q
    # IMD     |   |Z|     |   D
    # LPR     |   Lp      |   Rp
    # CPR     |   Cp      |   Rp


    ######################
    # BWFACT {1|2|3|4|5}
    # Sets the bandwidth. To set the bandwidth of each segment when creating
    # the list sweep table, also use this command.
    #
    # Description
    # 1 (Initial value) Specifies bandwidth 1 (shortest measurement time).
    # 2 Specifies bandwidth 2.
    # 3 Specifies bandwidth 3.
    # 4 Specifies bandwidth 4.
    # 5 Specifies bandwidth 5 (longest measurement time, accurate measurement).


    ######################
    # AVER {ON|OFF|1|0}
    # Enables/disables the sweep averaging function.
    #
    # Description
    # ON or 1
    # Enables the sweep averaging function.
    # OFF or 0 (Initial value) Disables the sweep averaging function.
    #
    ######################
    # AVERFACT <numeric>
    #
    # Description
    # Sets the averaging factor of the sweep averaging function.
    # Averaging factor
    # Range 1 to 256
    # Initial value 16
    # Resolution 1
    #
    ######################
    # AVERREST
    #
    # Description
    # Resets the data count used in averaging calculation of the
    # sweep averaging function to 0. Measured data before the execution of
    # this command is not used in averaging calculation. If this command is
    # executed while the 4294A is performing a sweep, it is restarted. (No
    # query)



    ######################
    # TRGS {INT|EXT|BUS|MAN}
    #
    # Description
    # Selects a trigger source.
    #
    # INT   (initial value) Specifies the internal trigger.
    # EXT   Specifies the external trigger inputted from the EXT TRIGGER
    #       terminal on the rear panel.
    # BUS   Specifies the GPIB/LAN trigger (trigger by executing the “*TRG”
    #       command on page 261).
    # MAN   Specifies the manual trigger (trigger by the following key sequence
    #       on the front panel: [Trigger] - SOURCE [ ] - MANUAL).
    #
    ######################
    # CONT
    #
    # Description
    # Sets the sweep mode to the auto continuous sweep (CONT). In this mode,
    # sweeps are repeated automatically and continuously.


    # The format applicable when you read measurement parameter settings
    # from the Agilent 4294A (as when you read the sweep start point with
    # “STAR?”) is the ASCII format, regardless of which data transfer format
    # has been specified.
    #
    # You can select either the ASCII format (default) or one of the binary
    # formats for reading measurement data, waveform analysis results and so
    # on from the Agilent 4294A (as when you read a data trace array with
    # “OUTPDTRC?”). As for binary formats, you can select the IEEE 32-bit
    # floating point format, IEEE 64-bit floating point format, or MS-DOS
    # personal computer format as the appropriate format for your
    # controller. Use the following commands for selecting a desired data
    # transfer format:
    #
    # Data Transfer Format                  Command
    # -------------------------------------------------------
    # IEEE 32-bit floating point format     “FORM2”
    # IEEE 64-bit floating point format     “FORM3”
    # ASCII format (Default)                “FORM4”
    # MS-DOS personal computer format       “FORM5”


    ######################
    # POIN <numeric>
    #
    # Description
    # Sets the number of points measured at each sweep. To set the
    # number-of-points setting of each segment when creating the list sweep
    # table, also use this command.
    #
    # Range 2 to 801 (Note on the number-of-points setting of a segment. The
    # upper limit is the smaller value: value obtained by subtracting the
    # sum of the numbers of points of already set segments from 801 or 201.)
    #
    # Initial value 201 (Note on the number-of-points setting of a segment.
    # If the maximum number of settable points is less than 201, the maximum
    # number of settable points.)
    #
    # Resolution 1

    ######################
    # TRAC {A|B}
    #
    # Description
    # Sets the active trace.
    # Parameters        Description
    # A (initial value) Specifies trace A as the active trace.
    # B                 Specifies trace B as the active trace.

    ######################
    # *CLS
    #
    # Description
    # Clears the error queue, Status Byte Register, Operation Status
    # Register, Standard Event Status Register, and Instrument Event Status
    # Register. This command has the same function as the “CLES” command on
    # page 279. (No query)


    ######################
    # *IDN?
    #
    # Description
    # Reads out the manufacturer, model number, serial number, and firmware
    # version number of the 4294A. (Query only)
    #
    # Query response {string 1},{string 2},{string 3},{string 4}<newline><^END>
    #
    # Readout data is as follows:
    # {string 1}    Manufacturer.   HEWLETT-PACKARD is always read out.
    # {string 2}    Model number.   4294A is always read out.
    # {string 3}    10-digit serial number (example: JP1KF00101).
    # {string 4}    Firmware version number (example: 01.00).


    ######################
    # *TRG
    #
    # Description
    # If the trigger mode is set to GPIB/LAN (set to BUS with the “TRGS”
    # command on page 459), triggers the 4294A waiting for a trigger.
    # (No query)


    ######################
    # PRES
    #
    # Description Resets to the preset state. The preset state is almost the
    # same as that of the reset using the "*RST” command on page 260, though
    # there are some differences shown bellow. (No query)
    #
    # - The sweep mode is set to CONT.
    # - The HP Instrument BASIC is not reset.

    """

    def __init__(self, host, port, debug=False, timeout=10, PRES=False):

        """
        host        Destination instrument host name or IP address
        port        Destination instrument TCP/IP socket port number
        debug
        PRES        Reset the instrument to default state at init
        sweep       Sweep type, default to 'LIN'. 'LOG' is only available for frequency sweep.
        """
        # Init variables
        ################
        self.host = host
        self.port = port
        self.timeout = float(timeout)
        self.debug = debug
        #self.chk_opc = False        # Set True if overlap commands are set
                                    # Unsafe to request data untill they are cleared

        # Constants
        #################
        # Acceptable list of measurement values and their corresponding parameters

        meas_dict = {}
        meas_dict['IMPH'] = ['|Z|', 'θ',  'Ω', '°']
        meas_dict['IRIM'] = ['R',   'X',  None, None]
        meas_dict['LSR']  = ['Ls',  'Rs', 'H', 'Ω']
        meas_dict['LSQ']  = ['Ls',  'Q',  'H', '']
        meas_dict['CSR']  = ['Cs',  'Rs', 'F', 'Ω']
        meas_dict['CSQ']  = ['Cs',  'Q',  'F', '']
        meas_dict['CSD']  = ['Cs',  'D',  'F', '']
        meas_dict['AMPH'] = ['|Y|', 'θ',  'S', '°']
        meas_dict['ARIM'] = ['G',   'B',  '', '']
        meas_dict['LPG']  = ['Lp',  'G',  'H', '']
        meas_dict['LPQ']  = ['Lp',  'Q',  'H', '']
        meas_dict['CPG']  = ['Cp',  'G',  'F', '']
        meas_dict['CPQ']  = ['Cp',  'Q',  'F', '']
        meas_dict['CPD']  = ['Cp',  'D',  'F', '']
        meas_dict['COMP'] = ['Z',   'Y',  '', '']
        meas_dict['IMLS'] = ['|Z|', 'Ls', 'Ω', 'H']
        meas_dict['IMCS'] = ['|Z|', 'Cs', 'Ω', 'F']
        meas_dict['IMLP'] = ['|Z|', 'Lp', 'Ω', 'H']
        meas_dict['IMCP'] = ['|Z|', 'Cp', 'Ω', 'F']
        meas_dict['IMRS'] = ['|Z|', 'Rs', 'Ω', 'Ω']
        meas_dict['IMQ']  = ['|Z|', 'Q',  'Ω', '']
        meas_dict['IMD']  = ['|Z|', 'D',  'Ω', '']
        meas_dict['LPR']  = ['Lp',  'Rp', 'F', 'Ω']
        meas_dict['CPR']  = ['Cp',  'Rp', 'H', 'Ω']

        self.meas_dict = meas_dict

        cmd_lst = []
        cmd_lst.append('*CLS')
        if PRES is True:
            cmd_lst.append('PRES')

        cmd_lst.append('*IDN?')

        self.IDN = self.send_cmd(cmd_lst).strip()

        return

    def __repr__(self):
        """
        Text representation of the object
        """
        desc = "This is a HEWLETT-PACKARD Agilent 4294A Precision Impedance Analyzer TCP/IP socket driver object." + linesep
        desc += "\tIt has the host address of: " + str(self.host) + ':' + str(self.port) + linesep
        desc += "\tAnd the poitIt has the host address of: " + str(self.host) + linesep

        return desc.rstrip()

    def __call__(self):
        return self.get_data_set()

    def send_cmd(self, cmd_lst, reply=None, reply_len=None, timeout=None):
        """
        cmd_lst         Python list for commands to send to the instrument
        reply           True if expecting a reply, guess if None
        reply_len       Length (in bytes) of the the expected reply message, guess if None
        timeout         Socket timeout, default to self.timeout
        """

        debug = self.debug
        if timeout is None:
            timeout = self.timeout

        msg = ';'.join(cmd_lst)

        if reply is None:           # Guess if a reply is expected
            if msg[-1] is '?':
                reply = True        # Expecting a reply
            else:
                reply = False       # Send and forget

        msg += '\n'           # Complete the msg

        retry = 5                           # Retry five times before giving up connecting to the instrument
        ################################################################
        while True:
            retry -= 1
            try:
                sock = socket.create_connection((self.host, self.port))     # Connect
                sock.setblocking(0)                                         # Non-blocking socket
                sock.sendall(msg)                                           # Send msg
                if debug: print('(-> Socket established to instrument, command sent: ' + str(repr(msg)))
                break
            except socket.error, err_msg:                                   # Failed to connect
                print('Unable to complete sending of commands to the instrument. Error message : ' + err_msg[1])
                if retry == 0:              # Ran out of retry attempts
                    try:
                        sock.close()
                        if debug: print('(-x Closed socket to instrument.')
                    except:
                        pass
                    finally:
                        raise RuntimeError("socket connection broken")
                else:
                    print(str(retry) + ' retry attempts remaining. Wait 2 seconds before retry.')
                    time.sleep(2)           # Sleep for 2 seconds before trying again

        ################################################################

        if reply is False:                  # Not expecting to receive anything,
            sock.close()                    # close socket straight away
            if debug:
                print('*** Reply is not expected.')
                print('(-x Closed socket to instrument.')
            return None
        else:                               # Expecting a reply
            if debug: print('*** Reply is expected.')
            data = None
            try:
                data = self.receive(sock, exp_len=reply_len, timeout=timeout)  # Wait for answers
            finally:
                sock.close()                    # Clean up
                if debug: print('(-x Closed socket to instrument.')
            return data

    def receive(self, sock, exp_len=None, timeout=None):
        """
        Receiver doesnt close the socket, it is up to the sender to decide

        When nbytes is not specified, and unable to guess the expected
        length of data, whatever is received is assumed complete and returned.

        sock            Opened socket to listen to
        exp_len         Number of bytes expected to receive. Will try to
                        guess if None.
        timeout         If nbytes is not given, and nothing is received
                        give up receiving after timeout amount in seconds.
                        default to self.timeout
        """

        debug = self.debug
        if timeout is None:
            timeout = self.timeout
        else:
            timeout = float(timeout)

        chunks = []
        d_len = 0               # Length of msg received so far

        if exp_len is None:
            exp_len = -1            # Expected number of bytes in the received message
            is_bin_data = False     # Count bytes if msg is binaray, look for '\n' otherwise.
        else:
            is_bin_data = True

        is_ascii_data = False       # We don't know yet

        data = ''                   # Init received byte stream holder
        while True:
            if is_bin_data:                             # looking for binaray data, count bytes
                if debug: print('### Byte counting, received [' + str(d_len) + ':' + str(exp_len) + '].')
                if d_len >= exp_len:
                    #print('### Received sufficient amount of data')
                    break

            ready = select.select([sock], [], [], timeout)       # Wait for more data
            if ready[0]:
                data = sock.recv(4096)
                d_len += len(data)
                chunks.append(data)

                #print('### Received ' + str(d_len) + 'bytes')
                if is_bin_data:
                    continue                            #1 Binary content, keep receiving.
                elif is_ascii_data:
                    if data[-1] == '\n':                #2 Found '\n'
                        if debug: print('### Found ASCII termination char \\n')
                        msg = ''.join(chunks)
                        break
                    else:
                        if debug: print('### Continue to look for ASCII termination char \\n')
                        continue
                else:                                   #1 ASCII content (asumed)
                    msg = ''.join(chunks)
                    if msg[:2] == '#6':                 #2 Looks like binary data is expected
                        if debug: print('### Receiver found "#6"')
                        if d_len < 8:                   #3 Should have received 8 char len header
                            if debug: print('### incomplete header yet')
                            continue
                        else:                           #3 Got enough to analyse the header
                            try:                        #4 Assemble the message received so far
                                exp_len = int(msg[2:8]) #4 Try to guess the expected singal-data length
                            except:
                                continue                #4 Really couldnt guess, assume ASCII

                            is_bin_data = True          #3 Message header seem complete
                            exp_len += 9                #3 '#6xxxxxx' + singal-data length + '\n'
                            if debug: print('### Receiver excpecting ' + str(exp_len) + ' bytes.')
                    else:
                        if data[-1] == '\n':                #2 Found '\n'
                            if debug: print('### Found ASCII termination char \\n')
                            msg = ''.join(chunks)
                            break
                        else:
                            is_ascii_data = True
                            if debug: print('### Assume ASCII')
                            if debug: print('### Continue to look for ASCII termination char \\n')
                            continue
            else:                                       # Waited too long
                msg = ''.join(chunks)
                if len(data) == 0:                      #1 Didnt get any data, timeout reached
                    raise RuntimeError("socket connection timeout, did not receive anything.")
                else:                                   #1 Got some data, timeout reached
                    if is_bin_data:
                        raise RuntimeError('socket connection timeout, received [' + str(d_len) + ':' + str(exp_len) + '].')

                    if debug: print('### Got some data, timeout reached, assume complete.')
                    break

        msg = ''.join(chunks)
        if debug: print('Received a total of ' + str(len(msg)) + 'bytes.')
        return msg

    def PRES(self):
        '''
        Send PRES command
        '''
        self.send_cmd(['PRES', '*OPC?'])
        return

    def chk_ready(self, timeout=None):
        """
        Check if the instrument has completed all overlap commands

        Unsafe to retrive data until *OPC? return 1
        """
        debug = self.debug

        if debug: print('Waiting for the completion of all overlap commands.')

        if timeout is None:         # No specific timeout, keep trying
            retry = 10              # Retry 10 times before giving up
            while True:
                if retry == 0:
                    raise RuntimeError('Given up waiting for the instrument to be ready.')
                try:
                    self.send_cmd(['*OPC?'], timeout=20)
                except Exception as detail:
                    if debug: print('Keep waiting...')
                    retry -= 1
                    continue

                break               # Got successful reply
        else:                       # Have a specific timeout limit
            self.send_cmd(['*OPC?'], timeout=timeout)

        return

    def cmd_meas(self, measurement='IMPH'):
        """
        Set measurement type.

        # MEAS {IMPH|IRIM|LSR|LSQ|CSR|CSQ|CSD|AMPH|ARIM|LPG|LPQ|CPG|CPQ|CPD|COMP
        #       |IMLS|IMCS|IMLP|IMCP|IMRS|IMQ|IMD|LPR|CPR}
        """

        if self.meas_dict.has_key(measurement):
            return 'MEAS ' + measurement
        else:
            raise ValueError(str(repr(measurement)) + " is not a recognised measurement type.")

    def set_meas(self, measurement):
        """
        Set measurement resolution. See cmd_meas(measurement)
        """
        mes = self.send_cmd([self.cmd_meas(measurement), 'MEAS?']).strip()

        if measurement != mes:
            warnings.warn('Requested measurement (' + str(measurement) + \
                          ') does not match the instrument state: ' + \
                          str(mes), RuntimeWarning)
        return mes

    def get_meas(self):
        """
        MEAS?
        Query the current measurement type.
        """
        return self.send_cmd(['MEAS?']).strip()

    def cmd_resolution(self, points=512):
        """
        2 to 801 (Note on the number-of-points setting of a segment. The
        upper limit is the smaller value: value obtained by subtracting the
        sum of the numbers of points of already set segments from 801 or
        201.)
        """
        pts = int(round(points))
        pts = min(max(2, pts), 801)

        if self.debug:
            if pts != points:
                warnings.warn('Requested resolution (' + str(points) + ') is coerced into: ' + str(pts), \
                              RuntimeWarning)

        return 'POIN ' + str(pts)

    def set_resolution(self, points):
        """
        Set measurement resolution. See cmd_resolution(points)
        """
        res = self.send_cmd([self.cmd_resolution(points), '*WAI', 'POIN?'])

        if points != int(res):
            warnings.warn('Requested resolution (' + str(points) + \
                          ') does not match the instrument state: ' + \
                          str(res), RuntimeWarning)
        return res

    def get_resolution(self):
        """
        POIN?
        Query the number of points in the measurement.
        """
        return int(self.send_cmd(['POIN?']).strip())

    def cmd_bandwidth(self, bandwidth=1):
        """
        BWFACT {1|2|3|4|5}
        Sets the bandwidth. To set the bandwidth of each segment when creating
        the list sweep table, also use this command.

        Description
        1 (Initial value) Specifies bandwidth 1 (shortest measurement time).
        2 Specifies bandwidth 2.
        3 Specifies bandwidth 3.
        4 Specifies bandwidth 4.
        5 Specifies bandwidth 5 (longest measurement time, accurate measurement).
        """

        bw = int(round(bandwidth))
        bw = min(max(1, bw), 5)

        if self.debug:
            if bw != bandwidth:
                warnings.warn('Requested bandwidth (' + str(bandwidth) + ') is coerced into: ' + str(bw), \
                              RuntimeWarning)

        return 'BWFACT ' + str(bw)

    def set_bandwidth(self, bandwidth):
        """
        Set measurement bandwidth factor. See cmd_bandwidth(bandwidth)
        """
        bw = self.send_cmd([self.cmd_bandwidth(bandwidth), '*WAI', 'BWFACT?']).strip()

        if bandwidth != int(bw):
            warnings.warn('Requested bandwidth (' + str(bandwidth) + \
                          ') does not match the instrument state: ' + \
                          str(bw), RuntimeWarning)
        return int(bw)

    def set_active(self, trac='A'):
        """
        Set active data trace
        """

        if trac == 'A':
            tr = self.send_cmd(['TRAC A', 'TRAC?']).strip()
        elif trac == 'B':
            tr = self.send_cmd(['TRAC B', 'TRAC?']).strip()
        else:
            raise ValueError(str(trac) + ' is not a recognised measurement trace identifier to set active.')

        if trac != tr:
            warnings.warn('Requested bandwidth active trace identifier (' + str(trac) + \
                             ') does not match the instrument state: ' + \
                             str(tr), RuntimeWarning)

        return tr

    def get_bandwidth(self):
        """
        BWFACT?
        Query the bandwidth factor in the measurement.
        """
        return int(self.send_cmd(['BWFACT?']).strip())

    def _set_average_factor(self, average):
        """
        NOT INTENDED FOR DIRECT USE, TRY set_average() INSTEAD.

        Sets the averaging factor value. Sensible value is dependent
        on other averaging related conditions. See set_average() for
        more details.
        """
        avg = self.send_cmd(['AVERFACT ' + str(average), 'AVERFACT?']).strip()

        if average != int(avg):
            warnings.warn('Requested averaging factor (' + str(average) + \
                          ') does not match the instrument state: ' + \
                          str(avg), RuntimeWarning)
        return int(avg)

    def get_averfact(self):
        """
        AVERFACT?
        Query the averaging factor in the measurement.
        """
        return int(self.send_cmd(['AVERFACT?']).strip())

    def _set_aver(self, state=1):
        """
        NOT INTENDED FOR DIRECT USE, TRY set_average() INSTEAD.

        Sets the averaging state. Sensible value is dependent
        on other averaging related conditions. See set_average() for
        more details.
        """
        aver = self.send_cmd(['AVER ' + str(int(state)), 'AVER?']).strip()

        if state != int(aver):
            warnings.warn('Requested averaging state (' + str(state) + \
                          ') does not match the instrument state: ' + \
                          str(aver), RuntimeWarning)
        return int(aver)

    def get_aver(self):
        """
        AVER
        """
        return int(self.send_cmd(['AVER?']).strip())

    def cmd_form(self, form='FORM3'):
        """
        Data Transfer Format                  Command
        -------------------------------------------------------
        IEEE 32-bit floating point format     “FORM2”
        IEEE 64-bit floating point format     “FORM3”
        ASCII format (Default)                “FORM4”
        MS-DOS personal computer format       “FORM5”
        """

        d_format = form.upper()
        if d_format in ['FORM2', 'FORM3', 'FORM4', 'FORM5']:
            return d_format
        else:
            raise ValueError(form + " is not a recognised Data Transfer Format type.")

        return

    def set_form(self, form='FORM3'):
        """
        Set data transfer format
        """
        return self.send_cmd([self.cmd_form(form)])

    def _chk_star(self, sta):
        """
        NOT INTENDED FOR DIRECT USE, TRY set_sweep() INSTEAD.

        Sets the sweep range start value. Acceptable value is dependent
        on other sweep conditions. See set_sweep() for more details.
        """
        start = self.get_start()

        if not isclose(sta, float(start), rtol=1e-11, atol=1e-3):
            warnings.warn('Requested sweep start (' + str(sta) + \
                          ') does not match the instrument state: ' + \
                          str(start), RuntimeWarning)
        return start

    def _chk_stop(self, stp):
        """
        NOT INTENDED FOR DIRECT USE, TRY set_sweep() INSTEAD.

        Sets the sweep range stop value. Acceptable value is dependent
        on other sweep conditions. See set_sweep() for more details.
        """
        stop = self.get_stop()

        if not isclose(stp, float(stop), rtol=1e-11, atol=1e-3):
            warnings.warn('Requested sweep start (' + str(sta) + \
                          ') does not match the instrument state: ' + \
                          str(stop), RuntimeWarning)
        return stop

    def _chk_swpt(self, swpt):
        """
        NOT INTENDED FOR DIRECT USE, TRY set_sweep() INSTEAD.

        Sets the sweep type. Acceptable value is dependent on other
        sweep conditions. See set_sweep() for more details.
        """
        sweep_type = self.get_swpt()

        if swpt != sweep_type:
            warnings.warn('Requested sweep type (' + str(swpt) + \
                          ') does not match the instrument state: ' + \
                          str(sweep_type), RuntimeWarning)
        return sweep_type

    def _chk_swpp(self, swpp):
        """
        NOT INTENDED FOR DIRECT USE, TRY set_sweep() INSTEAD.

        Sets the sweep parameter. Acceptable value is dependent on other
        sweep conditions. See set_sweep() for more details.
        """
        sweep_para = self.get_swpp()

        if swpp != sweep_para:
            warnings.warn('Requested sweep parameter (' + str(swpp) + \
                          ') does not match the instrument state: ' + \
                          str(sweep_para), RuntimeWarning)
        return sweep_para

    def get_swpp(self):
        """
        SWPP?
        Query the sweep type in the measurement.
        """
        return self.send_cmd(['SWPP?']).strip()


    def get_start(self):
        """
        STAR?
        Query the sweep range start value.
        """
        return float(self.send_cmd(['STAR?']).strip())

    def get_stop(self):
        """
        stop?
        Query the sweep range stop value.
        """
        return float(self.send_cmd(['stop?']).strip())

    def get_swpt(self):
        """
        SWPT?
        Query the sweep type in the measurement.
        """
        return self.send_cmd(['SWPT?']).strip()

    def get_trgs(self):
        """
        # TRGS {INT|EXT|BUS|MAN}
        #
        # Description
        # Selects a trigger source.
        #
        # INT   (initial value) Specifies the internal trigger.
        # EXT   Specifies the external trigger inputted from the EXT TRIGGER
        #       terminal on the rear panel.
        # BUS   Specifies the GPIB/LAN trigger (trigger by executing the “*TRG”
        #       command on page 261).
        # MAN   Specifies the manual trigger (trigger by the following key sequence
        #       on the front panel: [Trigger] - SOURCE [ ] - MANUAL).
        #
        TRGS?
        Query the trigger source in the measurement.
        """
        return self.send_cmd(['TRGS?']).strip()

    def set_cont(self):
        """
        #############################
        # CONT
        #
        # Sets the sweep mode to the auto continuous sweep (CONT). In
        # this mode, sweeps are repeated automatically and continuously.
        """
        cont = self.send_cmd(['CONT', 'CONT?']).strip()

        if int(cont) != 1:
            raise RuntimeError('Requested auto continuous sweep, ' + \
                               'but the instrument state (CONT? -> ' + \
                               str(cont) + ') does not reflect this.')
        return int(cont)

    def set_sing(self, timeout=None):
        """
        #############################
        # SING
        #
        # Performs a single sweep. After the sweep, the sweep mode goes to HOLD. (No query)

        # Sets the sweep mode to the auto continuous sweep (CONT). In
        # this mode, sweeps are repeated automatically and continuously.
        """
        if timeout is None:
            timeout = self.timeout

        self.send_cmd(['SING', '*OPC?'], timeout=timeout)

        return


    def get_cont(self):
        """
        CONT?
        Query the sweep mode in the measurement.
        """
        return int(self.send_cmd(['CONT?']).strip())

    def parse_output(self, data, data_points):
        """
        Parse the numerical data array into numpy array

        data            Byte stream from TCP/IP socket
        data_points     Number of sweep points, only used to calculate
                        data type for binary tranfers.

        OUTPDTRC?

        Description
        Reads out the values of all measurement points in a data trace
        array (refer to “Internal data arrays” on page 81).

        Query response
        {numeric 1},{numeric 2},..,{numeric NOP×2-1},{numeric NOP×2}<newline><^END>

        Reads out the readout and subsidiary readout of the measurement
        parameter value of each measurement point as shown below.
        Where, NOP is the number of points, and n is an integer between
        1 and NOP.

        {numeric n×2-1}:    Readout of the n-th measurement point. If
        the measurement parameter is a scalar value (for other than
        COMPLEX Z-Y), the measurement parameter value is read out. If
        the measurement parameter is a vector value (for COMPLEX Z-Y),
        resistance (for trace A) or conductance (for trace B) is read out.

        {numeric n×2}:      Subsidiary readout of the n-th measurement
        point. If the measurement parameter is a scalar value (for other
        than COMPLEX Z-Y), 0 is always read out. If the measurement
        parameter is a vector value (for COMPLEX Z-Y), reactance (for
        trace A) or susceptance (for trace B) is reads out.
        """
        debug = self.debug

        # Try to parse the received data
        if data[:2] == '#6':            # Check if it is one of the binary formats
            if debug: print('Found binary header, expecting binary data.')
            d_length = int(data[2:8])   # Read the header
            unpack_str = '>' + str(data_points) + gettypecode(d_length / data_points, 'float')
            ary = array(unpack(unpack_str, data[8:d_length+8]))
        else:
            if debug: print('Parser assuming ASCII data')
            try:
                ary = array(data.strip().split(',')).astype(float)  # Assume ASCII data
            except Exception as detail:
                print('Caught error trying to parse received data as ASCII mode.')
                print(str(type(detail)) + ': ' + str(detail.args))
                print('Expecting ' + str(data_points) + 'data points.')
                print('Received byte stream as follows: ')
                raise ValueError(str(data))

        return ary

    def get_sweep_values(self, resolution=None):
        """
        OUTPSWPRM?
        Syntax OUTPSWPRM?
        Description Reads out the sweep parameter values of all measurement points. (Query only)
        Query response {numeric 1},{numeric 2},..,{numeric NOP}<newline><^END>
        Where, NOP is the number of points.
        """

        if resolution is None:
            resolution = self.get_resolution()

        self.chk_ready()        # Make sure the instrument is ready before requesting data

        return self.parse_output(self.send_cmd(['OUTPSWPRM?']), resolution)

    def get_active_data_trace(self, resolution=None):
        """
        OUTPSWPRM?
        Syntax OUTPSWPRM?
        Description Reads out the sweep parameter values of all measurement points. (Query only)
        Query response {numeric 1},{numeric 2},..,{numeric NOP}<newline><^END>
        Where, NOP is the number of points.
        """
        if resolution is None:
            resolution = self.get_resolution()

        # There are twice the number of floats vs measurement resolution
        # See parse_output() for more details
        ary = self.parse_output(self.send_cmd(['OUTPDTRC?']), 2 * resolution)

        ary = ary.reshape((resolution, 2)).transpose()
        if (ary[1] == 0).all():
            return ary[0]               # Complex part is all zero
        else:
            return ary[0]+1.0j*ary[1]   # Return complex array

    def get_data_set(self, form='FORM3', sweep=True):
        """
        Return the current measurement data
        """
        resolution = self.get_resolution()

        # self.send_cmd([self.cmd_form(form)])

        self.set_active('A')
        data_A = self.get_active_data_trace(resolution)

        self.set_active('B')
        data_B = self.get_active_data_trace(resolution)

        if sweep is True:
            data_sweep = self.get_sweep_values(resolution)
            return data_A, data_B, data_sweep
        else:
            return data_A, data_B

    def set_sweep(self, start=None, stop=None, swpt=None, swpp=None):
        """
        Set sweep parameters given, keep the remaining untouched.
        Return the instrument sweep parameters at the end

        Safe guard sensible sweep parameters

        #############################
        # STAR <numeric>[HZ|MHZ|V|A]
        #
        # Sets the sweep range start value.
        # When the sweep parameter is frequency
        # 40 to 110E6 (for linear sweep)
        # 40 to 109.9998E6 (for log sweep)
        # Initial value 40
        # Unit Hz
        # Resolution 1E-3

        #############################
        # STOP <numeric>[HZ|MHZ|V|A]
        #
        # Sets the sweep range stop value.
        # When the sweep parameter is frequency
        # 40 to 110E6 (for linear sweep)
        # 60 to 110E6 (for log sweep)
        # Initial value 110E6
        # Unit Hz
        # Resolution 1E-3

        #############################
        # SWPT {LIN|LOG|LIST}
        #
        # Sets the sweep type.
        # LIN   Specifies the linear sweep.
        # LOG   Specifies the log sweep (settable only for frequency sweep).
        # LIST  Specifies the list sweep.

        #############################
        # SWPP {FREQ|OLEV|DCB}
        #
        # Sets the sweep parameter.
        # FREQ  (initial value) Specifies the frequency sweep.
        # OLEV  Specifies the oscillator (OSC) level sweep.
        # DCB   Specifies the dc bias level sweep.
        """

        #print(start, stop, swpt, swpp)

        # Init
        if start is None:
            flg_start = False
            start = self.get_start()
        else:
            flg_start = True

        if stop is None:
            flg_stop = False
            stop = self.get_stop()
        else:
            flg_stop = True

        if swpt is None:
            flg_swpt = False
            swpt = self.get_swpt()
        else:
            flg_swpt = True

        if swpp is None:
            flg_swpp = False
            swpp = self.get_swpp()
        else:
            flg_swpp = True

        #print(start, stop, swpt, swpp)

        # Check the conditions
        swpt = swpt.upper()
        if swpt == 'LIN':
            pass
        elif swpt == 'LOG':
            pass
        elif swpt == 'LIST':
            raise NotImplementedError('List sweep is currently unsupported.')
        else:
            raise ValueError(str(repr(sweep)) + " is not a recognised sweep type.")

        swpp = swpp.upper()
        if swpp.upper() == 'FREQ':
            pass
        elif swpp.upper() == 'OLEV':
            raise NotImplementedError('Oscillator level sweep. is currently unsupported.')
            #return 'SWPP OLEV'
        elif swpp.upper() == 'DCB':
            raise NotImplementedError('DC bias level sweep is currently unsupported.')
            #return 'SWPP DCB'
        else:
            raise ValueError(str(repr(swpp)) + " is not a recognised sweep parameter.")

        # Sweep condition check
        if not start < stop:
            raise ValueError('Start sweep value (' + str(start) + ') must be less than stop sweep value (' + str(stop) + ').')

        # The following is currently not needed as FREQ sweep is the only implemented sweep type
        # if swpt == 'LOG' and swpp != 'FREQ':
        #    raise ValueError('LOG sweep is only settable for frequency sweep.')

        # The following is currently not needed as FREQ sweep is the only implemented sweep type
        # if swpp == 'FREQ':
        if swpt == 'LOG':
            sta = min(max(40, start), 109.9998E6)
            stp = min(max(40, stop), 109.9998E6)
        else:
            sta = min(max(40, start), 110E6)
            stp = min(max(60, stop), 110E6)
        # The following is currently not needed as FREQ sweep is the only implemented sweep type
        # elif swpp == 'OLEV':
        # elif swpp == 'DCB':

        if self.debug:
            if sta != start:
                warnings.warn('Requested ' + swpt + ' sweep start (' + str(start) + ') is coerced into: ' + str(sta), \
                              RuntimeWarning)

            if stp != stop:
                warnings.warn('Requested ' + swpt + ' sweep stop (' + str(stop) + ') is coerced into: ' + str(stp), \
                              RuntimeWarning)

        if not start < stop:
            raise ValueError('Start sweep value (' + str(sta) + ') must be less than stop sweep value (' + str(stp) + ').')


        # Set the changes
        cmd_lst = []
        start = self.send_cmd(['STAR {0:+E}'.format(sta), 'STAR?']).strip()
        stop = self.send_cmd(['STOP {0:+E}'.format(stp), 'STOP?']).strip()
        sweep_para = self.send_cmd(['SWPP ' + swpp, 'SWPP?']).strip()
        sweep_type = self.send_cmd(['SWPT ' + swpt, 'SWPT?']).strip()

        if flg_start:
            cmd_lst.append['STAR {0:+E}'.format(sta)]

        if flg_stop:
            cmd_lst.append['STOP {0:+E}'.format(stp)]

        if flg_swpt:
            cmd_lst.append['SWPP ' + swpp]

        if flg_swpp:
            cmd_lst.append['SWPT ' + swpt]

        if cmd_lst != []:
            cmd_lst.append('*OPC?')
            self.send_cmd(cmd_lst)
        else:
            # No need to set anything, return the current state
            return sta, stp, swpt, swpp


        if flg_start:
            sta = self._chk_star(sta)

        if flg_stop:
            stp = self._chk_stop(stp)

        if flg_swpt:
            swpt = self._chk_swpt(swpt)

        if flg_swpp:
            swpp = self._chk_swpp(swpp)

        return sta, stp, swpt, swpp

    def set_average(self, averfact=16):
        """
        Safe guard sensible averaging parameters

        ######################
        # AVER {ON|OFF|1|0}
        # Enables/disables the sweep averaging function.
        #
        # Description
        # ON or 1
        # Enables the sweep averaging function.
        # OFF or 0 (Initial value) Disables the sweep averaging function.
        #
        ######################
        # AVERFACT <numeric>
        #
        # Description
        # Sets the averaging factor of the sweep averaging function.
        # Averaging factor
        # Range 1 to 256
        # Initial value 16
        # Resolution 1
        #
        ######################
        # AVERREST
        #
        # Description
        # Resets the data count used in averaging calculation of the
        # sweep averaging function to 0. Measured data before the execution of
        # this command is not used in averaging calculation. If this command is
        # executed while the 4294A is performing a sweep, it is restarted. (No
        # query)
        """
        avg = int(round(averfact))

        if avg > 1:
            avg = min(max(1, avg), 256)

            if self.debug:
                if avg != averfact:
                    warnings.warn('Requested averaging factor (' + str(averfact) + ') is coerced into: ' + str(avg), \
                                  RuntimeWarning)

            self._set_aver(1)
            avg = self._set_average_factor(avg)
        else:
            self._set_aver(0)

        return avg

    def get_powe(self):
        """
        POWE <numeric>[V|A]

        Sets the oscillator (OSC) power level. To set the oscillator power level of each segment
        when creating the list sweep table, also use this command. To select voltage or current to
        set the level, use the “POWMOD” command on page 404.
        POWE?

        Query the oscillator power mode in the measurement.
        """
        return float(self.send_cmd(['POWE?']).strip())

    def get_powmod(self):
        """
        POWMOD {VOLT|CURR}
        POWMOD?
        Selects voltage or current to set the oscillator (OSC) power level. To set the oscillator
        power level setting method of each segment when creating the list sweep table, use this
        command.

        Query the oscillator power mode in the measurement.
        """
        return self.send_cmd(['POWMOD?']).strip()

    def get_dcmod(self):
        """
        DCMOD
        DCMOD {VOLT|CURR|CVOLT|CCURR}
        DCMOD?
        Selects the dc bias output mode. To set the dc bias output mode of each segment when
        creating the list sweep table, also use this command.

        Query the DC bias mode in the measurement.
        """
        return self.send_cmd(['DCMOD?']).strip()

    def get_dcv(self):
        """
        DCV <numeric>[V]
        DCV?
        Sets the dc bias output level when the dc bias output mode is the voltage mode or
        constant-voltage mode. To set the dc bias output level of each segment when creating the
        list sweep table, also use this command.)

        Query the DC bias voltage in the measurement.
        """
        return float(self.send_cmd(['DCV?']).strip())

    def get_dci(self):
        """
        DCI <numeric>[A]
        DCI?
        Sets the dc bias output level when the dc bias output mode is the current mode or
        constant-current mode. To set the dc bias output level of each segment when creating the
        list sweep table, also use this command.
        <numeric>
        Description
        Output current value of dc bias
        Range
        -0.1 to 0.1
        Initial value
        0 (Note that the initial value is the current set value of the dc bias
        output current, when creating segment 1; the set value of the
        previous segment, when creating an additional segment.)
        Unit
        A (ampere)
        Resolution
        20E-6
        If the specified parameter is out of the allowable setting range, the minimum value (if the
        lower limit of the range is not reached) or the maximum value (if the upper limit of the
        range is exceeded) is set.
        {numeric}<newline><^END>

        Query the DC bias current in the measurement.
        """
        return float(self.send_cmd(['DCI?']).strip())

    def get_dco(self):
        """
        DCO {ON|OFF|1|0}
        DCO?
        Turns on/off the dc bias output.

        Query the DC bias state in the measurement.
        """
        return int(self.send_cmd(['DCO?']).strip())

    def eta(self):
        """
        Based of current instrument settings, try to estimate the time
        required to complete the measurements.
        """

        retry = 5
        while True:
            retry -= 1
            if retry == 0: raise RuntimeError('Unable to obtain sensible frequency sweep values.')

            f_ary = self.get_sweep_values()

            if (f_ary == 0).any():          # Just in case it is not yet ready
                if self.debug: print('*** Found zeros in frequency sweep values. ' + str(retry) + ' retry attrmpts remaining.')
            else:
                break

        bw = self.get_bandwidth()

        if bw == 1:
            eta = (30 / f_ary + 3.2e-3).sum() + 15
        elif bw == 2:
            eta = (30 / f_ary + 7.8e-3).sum() + 15
        elif bw == 3:
            eta = (24.8 / f_ary + 14.4e-3).sum() + 15
        elif bw == 4:
            eta = (24.8 / f_ary + 51.4e-3).sum() + 15
        elif bw == 5:
            eta = (24.8 / f_ary + 160e-3).sum() + 15
        else:
            raise ValueError('Unexpected measurement bandwidth setting ' + \
                             '(' + str(bw) + '), unable to estimate required measurement duration.')

        if self.get_aver() == 1: eta *= self.get_averfact()

        return int(round(eta))

    def get_meta_info(self, meta_info={}):
        """
        Return generic meta info as dict
        """

        meta_info['hp4294a.time'] = timestamp()
        meta_info['hp4294a.host'] = str(self.host) + ':' + str(self.port)

        meta_info['hp4294a.*IDN?'] = self.IDN

        meta_info['hp4294a.MEAS?'] = self.get_meas()
        meta_info['hp4294a.POIN?'] = self.get_resolution()
        meta_info['hp4294a.BWFACT?'] = self.get_bandwidth()

        sta, stp, swpt, swpp = self.set_sweep()

        meta_info['hp4294a.STAR?'] = sta
        meta_info['hp4294a.STOP?'] = stp
        meta_info['hp4294a.SWPT?'] = swpt
        meta_info['hp4294a.SWPP?'] = swpp

        if self.get_aver() == 1:
            meta_info['hp4294a.AVER?'] = 1
            meta_info['hp4294a.AVERFACT?'] = self.get_averfact()
        else:
            meta_info['hp4294a.AVER?'] = 0

        meta_info['hp4294a.OUTPERRO?'] = self.send_cmd(['OUTPERRO?']).strip()

        meta_info['hp4294a.TRGS?'] = self.get_trgs()
        meta_info['hp4294a.CONT?'] = self.get_cont()

        return meta_info

    def get_meta_info_freq_sweep(self, meta_info={}):
        """
        Return meta info including frequency sweep specific ones
        """
        meta_info = self.get_meta_info(meta_info)

        meta_info['hp4294a.POWMOD?'] = self.get_powmod()
        meta_info['hp4294a.POWE?'] = self.get_powe()

        dco = self.get_dco()
        meta_info['hp4294a.DCO?'] = dco
        if dco == 1:
            dcmod = self.get_dcmod()
            meta_info['hp4294a.DCMOD?'] = dcmod
            if dcmod[-4:] == 'VOLT':
                meta_info['hp4294a.DCV?'] = self.get_dcv()
            else:
                meta_info['hp4294a.DCI?'] = self.get_dci()

        return meta_info

    def meas_impedance(self, start=0.5e6, stop=50e6, resolution=512, \
                             average=0, bandwidth=1, sweep='LIN', form='FORM3'):
        """
        High level function
        impedance

        start           500kHz
        stop            50MHz
        resolution      512 points
        average         Averaging factor (1-256), default to Zero (disabled)
        bandwidth       Measurement bandwidth (1-5), default to 1 (shortest measurement time)
        sweep           Sweep type {LIN|LOG|LIST}, List type is unsupported.

        """
        meas = self.set_meas('IMPH')

        self.set_form(form)

        sta, stp, swpt, swpp = self.set_sweep(start=start, stop=stop, swpt=sweep, swpp='FREQ')

        res = self.set_resolution(resolution)

        bw = self.set_bandwidth(bandwidth)

        avg = self.set_average(average)

        eta = self.eta()

        if eta > (3 * self.timeout):
            print('Expecting to wait approximately ' + str(eta) + ' seconds to acquire the signal.')

        timeout = max(eta*1.25, self.timeout)

        # Begin the measurement
        if avg > 1:
            self.send_cmd(['AVERREST'])
            self.send_cmd(['NUMG ' + str(avg)])
            self.chk_ready(timeout=timeout)
            #self.send_cmd(['*OPC?'], timeout=timeout)
            avg_str = ' (' + str(avg) + ' averages)'
        else:
            self.send_cmd(['SING'])
            self.chk_ready(timeout=timeout)
            # self.send_cmd(['*OPC?'], timeout=timeout)
            avg_str = ''

        data_A, data_B, data_sweep = self.get_data_set(form=form)

        #lbl_A, lab_B, unit_A, unit_B = self.meas_dict['IMPH']

        data_A = metaArray(data_A)
        data_B = metaArray(data_B)

        data_A['name'] = 'Reactance measurement' + avg_str
        data_A['unit'] = 'Ohm'             # unicode(unit_A, encoding='utf-8')
        data_A['label'] = 'Reactance |Z|'   # unicode(lbl_A, encoding='utf-8')
        data_A.set_range(0, 'begin', data_sweep[0])
        data_A.set_range(0, 'end', data_sweep[-1])
        data_A.set_range(0, 'unit', 'Hz')
        data_A.set_range(0, 'label', 'Frequency')

        data_B['name'] = 'Phase angle measurement' + avg_str
        data_B['unit'] = 'Degree'           # unicode(unit_B, encoding='utf-8')
        data_B['label'] = 'Phase angle'     # unicode(lbl_A, encoding='utf-8')
        data_B.set_range(0, 'begin', data_sweep[0])
        data_B.set_range(0, 'end', data_sweep[-1])
        data_B.set_range(0, 'unit', 'Hz')
        data_B.set_range(0, 'label', 'Frequency')

        if swpt == 'LOG':
            data_A.set_range(0, 'log', True)
            data_B.set_range(0, 'log', True)

        meta_info = self.get_meta_info_freq_sweep()

        data_A.update(meta_info)
        data_B.update(meta_info)

        return data_A, data_B
