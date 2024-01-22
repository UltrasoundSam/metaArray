#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  RitecController.py
#
#  Copyright 2016 Sam Hill <samuel.hill@warwick.ac.uk>
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
#pylint: disable=too-many-instance-attributes, too-many-public-methods
# Lots of control parameters for Ritec, so need to control them all!
"""
Control Class for Ritec Rpr4000 Ultrasonic Pulser-Receiver
"""

import serial
import time

class Ritec4000(object):
    """
    RitecRpr4000    Control class for the RITEC RPR4000 unit.
    Simplifies external PC control for the RITEC RPR4000 unit,
    including checking for:
        - Over voltage condition;
        - Parameter bounds;
        - Parameter changes.

    Can use this class to change a wide range of Ritec parameter, such as:
        Frequency               self.set_frequency(1.23)
        Number of cycles        self.set_cycles(5)
        Operating Mode          self.set_mode('UP:')
        (Un)lock keypad         self.set_keypad('KU:')
        Tracking                self.set_tracking('Y')
        Control                 self.set_control(75)
        Gain                    self.set_gain(70, channel=1)
        High-Pass filter        self.set_hpfilter(2, channel='2')
        Low-Pass filter         self.set_lpfilter(1, channel='1')
        Repetition Rate         self.set_reprate(10)
        Trigger setting         self.set_trigger('I')

    Information about the current settings can be found by replacing 'set'
    with 'get'; i.e. self.get_frequency() returns the current frequency
    value. Or, can get all of the current settings saved into a dict using
    self.get_settings()

    If using a computer trigger setting, can send trigger pulse to Ritec
    by using self.external_pc_trigger().

    Can power down Ritec (for instance, once experiment has finished) by
    calling self.power_down() method.

    NOTE: This class automatically locks the Ritec keypad - to unlock it, use
    self.set_keypad('KU:')

    Once finished with the Ritec, call self.disconnect() to close serial
    connection and unlock Ritec keypad.
    """

    def __init__(self, port, timeout=0.1, retries=2):
        """
        Intiates connection with Ritec over serial port
        Inputs:
            port:               Valid address to serial port
            timeout:            Set timeout value for serial connection
            retries:            The number of tries to send/receive data
                                over serial connection
        """
        self.connected = False
        self.port = port
        self.timeout = float(timeout)
        self.retries = retries

        # Open up the serial port and connect to Ritec
        try:
            #Connect to self.serial_port
            self.serial_port = serial.Serial(\
                                port=self.port, \
                                baudrate=57600, \
                                timeout=timeout, \
                                rtscts=True)
            self.connected = True
        except serial.serialutil.SerialException:
            # Not a valid port number
            print('Not a valid serial port - check available ports')

        self.serial_port.flushInput()	# Clear input buffer

        # Getting current settings from Ritec
        initial_settings = self.get_settings()
        self.frequency = initial_settings['Frequency']
        self.cycles = initial_settings['Cycles']
        self.mode = initial_settings['Mode']
        self.tracking = initial_settings['Tracking']
        self.control = initial_settings['Control']
        self.trigger = initial_settings['Trigger']
        self.rep_rate = initial_settings['Rep Rate']
        self.channel = initial_settings['Channel']
        self.gain = initial_settings['Gain']
        self.hp_filter = initial_settings['HP Filter']
        self.lp_filter = initial_settings['LP Filter']
        self.burst_peak_voltage = initial_settings['Burst Peak Voltage']

        # Lock key-pad so that it can only be changed by computer
        self.set_keypad('KL:')

        # Check over-voltage condition
        self.overvolt = self.overvolt_check()

        # For some reason, Ritec never accepts first freq command - send it now
        self.serial_port.write('FR:00.500000\r')

    def __del__(self):
        """
        Destructor for Ritec control object. I know this is a bad idea
        but need to make sure that front panel is unlocked ('KU:') if Python
        unexpectedly crashes, or if user forgets to use self.disconnect
        """
        self.disconnect()

    def __repr__(self):
        """
        Text representation of object
        """
        desc = (' This is a Ritec RPR-4000 control object ' \
        'connected on port {0}\n'.format(self.port) + 72 * '=')

        desc += ('\n\tFrequency:\t{0:09.6f} MHz\t' \
                 'Cycles:\t{1:04d}\n'.format(self.frequency, self.cycles))

        desc += ('\tControl Level:\t{0:03d}\t\t' \
                 'Gain:\t{1:04.1f} dB\n'.format(self.control, self.gain))

        desc += ('\tActive Channel:\t{0}\t\t' \
                 'RepRate:{1:09.3f} Hz\n'.format(self.channel, self.rep_rate))

        desc += 72 * '='

        return desc

    def _set_parameter(self, cmd):
        """
        Sets a parameter and sends it to the Ritec - Not meant to be
        used directly by user.
        Inputs:
            cmd:                Command to be send to Ritec to set parameter

        Returns:
            Returns boolean value to indicate if change is successful or not
        """
        # Checking overvoltage condition before getting parameters
        self.overvolt = self.overvolt_check()

        result = False
        # Attempt to send parameter change self.Retries times
        for _ in xrange(self.retries):
            try:
                self.serial_port.write(cmd + '\r')
                result = True
                break
            except ValueError:
                # Attempting to use a port that is not open - try to reopen it
                self.connect()
                self.serial_port.write(cmd + '\r')
                result = True
                break
        return result

    def _get_parameter(self, cmd):
        """
        Method for getting desired value from the values of Ritec value -
        - Not meant to be used directly by user.
        Inputs:
            cmd:                Command sent to Ritec to get parameter value

        Returns:
            value:              Value of requested parameter
        """
        time.sleep(0.05)
        # Checking overvoltage condition before getting parameters
        self.overvolt = self.overvolt_check()

        # Will attempt to get parameter self.Retries times
        for _ in xrange(self.retries):
            try:
                self.serial_port.write(cmd + '\r')
                time.sleep(0.5)
                data = self.serial_port.read(16)
            except ValueError:
                # Attempting to use a port that is not open - try to reopen it
                self.connect()
                self.serial_port.write(cmd + '\r')
                time.sleep(0.5)
                data = self.serial_port.read(16)

        self.serial_port.flushInput()
        return data.strip('\r')

    def connect(self):
        """
        Connects the serial port connection to Ritec
        """
        try:
            self.serial_port.open()
            self.connected = True
            self.set_keypad('KL:')
        except serial.serialutil.SerialException:
            print('Serial connection to Ritec is already open')

    def disconnect(self):
        """
        Disconnects the serial connection to Ritec
        """
        self.set_keypad('KU:')
        self.serial_port.close()
        self.connected = False

    def power_down(self):
        """
        Powers down the Ritec by setting control setting
        to 0 after setting tracking to 'Y'.
        """
        self.set_tracking('Y')
        self.set_control(0)

    def overvolt_check(self):
        """
        Checks to see whether Ritec is in overvoltage condition.
        If it is overvolting, the unit is powered down.
        """
        # Checks to see if Ritec is sending over-voltage warning
        result = False
        while self.serial_port.inWaiting() > 0:
            buff = self.serial_port.read(20)
            if buff == '0.V. RESET CONT':
                self.power_down()
                result = True
        return result

    def defaults(self):
        """
        Sets Ritec to known, safe default values
        """
        self.set_mode('UP:')
        self.set_tracking('Y')
        self.set_control(0)
        self.set_trigger('I')
        self.set_frequency(0.3)
        self.set_cycles(3)
        self.set_reprate(10.)
        self.set_gain(30., channel='2')
        self.set_hpfilter(1, channel='2')
        self.set_lpfilter(1, channel='2')
        self.set_gain(30., channel='1')
        self.set_hpfilter(1, channel='1')
        self.set_lpfilter(1, channel='1')
        self.set_keypad('KU:')

    def get_settings(self):
        """
        Get the current Ritec setting and returning them in a dict
        """
        settings = {}
        settings['Frequency'] = self.get_frequency()
        settings['Cycles'] = self.get_cycles()
        settings['Mode'] = self.get_mode()
        settings['Tracking'] = self.get_tracking()
        settings['Control'] = self.get_control()
        settings['Trigger'] = self.get_trigger()
        settings['Rep Rate'] = self.get_reprate()
        settings['Channel'] = self.get_channel()
        settings['Gain'] = self.get_gain()
        settings['HP Filter'] = self.get_hpfilter()
        settings['LP Filter'] = self.get_lpfilter()
        settings['Burst Peak Voltage'] = self.get_bpv()
        return settings

    def set_keypad(self, value):
        """
        Changes the value of Keypad lock
        Input:
            value:                  String; KU; for unlock, or KL: for lock
        """
        if not value in ['KU:', 'KL:']:
            print('Not a valid Keypad option; either KU: or KL:')
            return

        cmd = value
        result = self._set_parameter(cmd)    # Send the command

        if not result:
            print('Ritec Parameter update was not successful!')

    def get_keypad(self):
        """
        Gets the current state of the keypad - either locked or unlocked
        """
        result = self._get_parameter('KS:?')
        if result[0] == 'K':
            return result
        else:
            result = float('nan')
            print('Request failure: could not request Keypad value')

    def set_frequency(self, value):
        """
        Changes the value of the excitation frequency
        Input:
            value:                  Excitation frequency in MHz
                                    (Between 0.03 and 21.999999)
        """
        # Check if frequency is in valid range
        if not 0.03 <= value <= 21.999999:
            print('Frequency not in valid range')
            return

        cmd = 'FR:{0:09.6f}'.format(value)  # Format value to has correct form
        result = self._set_parameter(cmd)   # Send command to change value

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set frequency to object parameters
        self.frequency = self.get_frequency()

    def get_frequency(self):
        """
        Gets the current value of the excitation frequency
        """
        result = self._get_parameter('FR:?')
        # Sanity check the result
        if (result[:3] == 'FR:') and (len(result) == 12):
            freq = float(result[3:])
            self.frequency = freq
        else:
            freq = float('nan')
            print('Request failure: could not request frequency value')

        return freq

    def set_cycles(self, value):
        """
        Changes the number of cycles in excitation signal
        Input:
            value:                Number of cycles (int) between
                                  0 and 4444
        """
        # Check if cycles is in valid range
        if not 0 <= value <= 4444:
            print('Number of cycles not in valid range')
            return

        cmd = 'CY:{0:04d}'.format(value)    # Format value to has correct form
        result = self._set_parameter(cmd)   # Send command to change value

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set cycles to object parameters
        self.cycles = self.get_cycles()

    def get_cycles(self):
        """
        Gets the current value of the number of cycles
        """
        result = self._get_parameter('CY:?')
        # Sanity check the result
        if (result[:3] == 'CY:') and (len(result) == 7):
            cycle = int(result[3:])
            self.cycles = cycle
        else:
            cycle = float('nan')
            print('Request failure: could not request cycle value')

        return cycle

    def set_mode(self, value):
        """
        Changes the console update mode
        Input:                    'UP:' (Update) or 'PS:' (Passthrough) mode
        """
        # Check if valid input
        if not value in ['UP:', 'PS:']:
            print('Console update parameter not in valid range')
            return

        cmd = value
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set cycles to object parameters
        self.mode = self.get_mode()

    def get_mode(self):
        """
        Gets current consol update mode
        """
        result = self._get_parameter('MO:?')
        # Sanity check the result
        if len(result) == 15:
            mode = str(result)
            self.mode = mode
        else:
            mode = 'NaN'
            print('Request failure: could not request Mode setting')

        return mode

    def set_tracking(self, value):
        """
        Sets automatic or manual control of RF gain and bias level
        Input:                    'Y' (automatic) or 'N' (manual)
        """
        # Check if valid input
        if not value in ['Y', 'N']:
            print('Tracking parameter not in valid range')
            return

        cmd = 'TK:{0}'.format(value)
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set tracking to object parameters
        self.tracking = self.get_tracking()

    def get_tracking(self):
        """
        Gets current tracking mode
        """
        result = self._get_parameter('TK:?')
        # Sanity check the result
        if (result[:3] == 'TK:') and (len(result) == 4):
            track = str(result[3])
            self.tracking = track
        else:
            track = 'NaN'
            print('Request failure: could not request Tracking value')

        return track

    def set_control(self, value):
        """
        Sets control level for Ritec pulser amplitude level
        Input:                    Control level between 0 and 100 (int)
        """
        # Check that tracking is set to 'Y', else can't set control
        if self.tracking is not 'Y':
            print('Tracking is not set to "Y" - cannot set control level')
            return

        # Check if valid inputhttp://www.bbc.co.uk/news/world-us-canada-35829477
        if not 0 <= value <= 100:
            print('Control level parameter not in valid range')
            return

        cmd = 'CO:{0:03d}'.format(int(value))
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set control level to object parameters
        self.control = self.get_control()

    def get_control(self):
        """
        Gets current control level
        """
        result = self._get_parameter('CO:?')
        # Sanity check the result
        if (result[:3] == 'CO:') and (len(result) == 6):
            control = int(result[3:])
            self.control = control
        else:
            control = float('nan')
            print('Request failure: could not request Control level')

        return control

    def set_trigger(self, value):
        """
        Sets the trigger method for Ritec pulser - either internal or external
        Input:                    'I' (Internal), 'E' (External) or 'C' (RS232)
        """
        # Check if valid input
        if not value in ['I', 'E', 'C']:
            print('Trigger parameter not in valid range')
            return

        cmd = 'TG:{0}'.format(value)
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.trigger = self.get_trigger()

    def get_trigger(self):
        """
        Gets current Trigger setting
        """
        result = self._get_parameter('TG:?')
        # Sanity check the result
        if (result[:3] == 'TG:') and (len(result) == 6):
            trig = str(result[3:])
            self.trigger = trig
        else:
            trig = 'NaN'
            print('Request failure: could not request Trigger setting')

        return trig

    def set_reprate(self, value):
        """
        Sets the repetition rate of the Ritec
        Input:                Repetition rate in Hz (0.08 - 10000 Hz)
        """
        # Check if valid input
        if not 0.08 <= value <= 10000.:
            print('RepRate parameter not in valid range')
            return

        cmd = 'RR:{0:09.3f}'.format(value)
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.rep_rate = self.get_reprate()

    def get_reprate(self):
        """
        Gets current Trigger setting
        """
        result = self._get_parameter('RR:?')
        # Sanity check the result
        if (result[:3] == 'RR:') and (len(result) == 12):
            rep = float(result[3:])
            self.rep_rate = rep
        else:
            rep = float('nan')
            print('Request failure: could not request RepRate setting')

        return rep

    def set_channel(self, value):
        """
        Sets the active receiver input channel
        Input:                    Enter desired channel: '1', '2', or
                                  'A' (for alternating channels)
        """
        # Check if valid input
        if not value in ['1', '2', 'A']:
            print('Channel parameter not in valid range')
            return

        cmd = 'IN:{0}'.format(value)
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.channel = self.get_channel()

    def get_channel(self):
        """
        Gets current active channel
        """
        result = self._get_parameter('IN:?')
        # Sanity check the result
        if (result[:3] == 'IN:') and (len(result) == 4):
            channel = str(result[3])
            self.channel = channel
        else:
            channel = 'NaN'
            print('Request failure: could not request Channel setting')

        return channel

    def set_gain(self, value, channel=None):
        """
        Sets the amplifier gain value on a desired channel
        Inputs:
            value:          Value of gain that is desired
            channel:        Channel to which gain is applied
        """
        # Check that input is valid
        if not 20. <= value <= 99.9:
            print('Amplifier Gain not in valid range')
            return

        # If not channel is defined, use currently active channel
        if channel is None:
            channel = self.channel

        # If channel is 'A', need to apply it to each channel separately
        if channel is 'A':
            self.set_gain(value, channel='1')
            self.set_gain(value, channel='2')
            self.set_channel('A')
            return

        # Change active channel to the one requested
        self.set_channel(channel)
        cmd = 'GA:{0:04.1f}'.format(value)
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.gain = self.get_gain()

    def get_gain(self):
        """"
        Gets amplifier gain of current active channel
        """
        result = self._get_parameter('GA:?')
        # Sanity check the result
        if (result[:3] == 'GA:') and (len(result) == 7):
            gain = float(result[3:])
            self.gain = gain
        else:
            gain = float('nan')
            print('Request failure: could not request Gain setting')

        return gain

    def set_hpfilter(self, value, channel=None):
        """
        Set High Pass Filter value on desired channel
        Input:
            value                 Integer value: 1, 2, 3, 4 for 1st, 2nd...etc
                                  HP filter option. Changes with different
                                  Ritec units.
            channel               Apply HP Filter to this channel. Defaults
                                  to active channel.
        """
        # Check that input is valid
        if not value in range(1, 5):
            print('HP Filter setting not in valid range')
            return

        # If not channel is defined, use currently active channel
        if channel is None:
            channel = self.channel

        # Change active channel to the one requested
        self.set_channel(channel)
        cmd = 'HF:{0:1d}'.format(int(value))
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.hp_filter = self.get_hpfilter()

    def get_hpfilter(self):
        """
        Gets HP Filter setting on active channel
        """
        result = self._get_parameter('HF:?')
        # Sanity check the result
        if (result[:3] == 'HF:'):
            hpf = str(result[3:])
            self.hp_filter = hpf
        else:
            hpf = 'NaN'
            print('Request failure: could not request HPF setting')

        return hpf

    def set_lpfilter(self, value, channel=None):
        """
        Set Low Pass Filter value on desired channel
        Input:
            value                 Integer value: 1, 2, 3, 4 for 1st, 2nd...etc
                                  LP filter option. Changes with different
                                  Ritec units.
            channel               Apply LP Filter to this channel. Defaults
                                  to active channel.
        """
        # Check that input is valid
        if not value in range(1, 5):
            print('LP Filter setting not in valid range')
            return

        # If not channel is defined, use currently active channel
        if channel is None:
            channel = self.channel

        # Change active channel to the one requested
        self.set_channel(channel)
        cmd = 'LF:{0:1d}'.format(int(value))
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.lp_filter = self.get_lpfilter()

    def get_lpfilter(self):
        """
        Gets LP Filter setting on active channel
        """
        result = self._get_parameter('LF:?')
        # Sanity check the result
        if (result[:3] == 'LF:'):
            lpf = str(result[3:])
            self.lp_filter = lpf
        else:
            lpf = float('nan')
            print('Request failure: could not request LPF setting')

        return lpf

    def get_bpv(self):
        """
        Gets Burst Peak Voltage from Ritec
        """
        result = self._get_parameter('BV:?')
        # Sanity check the result
        if (result[:3] == 'BV:') and (len(result) == 8):
            bpv = float(result[3:])
            self.burst_peak_voltage = bpv
        else:
            bpv = float('nan')
            print('Request failure: could not request BPV setting')

        return bpv

    def external_pc_trigger(self):
        """
        Sends a Trigger pulse if the Trigger setting is set to 'C' for
        computer control trigger setting.
        """
        #if self.trigger == 'XPC':
            #result = self._set_parameter('CT:')
            #if not result:
                #print('Trigger pulse was unsuccessful')
        #else:
            #print('Trigger setting is not set to "C"')
        pass
