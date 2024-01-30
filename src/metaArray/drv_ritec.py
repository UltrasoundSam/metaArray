# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 2024 14:05

@author: samhill

Control Class for Ritec Rpr4000 Ultrasonic Pulser-Receiver
"""

import serial
import time


class Ritec4000:
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

    def __init__(self, port: str, timeout: float = 0.1,
                 retries: int = 2) -> None:
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
            # Connect to self.serial_port
            self.serial_port = serial.Serial(port=self.port,
                                             baudrate=57600,
                                             timeout=timeout,
                                             rtscts=True)
            self.connected = True
        except serial.serialutil.SerialException:
            # Not a valid port number
            print('Not a valid serial port - check available ports')

        # Clear input buffer
        self.serial_port.flushInput()

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

    def __del__(self) -> None:
        """
        Destructor for Ritec control object. I know this is a bad idea
        but need to make sure that front panel is unlocked ('KU:') if Python
        unexpectedly crashes, or if user forgets to use self.disconnect
        """
        self.disconnect()

    def __repr__(self) -> str:
        """
        Text representation of object
        """
        desc = ' This is a Ritec RPR-4000 control object '
        desc += f'connected on port {self.port}'
        desc += 72 * '='

        desc += f'\n\tFrequency:\t{self.frequency:09.6f} MHz\tCycles:\t{self.cycles:04d}\n'  # noqa: E501

        desc += f'\tControl Level:\t{self.control:03d}\t\tGain:\t{self.gain:04.1f} dB\n'  # noqa: E501

        desc += f'\tActive Channel:\t{self.channel}\t\tRepRate:{self.rep_rate:09.3f} Hz\n'  # noqa: E501

        desc += 72 * '='

        return desc

    def _set_parameter(self, cmd: str) -> bool:
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

        # Change cmd into bytes
        cmd = cmd.encode('utf-8')

        result = False
        # Attempt to send parameter change self.Retries times
        for _ in range(self.retries):
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

    def _get_parameter(self, cmd: str) -> str:
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

        # Convert command to bytes
        cmd = cmd.encode('utf-8')

        # Will attempt to get parameter self.Retries times
        for _ in range(self.retries):
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
        return data.strip(b'\r').decode('utf-8')

    def connect(self) -> None:
        """
        Connects the serial port connection to Ritec
        """
        try:
            self.serial_port.open()
            self.connected = True
            self.set_keypad('KL:')
        except serial.serialutil.SerialException:
            print('Serial connection to Ritec is already open')

    def disconnect(self) -> None:
        """
        Disconnects the serial connection to Ritec
        """
        self.set_keypad('KU:')
        self.serial_port.close()
        self.connected = False

    def power_down(self) -> None:
        """
        Powers down the Ritec by setting control setting
        to 0 after setting tracking to 'Y'.
        """
        self.set_tracking('Y')
        self.set_control(0)

    def overvolt_check(self) -> bool:
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

    def defaults(self) -> None:
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

    def get_settings(self) -> dict:
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

    def set_keypad(self, value: str) -> None:
        """
        Changes the value of Keypad lock
        Input:
            value:                  String; KU; for unlock, or KL: for lock
        """
        if value not in ['KU:', 'KL:']:
            print('Not a valid Keypad option; either KU: or KL:')
            return

        cmd = value
        result = self._set_parameter(cmd)    # Send the command

        if not result:
            print('Ritec Parameter update was not successful!')

    def get_keypad(self) -> str:
        """
        Gets the current state of the keypad - either locked or unlocked
        """
        result = self._get_parameter('KS:?')
        if result[0] == 'K':
            return result
        else:
            result = float('nan')
            print('Request failure: could not request Keypad value')

    def set_frequency(self, value: float) -> None:
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

        cmd = f'FR:{value:09.6f}'           # Format value to has correct form
        result = self._set_parameter(cmd)   # Send command to change value

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set frequency to object parameters
        self.frequency = self.get_frequency()

    def get_frequency(self) -> float:
        """
        Gets the current value of the excitation frequency
        """
        result = self._get_parameter('FR:?')
        # Sanity check the result
        if (result.startswith('FR:')) and (len(result) == 12):
            freq = float(result[3:])
            self.frequency = freq
        else:
            freq = float('nan')
            print('Request failure: could not request frequency value')

        return freq

    def set_cycles(self, value: int) -> None:
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

        cmd = f'CY:{value:04d}'             # Format value to has correct form
        result = self._set_parameter(cmd)   # Send command to change value

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set cycles to object parameters
        self.cycles = self.get_cycles()

    def get_cycles(self) -> int:
        """
        Gets the current value of the number of cycles
        """
        result = self._get_parameter('CY:?')
        # Sanity check the result
        if (result.startswith('CY:')) and (len(result) == 7):
            cycle = int(result[3:])
            self.cycles = cycle
        else:
            cycle = float('nan')
            print('Request failure: could not request cycle value')

        return cycle

    def set_mode(self, value: str) -> None:
        """
        Changes the console update mode
        Input:                    'UP:' (Update) or 'PS:' (Passthrough) mode
        """
        # Check if valid input
        if value not in ['UP:', 'PS:']:
            print('Console update parameter not in valid range')
            return

        cmd = value
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set cycles to object parameters
        self.mode = self.get_mode()

    def get_mode(self) -> str:
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

    def set_tracking(self, value: str) -> None:
        """
        Sets automatic or manual control of RF gain and bias level
        Input:                    'Y' (automatic) or 'N' (manual)
        """
        # Check if valid input
        if value not in ['Y', 'N']:
            print('Tracking parameter not in valid range')
            return

        cmd = 'TK:{value}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set tracking to object parameters
        self.tracking = self.get_tracking()

    def get_tracking(self) -> str:
        """
        Gets current tracking mode
        """
        result = self._get_parameter('TK:?')
        # Sanity check the result
        if (result.startswith('TK:')) and (len(result) == 4):
            track = str(result[3])
            self.tracking = track
        else:
            track = 'NaN'
            print('Request failure: could not request Tracking value')

        return track

    def set_control(self, value: int) -> None:
        """
        Sets control level for Ritec pulser amplitude level
        Input:                    Control level between 0 and 100 (int)
        """
        # Check that tracking is set to 'Y', else can't set control
        if self.tracking != 'Y':
            print('Tracking is not set to "Y" - cannot set control level')
            return

        # Check if valid input
        if not 0 <= value <= 100:
            print('Control level parameter not in valid range')
            return

        cmd = f'CO:{int(value):03d}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set control level to object parameters
        self.control = self.get_control()

    def get_control(self) -> int:
        """
        Gets current control level
        """
        result = self._get_parameter('CO:?')
        # Sanity check the result
        if (result.startswith('CO:')) and (len(result) == 6):
            control = int(result[3:])
            self.control = control
        else:
            control = float('nan')
            print('Request failure: could not request Control level')

        return control

    def set_trigger(self, value: str) -> None:
        """
        Sets the trigger method for Ritec pulser - either internal or external
        Input:                    'I' (Internal), 'E' (External) or 'C' (RS232)
        """
        # Check if valid input
        if value not in ['I', 'E', 'C']:
            print('Trigger parameter not in valid range')
            return

        cmd = f'TG:{value}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.trigger = self.get_trigger()

    def get_trigger(self) -> str:
        """
        Gets current Trigger setting
        """
        result = self._get_parameter('TG:?')
        # Sanity check the result
        if (result.startswith('TG:')) and (len(result) == 6):
            trig = str(result[3:])
            self.trigger = trig
        else:
            trig = 'NaN'
            print('Request failure: could not request Trigger setting')

        return trig

    def set_reprate(self, value: float) -> None:
        """
        Sets the repetition rate of the Ritec
        Input:                Repetition rate in Hz (0.08 - 10000 Hz)
        """
        # Check if valid input
        if not 0.08 <= value <= 10000.:
            print('RepRate parameter not in valid range')
            return

        cmd = f'RR:{value:09.3f}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.rep_rate = self.get_reprate()

    def get_reprate(self) -> float:
        """
        Gets current Trigger setting
        """
        result = self._get_parameter('RR:?')
        # Sanity check the result
        if (result.startswith('RR:')) and (len(result) == 12):
            rep = float(result[3:])
            self.rep_rate = rep
        else:
            rep = float('nan')
            print('Request failure: could not request RepRate setting')

        return rep

    def set_channel(self, value: str) -> None:
        """
        Sets the active receiver input channel
        Input:                    Enter desired channel: '1', '2', or
                                  'A' (for alternating channels)
        """
        # Check if valid input
        if value not in ['1', '2', 'A']:
            print('Channel parameter not in valid range')
            return

        cmd = f'IN:{value}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.channel = self.get_channel()

    def get_channel(self) -> str:
        """
        Gets current active channel
        """
        result = self._get_parameter('IN:?')
        # Sanity check the result
        if (result.startswith('IN:')) and (len(result) == 4):
            channel = str(result[3])
            self.channel = channel
        else:
            channel = 'NaN'
            print('Request failure: could not request Channel setting')

        return channel

    def set_gain(self, value: float,
                 channel: str = None) -> None:
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
        if channel == 'A':
            self.set_gain(value, channel='1')
            self.set_gain(value, channel='2')
            self.set_channel('A')
            return

        # Change active channel to the one requested
        self.set_channel(channel)
        cmd = f'GA:{value:04.1f}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.gain = self.get_gain()

    def get_gain(self) -> float:
        """"
        Gets amplifier gain of current active channel
        """
        result = self._get_parameter('GA:?')
        # Sanity check the result
        if (result.startswith('GA:')) and (len(result) == 7):
            gain = float(result[3:])
            self.gain = gain
        else:
            gain = float('nan')
            print('Request failure: could not request Gain setting')

        return gain

    def set_hpfilter(self, value: int, channel: str = None) -> None:
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
        if value not in range(1, 5):
            print('HP Filter setting not in valid range')
            return

        # If not channel is defined, use currently active channel
        if channel is None:
            channel = self.channel

        # Change active channel to the one requested
        self.set_channel(channel)
        cmd = f'HF:{int(value):1d}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.hp_filter = self.get_hpfilter()

    def get_hpfilter(self) -> str:
        """
        Gets HP Filter setting on active channel
        """
        result = self._get_parameter('HF:?')
        # Sanity check the result
        if result.startswith('HF:'):
            hpf = str(result[3:])
            self.hp_filter = hpf
        else:
            hpf = 'NaN'
            print('Request failure: could not request HPF setting')

        return hpf

    def set_lpfilter(self, value: int, channel: str = None) -> None:
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
        if value not in range(1, 5):
            print('LP Filter setting not in valid range')
            return

        # If not channel is defined, use currently active channel
        if channel is None:
            channel = self.channel

        # Change active channel to the one requested
        self.set_channel(channel)
        cmd = f'LF:{int(value):1d}'
        result = self._set_parameter(cmd)

        if not result:
            print('Ritec Parameter update was not successful!')

        # Assign new set trigger to object parameters
        self.lp_filter = self.get_lpfilter()

    def get_lpfilter(self) -> str:
        """
        Gets LP Filter setting on active channel
        """
        result = self._get_parameter('LF:?')
        # Sanity check the result
        if result.startswith('LF:'):
            lpf = str(result[3:])
            self.lp_filter = lpf
        else:
            lpf = float('nan')
            print('Request failure: could not request LPF setting')

        return lpf

    def get_bpv(self) -> float:
        """
        Gets Burst Peak Voltage from Ritec
        """
        result = self._get_parameter('BV:?')
        # Sanity check the result
        if (result.startswith('BV:')) and (len(result) == 8):
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
        # if self.trigger == 'XPC':
        #    result = self._set_parameter('CT:')
        #    if not result:
        #        print('Trigger pulse was unsuccessful')
        # else:
        #    print('Trigger setting is not set to "C"')
        pass
