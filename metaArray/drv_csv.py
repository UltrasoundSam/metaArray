#       drv_csv.py
#
#       Copyright 2009 charley <y.fan@warwick.ac.uk>
#
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
File for controlling csv file reading
'''
from itertools import groupby
from os import linesep
from operator import itemgetter

from numpy import array

from metaArray.misc import filePath

def open_csv(path, mode, debug=False, field_delimiter=',', text_delimiter='"', linesep="\r\n"):
    """
    CSV file parallel of the build in open function
    Inputs:
        path            Path of the CSV file
        mode            {'r'|'w'}
        linesep         new line char

    In Windows platforms, a newline is typically encoded as \r\n
    In Linux/Unix/OS X, a newline is typically encoded as \n
    In RISC OS spooled text output, a newline is typically encoded as \n\r

    Any well writen program should be able to understand any of the above conventions

    Output:
        The csv file object
    """
    if mode == 'r':
        return csv_file(path, debug=debug, field_delimiter=field_delimiter, text_delimiter=text_delimiter)
    elif mode == 'w':
        raise NotImplementedError
        return csv_write(path, debug=debug, field_delimiter=field_delimiter, text_delimiter=text_delimiter, linesep="\r\n")
    else:
        raise NotImplementedError

    return

#class csv_write(object):
    #"""
    #CSV file class. Write CSV files.
    #"""

    #def __init__(self, path = None, debug = False, \
            #field_delimiter = ',', text_delimiter = '"', linesep = "\r\n"):

        ## User options
        #self.path = filePath(path)
        #self.debug = debug
        #self.field_delimiter = field_delimiter
        #self.text_delimiter = text_delimiter
        #self.linesep = linesep


        ## self.mode = 'ab'                    # Default file openening mode is append

        #if not self.path.write:
            #print "Given file path is not writable." + self.path.full
            #raise IOError

        ## Really try to make sure it is writable
        ##f = open(self.path.full, 'wb')
        ##f.write('')
        ##f.close()

        #return

    #def write(self, rows):
        #"""
        #Write the data into csv file.

        #The data (rows) must come in a two dimensional list-like object, such as:

        #[[1,2], [3,4], [5,6]]


        #"""
        #mode = self.mode







        #f = open(self.path.full, mode)





        #f.close()

        #if mode == 'wb':                    # Reset after first write
            #self.mode = 'ab'

        #return

    #def close():
        #"""
        #Specifically close the file, subsequent writes will over wite the existing file.

        #This file object is normally closed, except for when the write() method is
        #active, and will be automatically close after each write, this is done to reduce
        #the risk content corruption in the event of exceptions when the file handle may
        #be left open.

        #This close() method is only provided for consistancy with the typical file
        #object behaviour.
        #"""
        #self.mode = 'wb'                    # Change the next write to over write, instead of append.
        #return


class csv_file(object):
    """
    CSV file class. Reads CSV files.

    It will try to identify where the data column starts, and use the preceding
    field as the name of the data colum.

    csv_file[column_number] = ['Name_str', numpy.ndarray]

    The build-in CSV class is less suitable for typical scientific use. This
    CSV file class allows easy access to rows and columns of the file with the
    getrow() and getcolumn() methods. The getmetainfo() will also provide some
    satistics of the files.

    """

    def __init__(self, path=None, debug=False, analyse=True, \
        buffer_size=10485760, field_delimiter=',', text_delimiter='"'):
        """
        Given the file path, prepare the csv file.

        Options:

        analyse     If true will try to scan the file to guess where the
                       numerical data starts. It can take a long time for large
                       files of unsual layouts.
        """
        self.file_path = path
        self.name = filePath(path).name
        self.analyse = analyse
        self.debug = debug
        # self.f_handler = None
        self.rows = None                    # Number of rows
        self.cols = None                    # Number of column
        self.idx = None                     # File seek index, linking the file
                                            #  byte position with each non-blank
                                            #  rows.
        self.metainfo = {}                # File header representations

        self.buffer_size = buffer_size      # Read file in this size chunks
        self.field_delimiter = field_delimiter
        self.text_delimiter = text_delimiter

        self.label_row = 0                  # Row number for column labels
        self.data_start = 0                 # Row number for where data starts

        # Check if the file is readable
        f = self.open()

        cols = 0
        f.seek(0)
        idx = [0]               # Byte position index of all non-blank rows

        #######################################################################
        # "for line in f" doesnt work because f.tell() will be stuck at certain
        # multiples of buffer length.
        #######################################################################
        #for line in f:
        #
        #    line = line.strip()
        #
        #    # Skip blank lines
        #    if ''.join(line.split(field_delimiter)) == '':
        #        continue
        #    else:
        #        idx.append(f.tell())
        #
        #    items = len(line.split(field_delimiter))
        #    if items > cols:
        #        cols = items
        #######################################################################

        while True:
            line = f.readline()         # Read the CVS file line by line.
                                        # Not used readlines because the file
                                        #     can potentially be very large

            if line == '':
                del idx[-1]             # Remove the last index entry
                break                   # Reached the end of the file

            line = line.strip()         # strip of '\n'
            lst = line.split(field_delimiter)

            #if ''.join(lst) == '':
            #    continue                # In case of a blank row
            #else:
            #    idx.append(f.tell())    # Write down the next byte position

            idx.append(f.tell())         # Don't ignore blank rows

            items = len(lst)
            if items > cols:
                cols = items

        self.rows = len(idx)
        self.cols = cols
        self.idx = idx

        f.close()

        if analyse == True:
            self.chk_data_start()
            # Try to read the header if exists (i.e. data isn't starting on the 1st row
            if self.data_start != 0:
                self.getmetainfo()

        return

    def flush(self):
        """
        Flush out all the metaInfo and guessed details
        """
        self.metainfo = {}                # File header representations

        self.label_row = 0                  # Row number for column labels
        self.data_start = 0                 # Row number for where data starts

        return


    def __getitem__(self, key):
        """
        Return the column of data in a tuple.
            content = self[key]
            content[0] = label
            content[1] = data
        """

        col = self.getcolumn(key)
        label = col[self.label_row].strip(self.text_delimiter)
        data = col[self.data_start:]

        ary = []
        try:                # Try to convert all data into numeric type.
            for i in range(len(data)):
                ary.append(float(data[i]))
            data = array(ary)
        except ValueError:
            del ary                 # Remain as text type if doesnt work

        return (label, data)

    def getcolumn(self, key):
        """
        Return the column of data as is
        """
        if not key < self.cols:
            raise IndexError("list index (" + str(key) + \
                ") out of range (" + str(self.cols - 1) + ").")

        f = self.open()

        idx = self.idx
        field_delimiter = self.field_delimiter
        text_delimiter = self.text_delimiter

        output = []

        for i in idx:
            f.seek(i)
            row = f.readline().strip().split(field_delimiter)
            try:
                output.append(row[key])
            except IndexError:
                output.append('')       # Not all rows has the same number of columns

        f.close()

        return output

    def __len__(self):
        return self.cols

    def open(self):
        """
        Open the file for read
        """
        # Check if the file is readable
        try:
            f = open(self.file_path, 'rU')
            return f
        except IOError:
            raise IOError("Unable to read file: " + self.file_path)


    def set_data_start(self, row):
        """
        Indicate numerical data to start from given row index (start with 0).
        """
        self.data_start = row
        return

    def chk_data_start(self):
        """
        Not all csv file contains only numerical data, many contain labels
        for each column as well as some header information.

        label = self[column][self.label_row]
        data = self[column][self.data_start:]
        """
        field_delimiter = self.field_delimiter
        i = 0
        while i < self.rows:
            row = self.getrow(i)

            try:
                for item in row:
                    float(item)
                break
            except ValueError:
                i += 1          # Not this row, some items are not numeric, try again


        if i == self.rows - 1:
            # Searched through all records, still no idea where it should start
            self.label_row = 0                  # Row number for column labels
            self.data_start = 0                 # Row number for where data starts
        else:
            self.label_row = i - 1
            self.data_start = i
        return

    def getrow(self, key):
        """
        counterpart of __getitem__(), which will return the column
        """
        if not key < self.rows:
            raise IndexError("list index (" + str(key) + \
                ") out of range (" + str(self.rows - 1) + ").")

        f = self.open()
        f.seek(self.idx[key])
        row = f.readline().strip().split(self.field_delimiter)
        f.close()
        return row


    def get_data_col(self, column):
        """
        Unlike __getitem__(), this will only return the data column
        """
        if not column < self.cols:
            raise IndexError("Requested column index (" + str(key) + \
                ") out of range (" + str(self.cols - 1) + ").")

        f = self.open()

        idx = self.idx
        field_delimiter = self.field_delimiter
        text_delimiter = self.text_delimiter

        output = []

        for i in idx[self.data_start:]:
            f.seek(i)
            row = f.readline().strip().split(field_delimiter)
            try:
                output.append(row[column])
            except IndexError:
                output.append('')       # Not all rows has the same number of columns

        f.close()

        return output


    def getmetainfo(self):
        """
        Some instruments generates csv file headers, try to put those in
        the dictionary

        """
        info_pair = []
        metainfo = {}

        for i in range(self.data_start - 1):
            lst = self.getrow(i)              # ['lbl0', 'val0', 'lbl1', 'val1' ...]
            # If the current row contains even number of entries
            # Assume the Odd index items are labels, and Even index items are values
            if len(lst)%2 == 0:
                # [('lbl0', 'val0'), ('lbl1', 'val1'), ...]
                info_pair += zip(lst[0::2], lst[1::2])


        info_pair.sort(key=itemgetter(0))
        for k, g in groupby(info_pair, key=itemgetter(0)):

            val = map(itemgetter(1), g)[0]
            if k == '':
                if ''.join(val) == '':
                    # Ohh dear, all blanks. Skip the entry.
                    continue
                # No idea what the key is, gives it a default value
                k = 'Unknown'


            metainfo[k] = val

        self.metainfo = metainfo

        return


    def __repr__(self):
        """
        Text representation of the object
        """

        desc = ''
        desc += 'CSV file object at: ' + self.file_path + linesep
        desc += '\tNumber of rows: ' + str(self.rows) + linesep
        desc += '\tNumber of columns: ' + str(self.cols) + linesep
        desc += '\tField delimiter is: ' + self.field_delimiter + linesep
        desc += '\tText delimiter is: ' + self.text_delimiter + linesep
        desc += '\tData is thought to start at row number: ' + str(self.data_start) + linesep

        if not self.metainfo == {}:
            desc += '\t' + repr(self.metainfo)

        return desc


    def update_metainfo(self, keys, func):
        """
        Provide a method to transform given metainfo with func
        """
        metainfo = self.metainfo
        for field in keys:
            if metainfo.has_key(field):
                metainfo[field] = func(metainfo[field])

        self.metainfo = metainfo
        return

def __dict_loop(dic, h5):
    """
    Recursively store items in a dictionary object into given h5py object as
    h5py.Dataset.

    If the dictionary item is itself an dictrionay object, create a h5py.Group,
    and store subsequent items within.
    """

    # List of keys to the items in the current dict is itself a dict
    dict_lst = []

    # First loop, store all the simple data, and build the dict_lst
    for key in dic:
        if isinstance(dic[key], dict):
            dict_lst.append(key)
        else:
            h5.create_dataset(key, data=dic[key])

    # Process the dic_lst
    for key in dict_lst:

        # Create a subgroup
        grp = h5.create_group(key)

        __dict_loop(dic[key], grp)

    return


def to_csv(metAry, path, debug=False, \
        field_delimiter=',', text_delimiter='"', linesep="\r\n"):
    """
    Write the metAry into given file path in the CSV format

    dest can be string or h5py.Group object.

    If dest is a string, it is interpret as the destination file path.

    If dest is a h5py.Group object, the content will be store under the given group.
    """

    path = filePath(path)

    FD = field_delimiter
    TD = text_delimiter
    LS = linesep

    if not path.write:
        print("Given file path is not writable." + path.full)
        raise IOError

    info = metAry.copy_info()
    data = metAry.data

    nfo_keys = info.keys()
    nfo_keys.sort()

    with open(path.full, 'wb') as f:

        # Write out the meta info first
        for key in nfo_keys:

            val = info.pop(key)

            if isinstance(val, (int, long, float, complex)):
                val = str(val)
            else:
                val = TD + str(val) + TD

            f.write(key + FD + val + LS)

        # Write out the content, format depends on the number of data dimensions
        if data.ndim == 1:
            # One dimentional data, write out the index - value pairs

            content = zip(metAry.get_axis(), data)

            for idx, val in content:
                f.write(str(idx) + FD + str(val) + LS)

        elif data.ndim == 2:
            # Two dimensional data, write out the x-y grid
            x, y = data.shape

            for x_idx in range(x):
                row = FD.join(map(str, data[x_idx]))
                f.write(row + LS)

        else:
            # N-dimensional data, flatten the array, the dump
            f.write(LS.join(map(str, data.flatten())))
            f.write(LS)

    return
