# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 2024 10:10

@author: samhill

File for controlling csv file reading
"""
import numpy.typing as npt
import typing

from os import linesep
from itertools import groupby
from operator import itemgetter
from numpy import array

from .misc import filePath
from .core import metaArray


class CSVFile:
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

    def __init__(self, path: str = None, debug: bool = False,
                 analyse: bool = True, buffer_size: int = 10485760,
                 field_delimiter: str = ',',
                 text_delimiter: str = '"') -> None:
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
        self.rows = None            # Number of rows
        self.cols = None            # Number of column
        self.idx = None             # File seek index, linking the file

        self.metainfo = {}          # File header representations

        self.buffer_size = buffer_size      # Read file in this size chunks
        self.field_delimiter = field_delimiter
        self.text_delimiter = text_delimiter

        self.label_row = 0          # Row number for column labels
        self.data_start = 0         # Row number for where data starts

        # Check if the file is readable
        with open(self.file_path, 'r') as f:
            cols = 0
            f.seek(0)
            idx = [0]               # Byte position index of all non-blank rows

            while True:
                # Read line by line as file could be very large
                line = f.readline()

                if line == '':
                    del idx[-1]             # Remove the last index entry
                    break                   # Reached the end of the file

                line = line.strip()         # strip of '\n'
                lst = line.split(field_delimiter)

                idx.append(f.tell())         # Don't ignore blank rows

                items = len(lst)
                if items > cols:
                    cols = items

            self.rows = len(idx)
            self.cols = cols
            self.idx = idx

        if analyse:
            self.chk_data_start()
            # Try to read the header if exists (i.e. data isn't
            # starting on the 1st row
            if self.data_start != 0:
                self.getmetainfo()

    def __repr__(self) -> str:
        """
        Text representation of the object
        """

        desc = ''
        desc += f'CSV file object at: {self.file_path}{linesep}'
        desc += f'\tNumber of rows: {self.rows}{linesep}'
        desc += f'\tNumber of columns: {self.cols}{linesep}'
        desc += f'\tField delimiter is: {self.field_delimiter}{linesep}'
        desc += f'\tText delimiter is: {self.text_delimiter}{linesep}'
        desc += f'\tData is thought to start at row number: {self.data_start}'
        desc += linesep

        if self.metainfo:
            desc += '\t' + repr(self.metainfo)

        return desc

    def __getitem__(self, key: int) -> tuple[str, npt.NDArray]:
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
        try:        # Try to convert all data into numeric type.
            for dat in data:
                ary.append(float(dat))
            data = array(ary)
        except ValueError:
            del ary                 # Remain as text type if doesnt work

        return (label, data)

    def __len__(self) -> int:
        """
        Returns the number of columns in csv file
        """
        return self.cols

    def flush(self) -> None:
        """
        Flush out all the metaInfo and guessed details
        """
        self.metainfo = {}                # File header representations

        self.label_row = 0                # Row number for column labels
        self.data_start = 0               # Row number for where data starts

    def getcolumn(self, key: int) -> list[str]:
        """
        Return the column of data as is
        """
        if key > self.cols:
            raise IndexError(f"list index ({key}) out of range ({self.cols-1})")

        with open(self.file_path, 'r') as f:
            idx = self.idx
            field_delimiter = self.field_delimiter

            output = []

            for i in idx:
                f.seek(i)
                row = f.readline().strip().split(field_delimiter)
                try:
                    output.append(row[key])
                except IndexError:
                    # Not all rows has the same number of columns
                    output.append('')

        return output

    def set_data_start(self, row: int) -> None:
        """
        Indicate numerical data to start from given row index (start with 0).
        """
        self.data_start = row

    def chk_data_start(self) -> None:
        """
        Not all csv file contains only numerical data, many contain labels
        for each column as well as some header information.

        label = self[column][self.label_row]
        data = self[column][self.data_start:]
        """
        i = 0
        while i < self.rows:
            row = self.getrow(i)

            try:
                for item in row:
                    float(item)
                break
            except ValueError:
                # Not this row, some items are not numeric, try again
                i += 1

        if i == self.rows - 1:
            # Searched through all records, still no idea where it should start
            self.label_row = 0
            self.data_start = 0
        else:
            self.label_row = i - 1
            self.data_start = i

    def getrow(self, key: int) -> list[str]:
        """
        counterpart of __getitem__(), which will return the column
        """
        if key > self.rows:
            raise IndexError(f"list index ({key}) out of range ({self.ros-1})")

        with open(self.file_path, 'r') as f:
            f.seek(self.idx[key])
            row = f.readline().strip().split(self.field_delimiter)

        return row

    def get_data_col(self, column: int) -> list[str]:
        """
        Unlike __getitem__(), this will only return the data column
        """
        if column > self.cols:
            max_c = self.cols-1
            msg = f"Requested column index ({column}) out of range ({max_c})"
            raise IndexError(msg)

        with open(self.file_path, 'r') as f:
            idx = self.idx
            field_delimiter = self.field_delimiter

            output = []

            for i in idx[self.data_start:]:
                f.seek(i)
                row = f.readline().strip().split(field_delimiter)
                try:
                    output.append(row[column])
                except IndexError:
                    # Not all rows has the same number of columns
                    output.append('')

        return output

    def get_meta_col(self, column: int) -> dict:
        """
        Gets meta info from a particular data column.
        """
        metainfo = {}
        for key, val in self.metainfo.items():
            metainfo[key] = val[column-1]

        self.metainfo = metainfo
        return metainfo

    def getmetainfo(self) -> None:
        """
        Some instruments generates csv file headers, try to put those in
        the dictionary
        """
        info_pair = []
        metainfo = {}

        for i in range(self.data_start - 1):
            # ['lbl0', 'val0', 'lbl1', 'val1' ...]
            lst = self.getrow(i)
            lst_len = len(lst)
            # New layout of metainfo labl, val0, val1, val2...
            # Disguard empty rows
            if lst_len >= 2:
                # [('lbl0', 'val0'), ('lbl1', 'val1'), ...]
                info_pair += zip((lst_len-1)*[lst[0]], lst[1:])

        info_pair.sort(key=itemgetter(0))
        for k, g in groupby(info_pair, key=itemgetter(0)):

            val = list(map(itemgetter(1), g))
            if k == '':
                if ''.join(val) == '':
                    # Ohh dear, all blanks. Skip the entry.
                    continue
                # No idea what the key is, gives it a default value
                k = 'Unknown'

            # Annoyingly, new format doesn't repeat horizontal metadata
            # or things that apply to all channels
            horizontal = ['Horizontal Scale', 'Sample Interval',
                          'Record Length', 'Filter Frequency',
                          'Horizontal Units', 'Firmware Version', 'Model']
            # Repeat them manually
            if k in horizontal:
                val = (self.cols - 1) * [val[0]]
            metainfo[k] = val

        self.metainfo = metainfo

    def update_metainfo(self, keys: typing.Hashable,
                        func: typing.Callable) -> None:
        """
        Provide a method to transform given metainfo with func
        """
        metainfo = self.metainfo
        for field in keys:
            if field in metainfo:
                try:
                    metainfo[field] = func(metainfo[field])
                except (ValueError, TypeError):
                    metainfo[field] = 'Unknown'

        self.metainfo = metainfo
        return


def open_csv(path: str, mode: str, debug: bool = False,
             field_delimiter: str = ',', text_delimiter='"') -> CSVFile:
    """
    CSV file parallel of the build in open function
    Inputs:
        path            Path of the CSV file
        mode            {'r'|'w'}
        linesep         new line char

    In Windows platforms, a newline is typically encoded as \r\n
    In Linux/Unix/OS X, a newline is typically encoded as \n
    In RISC OS spooled text output, a newline is typically encoded as \n\r

    Any well writen program should be able to understand any of the
    above conventions

    Output:
        The csv file object
    """
    if mode == 'r':
        return CSVFile(path, debug=debug,
                       field_delimiter=field_delimiter,
                       text_delimiter=text_delimiter)
    elif mode == 'w':
        raise NotImplementedError
    else:
        raise NotImplementedError


def to_csv(metAry: metaArray, path: str, debug: bool = False,
           field_delimiter: str = ',', text_delimiter: str = '"',
           linesep="\r\n") -> None:
    """
    Write the metAry into given file path in the CSV format
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

            if isinstance(val, (int, float, complex)):
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
            x, _ = data.shape

            for x_idx in range(x):
                row = FD.join(map(str, data[x_idx]))
                f.write(row + LS)

        else:
            # N-dimensional data, flatten the array, the dump
            f.write(LS.join(map(str, data.flatten())))
            f.write(LS)
