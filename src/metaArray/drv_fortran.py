# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 2024 10:03

@author: samhill

This file contains the fortran record object
"""

from os import sep, linesep
from struct import unpack, calcsize

from .misc import filePath


class binrecord:
    """"

    Binary record example
    | Record len (4byte Int) | ... Binary data ... | Record len (4byte Int) |


    Methods:

    read(N = None)
        Retune a number of N records in a tuple
        If N is not specified, return all available records

    next()
        Return the next record

    write(string)
        (over)write record to a file

    append(string)
        create/append record to a existing file

    seek(n)
        Go to the Nth record

    tell()
        Return the current record number (position)


    Instants:

    self.file_path

    self.unpack_str = '<L'          # big-endian (>) or little-endian (<)
                                    # unsign long
    self.header_len = 4             # Number of bytes for the record len
                                    # register

    self.record_pos = 0             # Current record position (i.e. at the
                                    # Nth record)

    self.record_num = 0             # update together # Number of records
    self.record_index = []          # update together #

    self.flg_debug = False
    """
    def __init__(self, path: str,
                 debug: bool = False) -> None:

        self.file_path = filePath(path)

        self.unpack_str = ''
        self.header_len = 0
        self.endian = '<'

        self.record_pos = 0
        self.record_num = 0

        self.record_index = []

        self.flg_debug = debug

        # Try to guess parameters like byte Order and header length.
        if self._parameterChk_():
            self._index_()      # Index the file
        else:
            # No idea how to unpack the records
            msg = "Can not guess how to unpack the records. " + \
                "Try to specify the following parameters manually: " + sep + \
                "'.header_len'" + sep + \
                "'.unpack_str'" + sep + \
                "'.endian'" + sep + \
                "then index the file manually using '._index_()'"
            print(msg)

        if debug:
            print('file_path: ', self.file_path.full)

            print('unpack_str: ', self.unpack_str)
            print('header_len: ', self.header_len)

            print('record_pos: ', self.record_pos)
            print('record_num: ', self.record_num)

    def __call__(self) -> None:
        return self.showIndex()

    def __len__(self) -> int:
        return len(self.record_index)

    def __getitem__(self, key: int) -> bytes:
        key = self._chk_index(key)
        self.seek(key)
        return self.next()

    def __setitem__(self, key, value) -> None:
        raise NotImplementedError

    def _chk_index(self, n: int) -> int:
        """
        Check the given index value, and do the ring buffer trick
        """
        if n < 0:
            n += self.record_num

        if n >= self.record_num:
            raise IndexError(f"Requested record number: {n} outside range. Only {self.record_num} record(s) exist")  # noqa: E501

        return n

    def _index_(self) -> None:
        """
        This is to index the file

        Once indexed, file should remain open as binary read
        """

        record_index = []  # Index place holder

        header_len = self.header_len
        unpack_str = self.unpack_str

        # The index operation should stop and return what it found if
        # any exception is raised on the way

        with open(self.file_path.full, 'rb') as f:

            while True:
                file_pos = f.tell()
                head = f.read(header_len)

                if head == b'':
                    # Humm, hit the end of the file, should stop now
                    break

                record_len = unpack(unpack_str, head)[0]

                f.seek(file_pos + header_len + record_len)
                tail = f.read(header_len)

                if head != tail:
                    print("Error, inconsistent record length descriptors")
                    print("\tInitial file position: ", file_pos)
                    print("\tHead desc: ", unpack(unpack_str, head)[0])
                    print("\tTail desc: ", unpack(unpack_str, tail)[0])
                    break
                else:
                    # The index of raw record string will have:
                    # | record begin | record end | raw record len |
                    rawRecLen = record_len + header_len + header_len
                    record_index.append([file_pos, file_pos + rawRecLen,
                                         rawRecLen])

        self.record_index = record_index
        self.record_num = len(self.record_index)

    def _parameterChk_(self) -> bool:
        """
        Try to guess the following parameters:

            self.header_len
            self.unpack_str
            self.endian
        """
        # | Record len (Int) | ... Binary data ... | Record len (Int) |

        # unpack string of the big and small endian types
        endian_lst = ('<', '>')

        # A tuple of int types to check out. These are type codes in the
        # struct library.
        type_lst = ('I', 'L', 'H', 'Q')

        with open(self.file_path.full, 'rb') as f:

            # Read the first 64 bytes of the file into buffer, should
            # be enough to cover upto 512bit Int
            buff = f.read(64)
            f.seek(0)

            # Brute force guessing all the different combinations
            # for the first record
            for endian in endian_lst:
                for dtype in type_lst:
                    header_len = calcsize(dtype)

                    # Assume the records start at the beginning of the file
                    head = buff[:header_len]

                    # Header length according to current int type.
                    record_len = unpack(endian + dtype, head)[0]

                    # unpack the record tail
                    f.seek(header_len + record_len)
                    tail = f.read(header_len)

                    if head == tail:
                        # Looks like it worked for the first record
                        # Hopefully this is not a fluke
                        self.header_len = header_len
                        self.unpack_str = endian + dtype
                        self.endian = endian

                        return True

            # Ran out of things to try
            return False

    def tell(self) -> int:
        return self.record_pos

    def seek(self, n: int) -> None:
        """
        Go to the Nth record
        Similar to f.seek()
        """
        n = self._chk_index(n)
        self.record_pos = n

    def showIndex(self, N: int = None) -> None:
        """
        Print out the index statistics of the Nth record
        if N not specified, print them all out
        """

        # If index doesnt exist yet, create it.
        if self.record_index == []:
            msg = 'There are no records currently indexed in this file.'
            print(msg)

        headers = self.header_len * 2
        record_index = self.record_index

        if N is None:
            msg = ''
            for i in range(len(record_index)):
                msg += f'Record number: {i}\t'
                msg += f'Length of: {record_index[i][2] - headers} bytes\t'
                msg += f'Starting: {record_index[i][0]}th byte\t'
                msg += f'Ending: {record_index[i][1]}th byte\t'
                msg += linesep
        else:
            N = self._chk_index(N)      # Sanity check
            msg = f'Record number: {N}\t'
            msg += f'Length of: {record_index[N][2] - headers} bytes\t'
            msg += f'Starting: {record_index[N][0]}th byte\t'
            msg += f'Ending: {record_index[N][1]}th byte\t'

        print(msg)

    def item_len(self, key: int) -> int:
        """
        Return the length of an record item, without reading the entire record
        """
        key = self._chk_index(key)

        return self.record_index[key][2] - self.header_len * 2

    def next(self) -> bytes:
        """
        Return the current record, and advance to the next record.

        Return a empty bytearray if file end is reached
        """
        return self.read(1)[0]

    def read(self, N: int = None) -> list[bytes]:
        """
        Retune a number of N records in a tuple
        If N is not specified, return all remaining records
        """
        f = open(self.file_path.full, 'rb')
        with open(self.file_path.full, 'rb') as f:
            record_pos = self.record_pos

            if N is None:
                # Read the rest out
                end_pos = self.record_num

            elif (N + record_pos) > self.record_num:
                raise IndexError("Requested " + str(N) +
                                 " records from record index: " +
                                 str(record_pos) +
                                 ", but only " +
                                 str(self.record_num) +
                                 " record(s) exist")
            else:
                end_pos = record_pos + N

            header_len = self.header_len
            record_index = self.record_index
            output = []

            for pos in range(record_pos, end_pos):

                rcd_begin, rcd_end, rcd_len = record_index[pos]

                # Where the raw record starts
                f.seek(rcd_begin)

                # Read the raw record
                raw_record = f.read(rcd_len)

                output.append(raw_record[header_len:-header_len])

        return output

    def __repr__(self) -> str:
        """
        Text representation of the object
        """

        desc = "This is a Fortran binary record object."
        desc += f"\tIt has {len(self)} record(s):" + linesep
        desc += "Use the .showIndex() method to obtain a detailed list of records"  # noqa: E501

        return desc

    def write(self, file_path) -> None:
        raise NotImplementedError

    def append(self, file_path) -> None:
        raise NotImplementedError


class NoOverlapError(IndexError):
    """
    NoOverlapError is raised when testing for overlaping regions in xyz space,
    but none found.
    """
    pass
