#       drv_fortran.py
#       
#       Copyright 2009 charley <charley@hosts-137-205-164-145.phys.warwick.ac.uk>
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
#
# This file contains the fortran record object
#

from os import sep, linesep
from os import access, R_OK
from struct import unpack, calcsize

from misc import filePath


class binrecord(object):
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
    
    self.unpack_str = '<L'          # big-endian (>) or little-endian (<) unsign long
    self.header_len = 4             # Number of bytes for the record len register
    
    self.record_pos = 0             # Current record position (i.e. at the Nth record)
    
    self.record_num = 0             # update together # Number of records
    self.record_index = []          # update together #
    
    self.flg_debug = False
    
    """
    
    def __init__(self, path, debug = False):
        
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
                    "Please try to specify the following parameters manually: " + sep + \
                    "'.header_len'" + sep + \
                    "'.unpack_str'"  + sep + \
                    "'.endian'"  + sep + \
                    "then index the file manually using '._index_()'"
            print(msg)
        
        if debug:
            print'file_path: ',  self.file_path.full
            
            print'unpack_str: ', self.unpack_str
            print'header_len: ', self.header_len
            
            print'record_pos: ', self.record_pos
            print'record_num: ', self.record_num
        
        return
    
    def __call__(self):
        return self.showIndex()
    
    def __len__(self):
        return len(self.record_index)
    
    def __getitem__(self, key):
        key = self._chk_index(key)
        self.seek(key)
        return self.next()
    
    def __setitem__(self, key, value):
        raise NotImplementedError
        return
    
    def _chk_index(self, n):
        """
        Check the given index value, and do the ring buffer trick
        """
        if n < 0:
            n += self.record_num
        
        if n >= self.record_num:
            raise IndexError("Requested record number: " + str(n) + " outside range. Only " + str(self.record_num) + " record(s) exist")
        
        return n
    
    
    def _index_(self):
        """
        This is to index the file
        
        Once indexed, file should remain open as binary read
        """
        
        record_index = [] # Index place holder
        
        header_len = self.header_len
        unpack_str = self.unpack_str
        
        
        # The index operation should stop and return what it found if 
        # any exception is raised on the way
        
        # f = open(self.file_path.full)
        with open(self.file_path.full, 'rb') as f:
            
            while True:
                file_pos = f.tell()
                head = f.read(header_len)
                
                if head == '':
                    break # Humm, hit the end of the file, should stop now
                
                record_len = unpack(unpack_str, head)[0]
                
                f.seek(file_pos + header_len + record_len)
                tail = f.read(header_len)
                
                if head != tail:
                    print "Error, inconsistent record length descriptors"
                    # print "\tFile position: ", f.tell() - unpack(unpack_str, head)[0] - header_len
                    print "\tInitial file position: ", file_pos
                    print "\tHead desc: ", unpack(unpack_str, head)[0]
                    print "\tTail desc: ", unpack(unpack_str, tail)[0]
                    break
                else:
                    # The index of raw record string will have:
                    # | record begin | record end | raw record len |
                    rawRecLen = record_len + header_len + header_len
                    record_index.append([file_pos, file_pos + rawRecLen, rawRecLen])
            
            # f.close()
            
        self.record_index = record_index
        self.record_num = len(self.record_index)
        
        return None
    
    
    
    def _parameterChk_(self):
        """
        Try to guess the following parameters:
        
            self.header_len
            self.unpack_str
            self.endian
        """
        # | Record len (Int) | ... Binary data ... | Record len (Int) |
        
        
        # unpack string of the big and small endian types
        endian_lst = ['<', '>']
        
        # A list of int types to check out. These are type codes in the
        # struct library.
        type_lst = ['I', 'L', 'H', 'Q']
                
        # f = open(self.file_path.full, 'rb')
        with open(self.file_path.full, 'rb') as f:
            
            # Read the first 32 bytes of the file into buffer, should 
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
                        
                        # f.close()
                        return True
            
            # Ran out of things to try
            # f.close()
            return False
    
    
    def tell(self):
        return self.record_pos
    
    def seek(self, n):
        """
        Go to the Nth record
        Similar to f.seek()
        """
        n = self._chk_index(n)
        self.record_pos = n
        return
    
    def showIndex(self, N = None):
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
        
        if N == None:
            msg = ''
            for i in range(len(record_index)):
                msg += 'Record number: ' + str(i) + '\t'
                msg += 'Length of: ' + str(record_index[i][2] - headers) + ' bytes \t'
                msg += 'Starting: ' + str(record_index[i][0]) + 'th byte\t'
                msg += 'Ending: ' + str(record_index[i][1]) + 'th byte\t'
                msg += linesep
        else:
            N = self._chk_index(N)      # Sanity check
            msg = 'Record number: ' + str(N) + '\t'
            msg += 'Length of: ' + str(record_index[N][2] - headers) + ' bytes \t'
            msg += 'Starting: ' + str(record_index[N][0]) + 'th byte\t'
            msg += 'Ending: ' + str(record_index[N][1]) + 'th byte\t'
        
        print msg
        return
    
    
    def item_len(self, key):
        """
        Return the length of an record item, without reading the entire record
        """
        key = self._chk_index(key)
        
        return self.record_index[key][2] - self.header_len * 2
    
    
    def next(self):
        """
        Return the current record, and advance to the next record.
        
        Return a empty string if file end is reached
        """
        
        return self.read(1)[0]
    
    
    
    def read(self, N = None):
        """
        Retune a number of N records in a tuple
        If N is not specified, return all remaining records
        """
        f = open(self.file_path.full, 'rb')
        
        record_pos = self.record_pos
        
        if N == None:
            end_pos = self.record_num                       # Read the rest out
            
        elif (N + record_pos) > self.record_num:
            raise IndexError("Requested " + str(N) + \
                            " records from record index: " + str(record_pos) + \
                            ", but only " + str(self.record_num) + " record(s) exist")
        else:
            end_pos = record_pos + N
        
        
        header_len = self.header_len
        record_index = self.record_index
        output = []
        
        for pos in range(record_pos, end_pos):
            
            rcd_begin, rcd_end, rcd_len = record_index[pos]
            
            f.seek(rcd_begin)                     # Where the raw record starts
            raw_record = f.read(rcd_len)          # Read the raw record
            
            output.append(raw_record[header_len:-header_len])
        
        f.close()
        return output
    
    
    def __repr__(self):
        """
        Text representation of the object
        """
        
        idx = self.record_index
        
        desc = "This is a Fortran binary record object."
        desc += "\tIt has " + str(len(self.record_index)) + " record(s):" + linesep
        desc += "Use the .showIndex() method to obtain a detailed list of records"        
        
        #for i in range(len(idx)):
            #index_str = "\t" + str(i) + " : "
            #hdr_pos, hdr_len, data_pos, data_len, hdr_dict, unpack_str = idx[i]
            
            #desc += index_str + "File location: " + str(hdr_pos) + linesep
            #desc += index_str + "Fortran header length: " + str(self.header_len) + linesep
            #desc += index_str + "Unpack string: " + self.unpack_str + linesep
            
            #for key, val in hdr_dict.iteritems():
                #desc += index_str + '[' + key + '] ' + str(val) + linesep
            
            #desc += linesep
            #desc += "Use the .showIndex() method to obtain a detailed list of records"
            #desc += linesep
        
        
        return desc
    
    
    def write(self, file_path):
        raise NotImplementedError
        return 
    
    def append(self, file_path):
        raise NotImplementedError
        return


class NoOverlapError(IndexError):
    """
    NoOverlapError is raised when testing for overlaping regions in xyz space,
    but none found.
    """
    pass
