# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 2024 19:53

@author: samhill

Demo Programme that should the basics of using metaArray project
"""
from os.path import join
from os import linesep

from textwrap import TextWrapper
from io import StringIO

from metaArray.misc import filePath

# Enviromental variables
demo_dir = join(filePath(__file__).baseDir, 'example')
tty_width = 72
partition = '-' * tty_width
prompt = '>>> '

# Configure text-wrapper
wrapper = TextWrapper()
wrapper.width = tty_width
wrapper.replace_whitespace = False
# wrapper.drop_whitespace = False
wrapper.initial_indent = "- "
wrapper.subsequent_indent = '- '

# Define comment text wrapper
comment_wrapper = TextWrapper()
comment_wrapper.replace_whitespace = False
# comment_wrapper.drop_whitespace = False
comment_wrapper.width = tty_width
comment_wrapper.subsequent_indent = '# '


class DemoItem:
    """
    Menu item

    Contains pointer to menu item.
    """
    def __init__(self, title: str = '', parent=None, exe=None) -> None:
        '''
        Set up titles, etc
        '''
        self.title = title
        self.parent = parent
        self.info = ''
        self.exe = exe

    def __call__(self) -> None:
        summary = self.exe()

        if summary is not None:
            print(linesep + ' demo summary '.center(tty_width, '-') + linesep)
            print(summary)
        else:
            print(linesep + linesep)

        print(' END of demo '.center(tty_width, '-') + linesep)

        if self.parent is not None:
            print(linesep + linesep)
            return self.parent()


class DemoMenu:
    """
    Menu object

    Contains a list of menu options, as well as the parent menu if exist.
    """

    def __init__(self, title: str = '', parent=None) -> None:
        '''
        Set up Titles, etcs
        '''
        self.title = title
        self.parent = parent
        self.info = ''
        self.items = {}

    def add_item(self, obj: DemoItem) -> None:
        '''
        Add item
        '''
        obj.parent = self
        self.items[obj.title] = obj

    def __call__(self):
        print(partition)
        if self.parent is not None:
            title = self.parent.title + ' - ' + self.title
        else:
            title = self.title

        print(title.center(tty_width))
        print(partition)
        print(wrapper.fill(self.info))
        print(partition)

        lst = list(self.items.keys())
        lst.sort()

        print('\tOption\tDescription')
        for i in range(len(lst)):
            print('\t' + str(i).rjust(6) + '\t' + self.items[lst[i]].title)

        # Present the return option if parent menu exists
        i += 1
        return_option = i
        if self.parent is not None:
            print('\t' + str(i).rjust(6) + '\tReturn to: ' + self.parent.title)
            i += 1

        print('\t' + str(i).rjust(6) + '\tQuit')
        quit_option = i

        print(partition)

        while True:
            option = input("Which option would you like to select? ")
            try:
                option = int(option)
            except ValueError:
                continue

            if option == quit_option:
                return

            if option == return_option:
                print(partition + linesep + linesep)
                return self.parent()

            if option >= 0 and option < len(lst):
                item = self.items[lst[option]]

                if isinstance(item, DemoItem):
                    print(linesep + linesep + partition)
                    print(wrapper.fill(item.info))

                print(partition + linesep + linesep)
                return item()

            continue


def prcs_demo(code: str, summary: str = '') -> str:
    # Fill the code block into IO buffer
    with StringIO() as str_buff:
        str_buff.write(code.strip())

        str_buff.seek(0)
        for st in str_buff:
            st = st.strip()
            summary += st + linesep

            if st == '':
                print(linesep)
            elif st[0] == '#':
                print(comment_wrapper.fill(st))
            else:
                print(prompt + st)
                exec(st)

    return summary


# Begin example codes
########################
main_menu = DemoMenu(title='metaArray demos')
main_menu.info = 'This is a list of demos to illustrate the usage of metaArray.'

###############################################################################
# I/O demos
########################
drv_menu = DemoMenu(title='File I/O demos')
drv_menu.info = 'This is a list of demos to illustrate the usage file I/O with metaArray.'  # noqa: E501


def isf_demo() -> str:
    """
    Example on Tek isf file reader
    """
    code = """
    from metaArray.drv_Tek import isf
    f = isf('""" + join(demo_dir, 'DPO4000B.isf') + """', debug = True)

    # Have a look at the file content
    #*********************************
    print(f)

    # Load it into metaArray
    #************************
    ary = f[0]
    # See what the metaArray looks like
    print(ary)

    # Here is a plot of the contents
    #********************************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    fig, ax = plot1d(ary, legend=-1)
    fig.savefig('demo_isf.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_isf = DemoItem(title='Read Tektronix isf file.', exe=isf_demo)
demo_isf.info = 'This demo will illustrate the usage of Tek .isf interpreter'
drv_menu.add_item(demo_isf)


def DPO2000_csv_demo() -> str:
    """
    Example on DPO2000 csv file reader
    """
    code = """
    from metaArray.drv_Tek import DPO2000_csv
    f = DPO2000_csv('""" + join(demo_dir, 'DPO2000.csv') + """', debug = True)

    # Have a look at the file content
    #*********************************
    print(f)

    # Load it into metaArray
    #************************
    ary = f()
    # See what the metaArray looks like
    print(ary)

    # Here is a plot of the contents
    #********************************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    fig, ax = plot1d(ary, legend=-1)
    fig.savefig('demo_DPO2000_csv.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_DPO2000_csv = DemoItem(title='Read Tektronix DPO2000 csv file.',
                            exe=DPO2000_csv_demo)
demo_DPO2000_csv.info = 'This demo will illustrate the usage of Tek DPO2000 series csv file interpreter'  # noqa: E501
drv_menu.add_item(demo_DPO2000_csv)


def TDS2000_csv_demo() -> str:
    """
    Example on TDS2000 csv file reader
    """
    code = """
    from metaArray.drv_Tek import TDS2000_csv
    f = TDS2000_csv('""" + join(demo_dir, 'TDS2000.csv') + """', debug = True)

    # Have a look at the file content
    #*********************************
    print(f)

    # Load it into metaArray
    #************************
    ary = f()

    # See what the metaArray looks like
    #***********************************
    print(ary)

    # Here is a plot of the contents
    #********************************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    fig, ax = plot1d(ary, legend=-1)
    fig.savefig('demo_TDS2000_csv.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_TDS2000_csv = DemoItem(title='Read Tektronix TDS2000 csv file.',
                            exe=TDS2000_csv_demo)
demo_TDS2000_csv.info = 'This demo will illustrate the usage of Tek TDS2000 series csv file interpreter'  # noqa: E501
drv_menu.add_item(demo_TDS2000_csv)


def pout_hist_demo() -> str:
    """
    Example on PZFlex POUT file reader
    """
    code = """
    from metaArray.drv_flex import pout_hist
    f = pout_hist('""" + join(demo_dir, 'june10j.flxhst') + """')

    # Have a look at the file content
    #*********************************
    print(f)

    # Load it into metaArray
    #************************
    ary = f[2]

    ary['name'] = ary['june10j.desc']
    ary.set_range(0, 'unit', 's')

    # See what the metaArray looks like
    #***********************************
    print(ary)

    # Here is a plot of the contents
    #********************************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    fig, ax = plot1d(ary, legend=-1)
    fig.savefig('demo_pout_hist.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_pout_hist = DemoItem(title='Read PZFlex flxhst file.', exe=pout_hist_demo)
demo_pout_hist.info = 'This demo will illustrate the usage of PZFlex flxhst \
 file interpreter. flxhst files are generated by invoking the POUT HIST \
 command in the PZFlex input file.'
drv_menu.add_item(demo_pout_hist)


def data_out1_demo() -> str:
    """
    Example on PZFlex data out1 file reader
    """
    code = """
    from metaArray.drv_flex import data_out1
    f = data_out1('""" + join(demo_dir, '3D_billet_10mm_40mm.flxdato') + """')

    # Have a look at the file content
    #*********************************
    print(f)

    # Load it into metaArray
    #************************
    from metaArray import metaArray
    ary = f[27][:,:,0]
    ary = metaArray(ary)

    ary['name'] = '3D_billet_10mm_40mm'
    ary['unit'] =  ''
    ary['label'] = 'x-velocity'
    ary.set_range(0, 'begin', 0)
    ary.set_range(0, 'end', 1.651)
    ary.set_range(0, 'unit', 'm')
    ary.set_range(0, 'label', 'x - Horizontal')
    ary.set_range(1, 'begin', 0)
    ary.set_range(1, 'end', 0.226)
    ary.set_range(1, 'unit', 'm')
    ary.set_range(1, 'label', 'y - Vertical')

    # See what the metaArray looks like
    #***********************************
    print(ary)

    # Here is a plot of the contents
    #********************************
    from metaArray.drv_pylab import plot2d
    from matplotlib.pyplot import show, close

    fig, ax = plot2d(ary, size = (10, 5), aspect_ratio = 'xy',
                     corient='horizontal')
    # fig.savefig('plot2d.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_data_out1 = DemoItem(title='Read PZFlex flxdato file.', exe=data_out1_demo)
demo_data_out1.info = 'This demo will illustrate the usage of PZFlex flxdato \
 file interpreter. flxdato files are generated by invoking the DATA OUT1 \
 command in the PZFlex input file.'
drv_menu.add_item(demo_data_out1)


def HDF5_demo() -> str:
    """
    Example on HDF5 file reader/writer
    """
    code = """
    from metaArray.drv_h5py import to_h5, from_h5

    # Load an example metaArray for demontration
    #*********************************
    from metaArray.drv_Tek import isf
    ary = isf('""" + join(demo_dir, 'DPO2000.isf') + """')[1]


    # Save it into a HDF5 file
    #*********************************
    to_h5(ary, 'demo.h5')
    print(ary)

    # Load it into a metaArray
    #*********************************
    bry = from_h5('demo.h5')
    print(bry)
    """

    return prcs_demo(code)


demo_HDF5 = DemoItem(title='Read/write metaArray into HDF5 file.',
                     exe=HDF5_demo)
demo_HDF5.info = 'This demo will illustrate the usage of saving/loading \
 metaArray into/from HDF5 file.'
drv_menu.add_item(demo_HDF5)

main_menu.add_item(drv_menu)

###############################################################################
#                        END I/O demo                                         #
###############################################################################
###############################################################################


###############################################################################
# Plotting demos
###############################################################################
plot_menu = DemoMenu(title='Plotting and visualisation demos')
plot_menu.info = 'This is a list of demos to illustrate the usage metaArray \
aware plotting and visualisation funtions. The plotting routines are based on \
matplotlib.'


def multi_1d_demo() -> str:
    """
    Example on multiple 1D plot usage
    """
    code = """
    # Load a selection of data files as example
    #*******************************************
    from metaArray.drv_Tek import isf
    ary1 = isf('""" + join(demo_dir, 'multi_1.isf') + """')()
    ary2 = isf('""" + join(demo_dir, 'multi_2.isf') + """')()
    ary3 = isf('""" + join(demo_dir, 'multi_3.isf') + """')()

    # Correct for the DC offsets
    #****************************
    ary1.data -= ary1[72e-6:100e-6].data.mean()
    ary2.data -= ary2[72e-6:100e-6].data.mean()
    ary3.data -= ary3[72e-6:100e-6].data.mean()

    # Only need to set the last label and range
    #*******************************************
    ary3['label'] = 'Voltage'
    ary3.set_range(0, 'label', 'Time')

    # Here is how to plot
    #*********************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close
    fig, ax = plot1d(ary1[72e-6:100e-6],
                     size = (20, 15), label = 'First signal')
    fig, ax = plot1d(ary2[72e-6:100e-6], size = (20, 15),
                     label = 'Second signal', fig=fig, ax=ax)
    fig, ax = plot1d(ary3[72e-6:100e-6], size = (20, 15),
                     label = 'Third signal', fig=fig, ax=ax)
    ax.legend(loc=0)
    ax.set_title('Comparison of Generated signal on bent coil', fontsize=20)
    fig.savefig('demo_multi_1d.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_multi_1d = DemoItem(title='Plotting multiple 1D (A-scan) data.',
                         exe=multi_1d_demo)
demo_multi_1d.info = 'This demo will illustrate the usage of the plot1d \
interface to put multiple 1D (A-scan) metaArray on the same plot.'
plot_menu.add_item(demo_multi_1d)


def plot1d_demo() -> str:
    """
    Example on matplotlib 1D plot interface
    """
    code = """
    # Load some data as example
    #***************************
    from metaArray.drv_Tek import isf
    ary = isf('""" + join(demo_dir, 'DPO2000.isf') + """')()

    # Here is how to plot
    #*********************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    fig, ax = plot1d(ary, legend=-1)
    fig.savefig('demo_plot1d.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_plot1d = DemoItem(title='Plotting of 1D (A-scan) data.', exe=plot1d_demo)
demo_plot1d.info = 'This demo will illustrate the usage of the plot1d function \
for metaArray.'
plot_menu.add_item(demo_plot1d)


def plot2d_demo() -> str:
    """
    Example on matplotlib 2D plot interface
    """
    code = """
    # Load some data as example
    #***************************
    from cPickle import load
    f = open('""" + join(demo_dir, 'rel_amplitude.pickle') + """', 'rb')
    a = load(f).transpose()
    f.close()

    # Construct metaArray from numpy ndarray
    #****************************************
    from metaArray import metaArray
    ary = metaArray(a)

    ary['name'] = 'Relative amplitude'
    ary['unit'] =  ''      # '' -> unitless; None -> undefined
    ary['label'] = 'Amplitude ratio'

    # Per axis definitions
    ary.set_range(0, 'begin', -5e-3)
    ary.set_range(0, 'end', 5e-3)
    ary.set_range(0, 'unit', 'm')
    ary.set_range(0, 'label', 'x - Horizontal')
    ary.set_range(1, 'begin', 5e-3)
    ary.set_range(1, 'end', -5e-3)
    ary.set_range(1, 'unit', 'm')
    ary.set_range(1, 'label', 'y - Vertical')

    # Now its ready for plotting
    #****************************
    from metaArray.drv_pylab import plot2d
    from matplotlib.pyplot import show, close
    fig, ax = plot2d(ary)
    fig.savefig('demo_plot2d', dpi=400, format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_plot2d = DemoItem(title='Plotting of 2D (B-scan) data.', exe=plot2d_demo)
demo_plot2d.info = 'This demo will illustrate the usage of the plot2d function \
for metaArray.'
plot_menu.add_item(demo_plot2d)


def plot_complex_demo() -> str:
    """
    Example on matplotlib complex array plotting interface
    """
    code = """
    from scipy.signal.wavelets import morlet
    from metaArray import metaArray
    from metaArray.drv_pylab import plotcomplex, plotcomplexpolar
    from matplotlib.pyplot import show, close

    ary = morlet(1000, w=5.0, s=0.5, complete=True)
    metAry = metaArray(ary)
    metAry['name'] = '5 cycle complex morlet'
    metAry['unit'] = ''
    metAry.set_range(0, 'begin', -0.5)
    metAry.set_range(0, 'end', 0.5)
    metAry.set_range(0, 'unit', '')
    metAry.set_range(0, 'label', 'Scale')

    fig, host, par = plotcomplex(metAry)
    show()
    close(fig)

    fig, host, par = plotcomplexpolar(metAry)
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_plot_complex = DemoItem(title='Plotting of complex number metaArray.',
                             exe=plot_complex_demo)
demo_plot_complex.info = 'This demo will illustrate the usage of the \
plotcomplex, and plotcomplexpolar function for metaArray.'
plot_menu.add_item(demo_plot_complex)


main_menu.add_item(plot_menu)

###############################################################################
# ################      END visualisation demo       ######################## #
###############################################################################
###############################################################################


###############################################################################
# Miscellaneous demos                                                         #
###############################################################################
misc_menu = DemoMenu(title='Misc. (non-metaArray) demos')
misc_menu.info = 'This is a list of demos for miscellaneous non-metaArray \
aware classes and functions. These are useful helper classes and function \
used by various metaArray components.'


def cplx_trig_func_demo() -> str:
    """
    Complex Trigonometric function demo
    """
    code = """
    # Three cycles, and 20 points
    #*****************************
    from metaArray.misc import cplx_trig_func
    func = cplx_trig_func(nLambda = 3, pts = 20)
    print(func())

    # Or, specify 1kHz signal, for 2ms, at 5kHz sampling rate
    #*********************************************************
    func = cplx_trig_func(freq = 1e3, length = 2e-3, samp_rate = 5e3)
    print(func())
    """

    return prcs_demo(code)


demo_cplx_trig_func = DemoItem(title='Complex trigonometry function generator.',
                               exe=cplx_trig_func_demo)
demo_cplx_trig_func.info = 'This demo will illustrate the usage of the \
cplx_trig_func object, it will return a complex numpy ndarray based on the \
given combinations of the following parameters (they are also attributes of \
class instance): nLambda => Number of wavelengths. pts => Number of samples \
(length of the array). freq => Frequency of the desire trigonometry function. \
length => Duration of the function in time. samp_rate => Sampling rate. dt => \
Sampling interval. Only the minimum combination set of these parameters needs \
to be specify. InsufficientInput error will be raised if the instanc is called\
 before the parameters are sufficiently defined.'
misc_menu.add_item(demo_cplx_trig_func)


def misc_units_demo() -> str:
    """
    Misc. unit formatting
    """
    code = """
    # Format number in engineering units
    #************************************
    from metaArray.misc import engUnit
    print(engUnit(1.23456e7, unit = 'eV', sigfig=3))
    print(engUnit(1.23456e8, unit = 'eV', sigfig=3))
    print(engUnit(1.23456e9, unit = 'eV', sigfig=4))

    # Find out a suitable SI unit prefix for a given number
    #*******************************************************
    from metaArray.misc import unitPrefix
    num, name, prefix, exponent = unitPrefix(1.23456e7)
    print('The scaled number is: ' + str(num)); print('SI unit \
prefix name is: ' + name); print('SI unit prefix is: ' + prefix); print('The \
exponent to scale number with is: ' + str(exponent))
    """

    return prcs_demo(code)


demo_misc_units = DemoItem(title='Work out suitable SI unit prefixes.',
                           exe=misc_units_demo)
demo_misc_units.info = ''
misc_menu.add_item(demo_misc_units)


def flist_demo() -> str:
    """
    Misc obtaining a list of files
    """
    code = """
    # Here is how to obtain a list of files under the given directory
    #*****************************************************************
    from metaArray.misc import file_list
    flist = file_list('""" + demo_dir + """')
    for fpath in flist: print('* ' + fpath)

    # You can also specify a particular file name extension, and \
whether or not to search the subdirectories.
    #**************************************************************************
    flist = file_list('""" + demo_dir + """', ext = 'flxhst', SubDir = False)
    for fpath in flist: print('* ' + fpath)
    """

    return prcs_demo(code)


demo_flist = DemoItem(title='Get a list of files under a given directory.',
                      exe=flist_demo)
demo_flist.info = 'This demo illustrate the usage of the file_list function, \
it will obtain a list of files under a given directory. It has the option to \
search for subdirectories (disabled by default), and return only those with \
matching file name extension.'
misc_menu.add_item(demo_flist)

main_menu.add_item(misc_menu)

###############################################################################
# #################      END I/O demo       ################################# #
###############################################################################

###############################################################################
# #################      General Demos      ################################# #
###############################################################################


def basic_demo() -> str:
    """
    Basic metaArray usage
    """
    code = """

    from numpy import linspace, cos, pi
    from metaArray import metaArray
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    # Construct some data
    #*********************
    ary = cos(linspace(-2*pi, 2*pi, 28))
    # Here is how the numpy ndarray look like:
    print(ary)

    # Construct a metaArray from the given numpy ndarray
    #****************************************************
    metaAry = metaArray(ary)
    # Here is how the metaArray look like:
    print(metaAry)

    # metaArray is just a wrapper on top of the ndarray, can directly access
    # the underlying ndarray like this:
    print(metaAry.data)

    # Setting the meta information
    #******************************
    metaAry['name'] = 'cosine function'
    metaAry['unit'] = None     # None => arbitrary unit, '' => unitless
    metaAry['label'] = 'Amplitude'

    # Set the axis attribute
    metaAry.set_range(0, 'begin', -2)     # 1st axis, the first value (x0)
    metaAry.set_range(0, 'end', 2)        # 1st axis, the  value (x1)
    metaAry.set_range(0, 'label', 'lambda')  # label for the 1st axis
    metaAry.set_range(0, 'unit', '')         # 1st axis, unit
    # This is how the metaArray look like now:
    print(metaAry)

    # This is how to retrive the meta data
    #**************************************
    print(metaAry['name'])
    print(metaAry.get_range(0, 'begin'))

    # You can plot the metaArray like this:
    #***************************************
    fig, ax = plot1d(metaAry, legend=-1)
    fig.savefig('demo_basic.png', format='png')
    show()
    close(fig)

    # You can slice the metaArray using array indexies like a ndarry:
    #*****************************************************************
    fig, ax = plot1d(metaAry[5:20], legend=-1)
    fig.savefig('demo_basic_index_slicing.png', format='png')
    show()
    close(fig)

    # Or you can slice the metaArray in real space like this:
    #*********************************************************
    fig, ax = plot1d(metaAry[-0.25:0.75], legend=-1)
    fig.savefig('demo_basic_meta_slicing.png', format='png')
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_basic = DemoItem(title='metaArray basic usage.', exe=basic_demo)
demo_basic.info = 'This demo will illustrate the basic usage of metaArray.'
main_menu.add_item(demo_basic)


def general_demo() -> str:
    """
    Example on meta functions
    """
    code = """
    # Load some data as example
    #***************************
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close
    from metaArray.drv_Tek import isf
    ary = isf('""" + join(demo_dir, 'DPO2000.isf') + """')[1]
    fig, ax = plot1d(ary, legend=-1)
    fig.savefig('demo_isf.png', format='png')
    show()
    close(fig)

    # Do a magnitude FFT, and save only the first 3kHz:
    #***************************************************
    from numpy import log10
    from metaArray.metaTrans import rfft
    fary = abs(rfft(ary))[:1e3]
    fary.log10()       # Put the values on log scale
    fig, ax = plot1d(fary, legend=-1)
    fig.savefig('demo_general_1.png', format='png')
    show()
    close(fig)

    # Try again with good amount of padding for the magnitude FFT:
    #**************************************************************
    from metaArray.metaFunc import padding_calc
    # Say 1024 points between 0 - 3kHz
    fary = rfft(ary, n = padding_calc(ary, min_freq = 0,
                                      max_freq = 1e3,
                                      resolution = 1024))
    fary = abs(fary)[:1e3].log10()
    fig, ax = plot1d(fary, legend=-1)
    fig.savefig('demo_general_2.png', format='png')
    show()
    close(fig)

    # Short-time FFT (STFFT):
    #*************************
    from metaArray.metaTrans import stfft
    from metaArray.drv_pylab import plot2d
    # 0-30kHz, temporal resolution 100, frequency resolution 256:
    tfary = stfft(ary, tres=100, fres=256, fmax=30e3)
    fig, ax = plot2d(tfary.log10())
    fig.savefig('demo_general_3.png', format='png')
    show()
    close(fig)

    # Down sample to 10kHz:
    # Zero group delay low-pass FIR filter will apply when down sampling
    #********************************************************************
    from metaArray.metaFunc import meta_resample
    bry = meta_resample(ary, rate=10e3)
    bry['name'] = 'Down sample to 10kHz'
    fig, ax = plot1d(bry, legend=-1)
    fig.savefig('demo_general_4.png', format='png')
    show()
    close(fig)

    # Up sample again, and look at the difference:
    #**********************************************
    bry = meta_resample(bry, rate=3.125e6)
    bry = ary - bry
    bry['name'] = 'Noise filtered out by down sampling'
    fig, ax = plot1d(bry, legend=-1)
    fig.savefig('demo_general_5.png', format='png')
    show()
    close(fig)

    # A straight forward zero group delay low pass filter at 5kHz:
    #**************************************************************
    from metaArray.metaFunc import meta_lowpass
    bry = meta_lowpass(ary, 5e3)
    fig, ax = plot1d(bry, legend=-1)
    fig.savefig('demo_general_6.png', format='png')
    show()
    close(fig)

    # A similar zero group delay high pass filter at 5kHz:
    #******************************************************
    from metaArray.metaFunc import meta_highpass
    bry = meta_highpass(ary, 5e3)
    fig, ax = plot1d(bry, legend=-1)
    fig.savefig('demo_general_7.png', format='png')
    show()
    close(fig)

    """

    return prcs_demo(code)


demo_general = DemoItem(title='metaArray general usage.', exe=general_demo)
demo_general.info = 'This demo will illustrate the usage of some of the more \
advanced usages of metaArray functions.'
main_menu.add_item(demo_general)


def hist_demo() -> str:
    """
    Example on meta functions
    """
    code = """
    from numpy import round
    from numpy.random import rand
    from metaArray import metaArray
    from metaArray.metaFunc import histogram
    from metaArray.drv_pylab import plot1d
    from matplotlib.pyplot import show, close

    metAry = metaArray(rand(100000)-0.5)
    metAry['name'] = 'a random distribution'
    hist = histogram(metAry, bins = 20)
    fig, ax = plot1d(hist)
    show()
    close(fig)

    for i in range(99): metAry.data += rand(100000)-0.5

    # If the data itself is already quantised into regular steps, \
such as by rounding it to the nearest integer value, but the steps need not be\
 integer values.
    #**************************************************************************
    metAry.data = round(metAry.data)
    metAry['name'] = 'the sum of 100 random distributions'

    # Then there is no need to specify number of bins
    hist = histogram(metAry)
    fig, ax = plot1d(hist)
    show()
    close(fig)
    """

    return prcs_demo(code)


demo_hist = DemoItem(title='metaArray histogram usage.', exe=hist_demo)
demo_hist.info = 'This demo will illustrate how to bin a metaArray into a \
histogram.'
main_menu.add_item(demo_hist)


###############################################################################
# #################      END General demo       ############################# #
###############################################################################


def demo():
    return main_menu()


if __name__ == "__main__":
    main_menu()
