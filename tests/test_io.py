# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 2024 10:55

@author: samhill

Tests I/O functionality of module
"""

from pathlib import Path
import typing
import pytest
import os

import metaArray as ma


def isf_files() -> typing.Iterator[str]:
    # Generates list of all isf files in directory
    basedir = os.path.join('.', 'src', 'metaArray', 'example')
    files = (os.path.join(basedir, fi) for fi in os.listdir(basedir)
             if fi.endswith('.isf'))
    return files


def test_DPO2000_isf_good():
    # Test that we can read in isf file from DPO2000 scope
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            'DPO2000.isf')
    buff = ma.isf(filename)
    assert len(buff)


def test_DPO2000_isf_bad():
    # Test that we can read in isf file from DPO2000 scope
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            'DPO2000.csv')
    buff = ma.isf(filename)
    assert len(buff) == 0


@pytest.mark.parametrize("filename", isf_files())
def test_isfs(filename):
    # Test that we can read_in isf file
    buff = ma.isf(filename)
    assert len(buff)


def test_DPO2000_csv_good():
    # Test that we can read in isf file from DPO2000 scope
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            'DPO2000.csv')
    buff = ma.DPO2000_csv(filename)
    assert len(buff)


def test_TDS2000_csv_good():
    # Test that we can read in isf file from TDS2000 scope
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            'TDS2000.csv')
    buff = ma.TDS2000_csv(filename)
    assert len(buff)


def test_flxhst():
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            '2D_Al_10_Avg.flxhst')

    flxhst = ma.pout_hist(filename)
    assert flxhst.fname == Path(filename).stem


def test_flxhst_fnf():
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            'NoFile.flxhst')
    with pytest.raises(FileNotFoundError):
        ma.pout_hist(filename)


def test_flxhst_wrongfile():
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            '3D_billet_10mm_40mm.flxdato')
    with pytest.raises(ValueError):
        ma.pout_hist(filename)


def test_flxdato():
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            '3D_billet_10mm_40mm.flxdato')
    flxdato = ma.data_out1(filename)
    assert flxdato.fname == Path(filename).stem


def test_flxdato_fnf():
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            'NoFile.flxhst')
    with pytest.raises(FileNotFoundError):
        ma.data_out1(filename)


def test_flxdato_wrongfile():
    filename = os.path.join('.', 'src', 'metaArray', 'example',
                            '2D_Al_10_Avg.flxhst')
    with pytest.raises(ValueError):
        ma.data_out1(filename)
