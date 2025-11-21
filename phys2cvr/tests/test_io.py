#!/usr/bin/env python3
"""Tests for io."""

import os

import nibabel as nib
import numpy as np

from phys2cvr import io

# ## Unit tests


def test_load_nifti_get_mask(nifti_data, nifti_mask):
    # Test loading a nifti file with default arguments
    fname, expected_data = nifti_data
    data, mask, img = io.load_nifti_get_mask(fname)
    assert np.array_equal(data, expected_data)
    assert np.array_equal(mask, expected_data.any(axis=-1).squeeze())
    assert isinstance(img, nib.nifti1.Nifti1Image)

    # Test loading a nifti mask file with is_mask=True
    fname, expected_data = nifti_mask
    data, mask, img = io.load_nifti_get_mask(fname, is_mask=True)
    assert np.array_equal(data, expected_data)
    assert np.array_equal(mask, expected_data != 0)
    assert isinstance(img, nib.nifti1.Nifti1Image)


def test_export_regressor(testdir):
    petco2hrf_shift = np.random.rand(10)
    ntp = 10
    outname = os.path.join(testdir, 'test_regressor')
    suffix = 'petco2hrf'
    ext = '.1D'

    petco2hrf_demean = io.export_regressor(
        petco2hrf_shift, ntp, outname, suffix=suffix, ext=ext
    )

    pco2 = petco2hrf_shift - petco2hrf_shift.mean()
    assert np.allclose(petco2hrf_demean, pco2, atol=1e-6, rtol=0)

    # Check if file was saved and has the correct content
    saved_file = np.loadtxt(f'{outname}_{suffix}{ext}')
    assert np.allclose(saved_file, pco2, atol=1e-6, rtol=0)
    os.remove(f'{outname}_{suffix}{ext}')


def test_export_nifti(testdir):
    # create some test data
    data = np.random.rand(10, 10, 10)
    affine = np.eye(4)
    header = nib.Nifti1Header()
    img = nib.Nifti1Image(data, affine, header)

    # create a temporary directory to store the output file
    fname = os.path.join(testdir, 'test_output.nii.gz')
    io.export_nifti(data, img, fname)

    # check if the output file exists
    assert os.path.exists(fname)

    # load the output file and check if the data matches the input
    out_img = nib.load(fname)
    assert np.allclose(out_img.get_fdata(), data, atol=1e-6, rtol=0)
    assert np.allclose(out_img.affine, affine, atol=1e-6, rtol=0)
    assert out_img.header.__dict__.keys() == header.__dict__.keys()
    # Eventually check that headers have the same content (although they don't now
    # for good reasons!)


# ## Break tests
