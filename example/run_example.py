# -*- coding: utf-8 -*-

""" This is an example on how to run the wavelet decomposition based source detection.
"""

import sys
sys.path.append('../')

import wvdet

def main():
    # Define input files
    imgFile = './patch_agn_clu.fits[0]'
    expFile = './patch_exp.fits[0]'

    skyPatch = wvdet.Wvdet(imgFile, expFile=expFile, bkgFile=None)

    # Run every step of the maximally clean (i.e., 7sigma) detection scheme
    # This is the default and EED output files will have the suffix 'mc'
    skyPatch.wvdecomp()

    # Run the maximally sensitive (i.e., 4sigma) detection scheme
    # The point-source-emission-detection scales are the same, thus set: runPedFlag=False
    # We need to change the primary and secondary filtering thresholds by setting 'ttm'
    skyPatch.wvdecomp(runPedFlag=False, ttm=[4.,1.6], suffix='ms')


if __name__ == '__main__':
    main()