# -*- coding: utf-8 -*-

""" Class to run the wavelet decomposition based source detection.
"""

import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve


### Define wavelet decomposition class
class Wvdet:
    def __init__(self, imgFile, expFile=None, bkgFile=None, chatty=True):
        """
        Input:

        imgFile (str): Path to event file that stores the photon image
        """
        self.imgFile = imgFile
        self.expFile = expFile
        self.bkgFile = bkgFile
        self.chatty = chatty
        
        if self.chatty:
            print('load {0}'.format(imgFile))
        self.loadFitsImage(imgFile, 'img')
        
        
        if expFile is not None:
            if self.chatty:
                print('load {0}'.format(expFile))
            self.loadFitsImage(expFile, 'exp')
        else:
            self.expHdr, self.expData = None, np.ones(np.shape(self.imgData))

            
        if bkgFile is not None:
            if self.chatty:
                print('load {0}'.format(bkgFile))
            self.loadFitsImage(bkgFile, 'bkg')
        else:
            self.bkgHdr, self.bkgData = None, np.zeros(np.shape(self.imgData))

        self.imgPath = os.path.dirname(self.imgFile)
        self.wvprefix = os.path.join(self.imgPath, os.path.basename(self.imgFile).split('.')[0] + '_wv')
        self.eedScales = ('', '.5', '.6', '.7')
        self.pedScales = ('', '.2', '.3', '.4', '.5')



    def wvdecomp(self, runPedFlag=False, convFlag=False, runEedFlag=False, runSexFlag=False):

        if runPedFlag:
            self.runWvdecompPedScales()        
        self.loadWvdecompPedScales()

        baseName = 'wv'
        ### use gaussian smoothed images instead of PSF image
        self.usePed = 'G'           
        kernelSizes = [16, 32]
        if convFlag:
            self.smoothImage(baseName, kernelSizes)
        else:
            for kk in kernelSizes:
                finput = os.path.join(os.path.dirname(self.imgFile), '{0}S{1}'.format(baseName,kk))
                if self.chatty:
                    print('load %s...' %(finput))
                self.loadFitsImage(finput, '{0}S{1}'.format(baseName,kk))


        ### Create new background image for EED scales
        ### Add PED scales to background
        kernelNorms = [0.47, 0.1]
        self.bkgData = self.bkgData + getattr(self, '{0}Data'.format(baseName))
        for nn,kk in zip(kernelNorms,kernelSizes):
            self.bkgData = self.bkgData + nn*getattr(self, '{0}S{1}Data'.format(baseName,kk))


        self.newBkgFile = os.path.join(os.path.dirname(self.imgFile), '{0}{1}bkg'.format(baseName,self.usePed))
        self.saveFitsImage(getattr(self, '{0}Hdr'.format(baseName)), self.bkgData, self.newBkgFile )


        if runEedFlag:
            self.runWvdecompEedScales()        
        self.loadWvdecompEedScales()

        if runSexFlag:
            self.runSextractor()
    

    def runWvdecompEedScales(self):
        ''' run wvdecomp on extended emission detection scales
        '''
        if os.system("which wvdecomp") != 0:
            raise OSError('zhtools does not seem to be initialized...')
        
        cmd = "wvdecomp {0} {1}{2} t=7. tmin=3.3 iter=5 smin=5 smax=6 detectnegative=no stat=poisson".format(self.imgFile, self.wvprefix, self.usePed)
  
        if self.expFile is not None:
            cmd += ' exp={0}'.format(self.expFile)
            
        ### Add background image (including PED scales)
        cmd += ' bg={0}'.format(self.newBkgFile)

        if self.chatty:
            print(cmd)
        os.system(cmd)


        ### copy WCS from image to wavelet files
        cmd = "copywcs {0}".format(self.imgFile)

        for wvscale in self.eedScales:
            cmd += ' {0}{1}{2}'.format(self.wvprefix, self.usePed, wvscale)

        if self.chatty:
            print(cmd)
        os.system(cmd)

        ### create input image for sextractor
        cmd = "imarith {0} = {1}{2} '/' {3}".format(os.path.join(self.imgPath, 'sex_input_wvhs'),
            self.wvprefix, self.usePed, self.expFile)

        if self.chatty:
            print(cmd)
        os.system(cmd)

        cmd = "imcarith {0} = {0} '*' 1.e10".format(os.path.join(self.imgPath, 'sex_input_wvhs'))

        if self.chatty:
            print(cmd)
        os.system(cmd)


    def runSextractor(self):
        cwd = os.getcwd()
        os.chdir(self.imgPath)
        
        if os.system("which sex") != 0:
            raise OSError('sextractor does not seem to be initialized...')

        cmd = "sex {0}".format('sex_input_wvhs')
        if self.chatty:
            print(cmd)
        os.system(cmd)
        os.system("mv wv.cat {0}".format('wvhs.cat')) 

        ### read in sextractor output
        self.sexDf = pd.read_table('wvhs.cat', header=None, skiprows=8, sep='\s+',
            usecols=[3,4,5,6,7], names=['x_pos', 'y_pos', 'a_rms', 'b_rms', 'angle']) 

        for rms in ('a_rms', 'b_rms'):
            self.sexDf[rms] *= 2

        ### create simple region file
        with open('wvhs.reg', 'w') as file_:
            for sexRow in self.sexDf.values:
                rowStr = 'image;ellipse({0},{1},{2},{3},{4})\n'.format(*sexRow)
                file_.write(rowStr)

        os.chdir(cwd)




    def loadWvdecompEedScales(self):    
        for wvscale in self.eedScales:
            if self.chatty:
                print('load {0}{1}{2}[0]...'.format(self.wvprefix, self.usePed, wvscale))
            self.loadFitsImage('{0}{1}{2}[0]'.format(self.wvprefix, self.usePed, wvscale), 'wv{0}{1}'.format(self.usePed, wvscale.replace('.','')))


    def runWvdecompPedScales(self):
        ''' run wvdecomp on point-like emission detection scales
        '''
        if os.system("which wvdecomp") != 0:
            raise OSError('zhtools does not seem to be initialized...')
        
        cmd = "wvdecomp {0} {1} t=3.3 tmin=1. iter=5 smin=2 smax=4 detectnegative=no stat=poisson".format(self.imgFile, self.wvprefix)
        if self.expFile is not None:
            cmd += ' exp={0}'.format(self.expFile)
            
        if self.bkgFile is not None:
            cmd += ' bg={0}'.format(self.bkgFile)

        if self.chatty:
            print(cmd)
        os.system(cmd)

        ### copy WCS from image to wavelet files
        cmd = "copywcs {0}".format(self.imgFile)

        for wvscale in self.pedScales:
            cmd += ' {0}{1}'.format(self.wvprefix, wvscale)

        if self.chatty:
            print(cmd)
        os.system(cmd)
    

    def loadWvdecompPedScales(self):    
        for wvscale in self.pedScales:
            if self.chatty:
                print('load {0}{1}[0]...'.format(self.wvprefix, wvscale))
            self.loadFitsImage('{0}{1}[0]'.format(self.wvprefix, wvscale), 'wv{0}'.format(wvscale.replace('.','')))
    
    
    def smoothImage(self, baseName, kernelSizes):
        img = getattr(self, '{0}Data'.format(baseName))
        hdr = getattr(self, '{0}Hdr'.format(baseName))

        for kk in np.atleast_1d(kernelSizes):
            kernel = Gaussian2DKernel(x_stddev=kk)
            conv = convolve(img, kernel)
            setattr(self, '{0}S{1}Data'.format(baseName,kk), conv)

            outfile = os.path.join(os.path.dirname(self.imgFile), '{0}S{1}'.format(baseName,kk))
            self.saveFitsImage(hdr, conv, outfile)

    
    def saveFitsImage(self, fhdr, fout, outfile):
        hdu = fits.PrimaryHDU(fout, header=fhdr)
        hdu.writeto(outfile, overwrite=True)
   

    def loadFitsImage(self, finput, attrStr):
        """ Function to check the validity of the input fits file
        
        Input:

        finput (str): Path to image file, e.g. "./examples/000/evt_000_a.fits[0]"
        
        Returns:
        header and image instances
        """

        if finput[-1] == ']':
            ext = int(finput.split('[')[-1].strip(']'))
            finput = finput.split('[')[0]
        else:
            if self.chatty:
                print('No extension number provided, it is set to 0...')
            ext = 0

        if not os.path.isfile(finput):
            raise FileNotFoundError('Input fits file not found: {0}...'.format(finput))

        with fits.open(finput) as hdul:
            if len(hdul)-1 < ext:
                raise FileNotFoundError('Provided extension number {0} not found...'.format(ext))
    
            hdr  = hdul[ext].header

            if ext == 0 and hdr['NAXIS'] == 2:
                data = hdul[ext].data
            elif ext != 0 and hdr['XTENSION'] == 'IMAGE':
                data = hdul[ext].data
            else:
                raise FileNotFoundError('No IMAGE found in extension {0}...'.format(ext))
        
        setattr(self, '{0}Hdr'.format(attrStr), hdr)
        setattr(self, '{0}Data'.format(attrStr), data)