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

        imgFile (str): Path to file that stores the photon image
        expFile (str): Path to file that stores the exposure map (optional)
        bkgFile (str): Path to file that stores the background image (optional)
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
        self.baseName = 'wv'
        self.wvprefix = os.path.join(self.imgPath, os.path.basename(self.imgFile).split('.')[0] + '_{0}'.format(self.baseName))
        
        ### PED scales definitions
        self.pedScales = ('', '.2', '.3', '.4', '.5')
        self.usePed = 'G'
        pedBkgSuffix = '{0}{1}bkg'.format(self.baseName,self.usePed)
        self.pedBkgFile = os.path.join(os.path.dirname(self.imgFile), pedBkgSuffix)



    def wvdecomp(self, runPedFlag=True, runEedFlag=True, ttm=[7.,3.3], ssm=[5,6], suffix='mc', runSexFlag=True):
        """ Helper function to run all steps of the detection scheme
        """

        if runPedFlag:
            self.runPedScales()

        if runEedFlag:
            self.runWvdecompEedScales(ttm, ssm, suffix)

        if runSexFlag:
            self.runSextractor(suffix)


    def runPedScales(self, convFlag=True):
        """
        Run detection of point sources on PED scales (scales 2-4).
        Create PSF templates by smoothing wavelet images with Gaussians.
        Model the predicted point-source emission using normalization coefficients and PSF templates.

        Output:

        *_wv (file): Wavelet image of combined scales 2-4
        *_wv.{2,3,4} (files): Wavelet images of individual scales 2-4
        *_wv.5 (file): Residual image
        
        wvS16, wvS32 (files): Gaussian smoothed images of the *_wv file
        wvGbkg (file): Background image including predicted point-source emission
        """

        self.runWvdecompPedScales()        
        self.loadPedScales()
      
        ### use gaussian smoothed images instead of PSF image           
        kernelSizes = [16, 32]
        if convFlag:
            self.smoothImage(self.baseName, kernelSizes)
        else:
            for kk in kernelSizes:
                finput = os.path.join(os.path.dirname(self.imgFile), '{0}S{1}'.format(self.baseName,kk))
                if self.chatty:
                    print('load %s' %(finput))
                self.loadFitsImage(finput, '{0}S{1}'.format(self.baseName,kk))

        ### Create new background image for EED scales
        ### Add PED scales to background
        kernelNorms = [0.47, 0.1]
        self.pedBkgData = self.bkgData + getattr(self, '{0}Data'.format(self.baseName))
        for nn,kk in zip(kernelNorms,kernelSizes):
            self.pedBkgData = self.pedBkgData + nn*getattr(self, '{0}S{1}Data'.format(self.baseName,kk))

        self.saveFitsImage(getattr(self, '{0}Hdr'.format(self.baseName)), self.pedBkgData, self.pedBkgFile )


    def runWvdecompPedScales(self):
        """ Run wvdecomp on point-like-emission detection scales
        """

        if os.system("which wvdecomp") != 0:
            raise OSError('zhtools does not seem to be initialized')
        
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
    

    def loadPedScales(self):    
        for wvscale in self.pedScales:
            if self.chatty:
                print('load {0}{1}[0]'.format(self.wvprefix, wvscale))
            self.loadFitsImage('{0}{1}[0]'.format(self.wvprefix, wvscale), 'wv{0}'.format(wvscale.replace('.','')))
        

    def runWvdecompEedScales(self, ttm=[7.,3.3], ssm=[5,6], suffix='mc'):
        """
        Run wvdecomp on extended-emission-detection scales to check for residual signal
        over the background and point-source emission on EED scales.

        Output:
        *_wvG{suffix} (file): Wavelet image of combined scales ssm[0]-ssm[1]
        *_wvG{suffix}.{ssm[0]-ssm[1]} (files): Wavelet images of individual scales
        *_wvG{suffix}.{ssm[1]+1} (file): Residual image

        sex_input_wv{suffix} (file): Input image for sextractor  
        """ 

        if os.system("which wvdecomp") != 0:
            raise OSError('zhtools does not seem to be initialized')
        
        base = "{0}{1}{2}".format(self.wvprefix, self.usePed, suffix)
        
        cmd = " ".join((
            "wvdecomp {0} {1}".format(self.imgFile, base),
            "t={0:.1f} tmin={1:.1f}".format(*ttm),
            "smin={0} smax={1}".format(*ssm),
            "iter=5 detectnegative=no stat=poisson"
            ))
  
        if self.expFile is not None:
            cmd += ' exp={0}'.format(self.expFile)
            
        ### add background image (including PED scales)
        cmd += ' bg={0}'.format(self.pedBkgFile)

        if self.chatty:
            print(cmd)
        os.system(cmd)


        ### copy WCS from image to wavelet files
        cmd = "copywcs {0}".format(self.imgFile)

        eedScales = [''] + ['.{0}'.format(ii) for ii in range(ssm[0],ssm[1]+2)]
        ### e.g.: ('', '.5', '.6', '.7') for ssm=[5,6]

        for wvscale in eedScales:
            cmd += ' {0}{1}'.format(base, wvscale)

        if self.chatty:
            print(cmd)
        os.system(cmd)

        ### create input image for sextractor
        sexFile = os.path.join(self.imgPath, 'sex_input_{0}{1}'.format(self.baseName, suffix))

        cmd = "imarith {0} = {1} '/' {2}".format(sexFile, base, self.expFile)

        if self.chatty:
            print(cmd)
        os.system(cmd)

        cmd = "imcarith {0} = {0} '*' 1.e10".format(sexFile)

        if self.chatty:
            print(cmd)
        os.system(cmd)


    def loadEedScales(self, ssm=[5,6], suffix='mc'):
        eedScales = [''] + ['.{0}'.format(ii) for ii in range(ssm[0],ssm[1]+2)]

        for wvscale in eedScales:
            ffile = '{0}{1}{2}{3}[0]'.format(self.wvprefix, self.usePed, wvscale, suffix)
            if self.chatty:
                print('load {0}'.format(ffile))
            self.loadFitsImage(ffile, 'wv{0}{1}{2}'.format(self.usePed, wvscale.replace('.',''), suffix))


    def runSextractor(self, suffix='mc'):
        """ Run sextractor

        Output:

        wv{suffix}.cat (file): Sextractor output
        wv{suffix}.reg (file): Ds9 region file for sextractor output
        """

        cwd = os.getcwd()
        os.chdir(self.imgPath)
        
        if os.system("which sex") != 0:
            raise OSError('sextractor does not seem to be initialized')

        preFix = '{0}{1}'.format(self.baseName, suffix) # whhs
        sexFile = 'sex_input_{0}'.format(preFix)

        cmd = "sex {0}".format(sexFile)
        if self.chatty:
            print(cmd)
        os.system(cmd)
        os.system("mv wv.cat {0}.cat".format(preFix)) 

        ### read in sextractor output
        self.sexDf = pd.read_table('{0}.cat'.format(preFix), header=None, skiprows=8, sep='\s+',
            usecols=[3,4,5,6,7], names=['x_pos', 'y_pos', 'a_rms', 'b_rms', 'angle']) 

        for rms in ('a_rms', 'b_rms'):
            self.sexDf[rms] *= 2

        ### create simple region file
        with open('{0}.reg'.format(preFix), 'w') as file_:
            for sexRow in self.sexDf.values:
                rowStr = 'image;ellipse({0},{1},{2},{3},{4})\n'.format(*sexRow)
                file_.write(rowStr)

        os.chdir(cwd)

    
    def smoothImage(self, baseName, kernelSizes):
        img = getattr(self, '{0}Data'.format(baseName))
        hdr = getattr(self, '{0}Hdr'.format(baseName))

        for kk in np.atleast_1d(kernelSizes):
            if self.chatty:
                print('Convolve {0} with kernel {1}'.format(baseName, kk))
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
                print('No extension number provided, it is set to 0')
            ext = 0

        if not os.path.isfile(finput):
            raise FileNotFoundError('Input fits file not found: {0}'.format(finput))

        with fits.open(finput) as hdul:
            if len(hdul)-1 < ext:
                raise FileNotFoundError('Provided extension number {0} not found'.format(ext))
    
            hdr  = hdul[ext].header

            if ext == 0 and hdr['NAXIS'] == 2:
                data = hdul[ext].data
            elif ext != 0 and hdr['XTENSION'] == 'IMAGE':
                data = hdul[ext].data
            else:
                raise FileNotFoundError('No IMAGE found in extension {0}'.format(ext))
        
        setattr(self, '{0}Hdr'.format(attrStr), hdr)
        setattr(self, '{0}Data'.format(attrStr), data)