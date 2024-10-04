import os
import warnings
import numpy as np
from io import BytesIO
from copy import deepcopy
import astropy.io.fits as fits


class Background(object):

    def __init__(self, bkg_file, ii=None):
        if ii is None:
            self.read_bkg(bkg_file)
        else:
            self.read_bkg2(bkg_file, int(ii))


    def read_bkg(self, bkg_file):
        if isinstance(bkg_file, BytesIO):
            bkg_file = deepcopy(bkg_file)
            self.abspath = bkg_file.name
            self.file_name = bkg_file.name
        else:
            self.abspath = os.path.abspath(bkg_file)
            self.file_name = bkg_file.split('/')[-1]
        bkg_hdu = fits.open(bkg_file, ignore_missing_simple=True)
        self.specExt = bkg_hdu["SPECTRUM"]

        self.specData = self.specExt.data
        self.numDetChans = self.specExt.header['NAXIS2']

        try:
            self.minDetChans = self.specExt.header['TLMIN1']
            self.maxDetChans = self.specExt.header['TLMAX1']
        except KeyError:
            self.minDetChans = 1
            self.maxDetChans = self.numDetChans
        else:
            assert (self.maxDetChans - self.minDetChans + 1) == self.numDetChans, 'please check header!'

        self.ChanIndex = self.specData['CHANNEL'].astype(int)
        self.BkgExpo = np.float128(self.specExt.header['EXPOSURE'])
        try:
            self.BkgCounts = self.specData['COUNTS'].astype(np.float128)
            try:
                self.BkgErr = self.specData['STAT_ERR'].astype(np.float128)
            except KeyError:
                warnings.warn('%s:\nbkg error is not specified and will default to Poisson error'%bkg_file)
                self.BkgErr = np.sqrt(self.BkgCounts)
        except KeyError:
            self.BkgCounts = self.specData['RATE'].astype(np.float128) * self.BkgExpo
            try:
                self.BkgErr = self.specData['STAT_ERR'].astype(np.float128) * self.BkgExpo
            except KeyError:
                warnings.warn('%s:\nbkg error is not specified and will default to Poisson error'%bkg_file)
                self.BkgErr = np.sqrt(self.BkgCounts)

        try:
            self.BkgBackSc = self.specData['BACKSCAL'].astype(float)
        except KeyError:
            try:
                self.BkgBackSc = float(self.specExt.header['BACKSCAL'])
            except KeyError:
                self.BkgBackSc = 1

        bkg_hdu.close()


    def read_bkg2(self, bkg_file, ii):
        if isinstance(bkg_file, BytesIO):
            bkg_file = deepcopy(bkg_file)
            self.abspath = bkg_file.name
            self.file_name = bkg_file.name
        else:
            self.abspath = os.path.abspath(bkg_file)
            self.file_name = bkg_file.split('/')[-1]
        bkg_hdu = fits.open(bkg_file, ignore_missing_simple=True)
        self.specExt = bkg_hdu["SPECTRUM"]

        self.specData = self.specExt.data

        self.BkgExpo = self.specData['EXPOSURE'][ii].astype(np.float128)
        try:
            self.BkgCounts = self.specData['COUNTS'][ii].astype(np.float128)
            try:
                self.BkgErr = self.specData['STAT_ERR'][ii].astype(np.float128)
            except KeyError:
                warnings.warn('%s:%d\nbkg error is not specified and will default to Poisson error'%(bkg_file, ii))
                self.BkgErr = np.sqrt(self.BkgCounts)
        except KeyError:
            self.BkgCounts = self.specData['RATE'][ii].astype(np.float128) * self.BkgExpo
            try:
                self.BkgErr = self.specData['STAT_ERR'][ii].astype(np.float128) * self.BkgExpo
            except KeyError:
                warnings.warn('%s:%d\nbkg error is not specified and will default to Poisson error'%(bkg_file, ii))
                self.BkgErr = np.sqrt(self.BkgCounts)

        try:
            self.numDetChans = self.specExt.header['DETCHANS']
        except KeyError:
            self.numDetChans = len(self.BkgCounts)

        try:
            self.ChanIndex = self.specData['CHANNEL'][ii].astype(int)
        except KeyError:
            self.ChanIndex = np.arange(1, self.numDetChans + 1).astype(int)

        self.minDetChans = min(self.ChanIndex)
        self.maxDetChans = max(self.ChanIndex)

        try:
            self.BkgBackSc = self.specData['BACKSCAL'][ii].astype(float)
        except KeyError:
            try:
                self.BkgBackSc = self.specExt.header['BACKSCAL']
            except KeyError:
                self.BkgBackSc = 1

        bkg_hdu.close()


    def scaling(self, src_sc, bkg_sc):
        sc = src_sc / bkg_sc
        self.BkgCounts = self.BkgCounts * sc
        self.BkgErr = self.BkgErr * sc


    def grouping(self, Grpg):
        self.Grpg = np.array(Grpg)
        if (self.Grpg == 0).all():
            pass
        else:
            NewChIdx = 0
            self.New_ChanIndex = []
            self.New_BkgCounts = []
            self.New_BkgErr = []
            for i, flag in enumerate(self.Grpg):
                if (NewChIdx == 0 and flag == -1) or (flag == 0):
                    continue
                if flag == 1:
                    NewChIdx += 1
                    self.New_ChanIndex.append(NewChIdx)
                    self.New_BkgCounts.append(self.BkgCounts[i])
                    self.New_BkgErr.append(self.BkgErr[i])
                elif flag == -1:
                    self.New_BkgCounts[-1] += self.BkgCounts[i]
                    self.New_BkgErr[-1] = np.sqrt(self.New_BkgErr[-1] ** 2 + self.BkgErr[i] ** 2)
            self.ChanIndex = np.array(self.New_ChanIndex).astype(int)
            self.BkgCounts = np.float128(self.New_BkgCounts)
            self.BkgErr = np.float128(self.New_BkgErr)

            self.minDetChans = min(self.ChanIndex)
            self.maxDetChans = max(self.ChanIndex)
            self.numDetChans = len(self.ChanIndex)


    @property
    def info(self):
        print('+-----------------------------------------------+')
        print(self.abspath)
        print(' There are %d channels from %d to %d.' % (self.numDetChans, self.minDetChans, self.maxDetChans))
        print(' There are %.2f photon events during exposure %.2f s.' % (sum(self.BkgCounts), self.BkgExpo))
        print('+-----------------------------------------------+\n')
