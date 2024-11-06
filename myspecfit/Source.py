import os
import warnings
import numpy as np
from io import BytesIO
from copy import deepcopy
import astropy.io.fits as fits


class Source(object):

    def __init__(self, src_file, ii=None):
        if ii is None:
            self.read_src(src_file)
        else:
            self.read_src2(src_file, int(ii))


    def read_src(self, src_file):
        if isinstance(src_file, BytesIO):
            src_file = deepcopy(src_file)
            self.abspath = src_file.name
            self.file_name = src_file.name
        else:
            self.abspath = os.path.abspath(src_file)
            self.file_name = src_file.split('/')[-1]
        src_hdu = fits.open(src_file, ignore_missing_simple=True)
        self.specExt = src_hdu["SPECTRUM"]

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
        self.SrcExpo = np.float64(self.specExt.header['EXPOSURE'])
        try:
            self.SrcCounts = self.specData['COUNTS'].astype(np.float64)
            try:
                self.SrcErr = self.specData['STAT_ERR'].astype(np.float64)
            except KeyError:
                warnings.warn('%s:\nsrc error is not specified and will default to Poisson error'%src_file)
                self.SrcErr = np.sqrt(self.SrcCounts)
        except KeyError:
            self.SrcCounts = self.specData['RATE'].astype(np.float64) * self.SrcExpo
            try:
                self.SrcErr = self.specData['STAT_ERR'].astype(np.float64) * self.SrcExpo
            except KeyError:
                warnings.warn('%s:\nsrc error is not specified and will default to Poisson error'%src_file)
                self.SrcErr = np.sqrt(self.SrcCounts)

        try:
            self.SrcQual = self.specData['QUALITY'].astype(int)
        except KeyError:
            self.SrcQual = np.zeros(len(self.ChanIndex)).astype(int)

        try:
            self.SrcGrpg = self.specData['GROUPING'].astype(int)
        except KeyError:
            self.SrcGrpg = np.zeros(len(self.ChanIndex)).astype(int)

        try:
            self.SrcBackSc = self.specData['BACKSCAL'].astype(float)
        except KeyError:
            try:
                self.SrcBackSc = float(self.specExt.header['BACKSCAL'])
            except KeyError:
                self.SrcBackSc = 1

        src_hdu.close()


    def read_src2(self, src_file, ii):
        if isinstance(src_file, BytesIO):
            src_file = deepcopy(src_file)
            self.abspath = src_file.name
            self.file_name = src_file.name
        else:
            self.abspath = os.path.abspath(src_file)
            self.file_name = src_file.split('/')[-1]
        src_hdu = fits.open(src_file, ignore_missing_simple=True)
        self.specExt = src_hdu["SPECTRUM"]

        self.specData = self.specExt.data

        self.SrcExpo = self.specData['EXPOSURE'][ii].astype(np.float64)
        try:
            self.SrcCounts = self.specData['COUNTS'][ii].astype(np.float64)
            try:
                self.SrcErr = self.specData['STAT_ERR'][ii].astype(np.float64)
            except KeyError:
                warnings.warn('%s:%d\nsrc error is not specified and will default to Poisson error'%(src_file, ii))
                self.SrcErr = np.sqrt(self.SrcCounts)
        except KeyError:
            self.SrcCounts = self.specData['RATE'][ii].astype(np.float64) * self.SrcExpo
            try:
                self.SrcErr = self.specData['STAT_ERR'][ii].astype(np.float64) * self.SrcExpo
            except KeyError:
                warnings.warn('%s:%d\nsrc error is not specified and will default to Poisson error'%(src_file, ii))
                self.SrcErr = np.sqrt(self.SrcCounts)

        try:
            self.numDetChans = self.specExt.header['DETCHANS']
        except KeyError:
            self.numDetChans = len(self.SrcCounts)

        try:
            self.ChanIndex = self.specData['CHANNEL'][ii].astype(int)
        except KeyError:
            self.ChanIndex = np.arange(1, self.numDetChans + 1).astype(int)

        self.minDetChans = min(self.ChanIndex)
        self.maxDetChans = max(self.ChanIndex)

        try:
            self.SrcQual = self.specData['QUALITY'][ii].astype(int)
        except KeyError:
            self.SrcQual = np.zeros(self.numDetChans).astype(int)

        try:
            self.SrcGrpg = self.specData['GROUPING'][ii].astype(int)
        except KeyError:
            self.SrcGrpg = np.zeros(self.numDetChans).astype(int)

        try:
            self.SrcBackSc = self.specData['BACKSCAL'][ii].astype(float)
        except KeyError:
            try:
                self.SrcBackSc = self.specExt.header['BACKSCAL']
            except KeyError:
                self.SrcBackSc = 1

        src_hdu.close()


    def grouping(self, Grpg):
        self.Grpg = np.array(Grpg)
        if (self.Grpg == 0).all():
            pass
        else:
            NewChIdx = 0
            self.New_ChanIndex = []
            self.New_SrcCounts = []
            self.New_SrcErr = []
            for i, flag in enumerate(self.Grpg):
                if (NewChIdx == 0 and flag == -1) or (flag == 0):
                    continue
                if flag == 1:
                    NewChIdx += 1
                    self.New_ChanIndex.append(NewChIdx)
                    self.New_SrcCounts.append(self.SrcCounts[i])
                    self.New_SrcErr.append(self.SrcErr[i])
                elif flag == -1:
                    self.New_SrcCounts[-1] += self.SrcCounts[i]
                    self.New_SrcErr[-1] = np.sqrt(self.New_SrcErr[-1] ** 2 + self.SrcErr[i] ** 2)
            self.ChanIndex = np.array(self.New_ChanIndex).astype(int)
            self.SrcCounts = np.float64(self.New_SrcCounts)
            self.SrcErr = np.float64(self.New_SrcErr)

            self.minDetChans = min(self.ChanIndex)
            self.maxDetChans = max(self.ChanIndex)
            self.numDetChans = len(self.ChanIndex)


    @property
    def info(self):
        print('+-----------------------------------------------+')
        print(self.abspath)
        print(' There are %d channels from %d to %d.' % (self.numDetChans, self.minDetChans, self.maxDetChans))
        print(' There are %.2f photon events during exposure %.2f s.' % (sum(self.SrcCounts), self.SrcExpo))
        print('+-----------------------------------------------+\n')
