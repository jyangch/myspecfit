import os
import numpy as np
from io import BytesIO
from copy import deepcopy
import astropy.io.fits as fits
from Tools import intersection, union


class Response(object):

    def __init__(self, rsp_file, specT=None, ii=None):
        if (type(rsp_file) is not list) or (type(rsp_file) is list and len(rsp_file) == 1):
            self.rsp_type = 'rsp'
            if type(rsp_file) is list: rsp_file = rsp_file[0]
            if isinstance(rsp_file, BytesIO):
                rsp_file = deepcopy(rsp_file)
                self.rsp_abspath = rsp_file.name
                self.rsp_file_name = rsp_file.name
            else:
                self.rsp_abspath = os.path.abspath(rsp_file)
                self.rsp_file_name = rsp_file.split('/')[-1]
            rsp_hdu = fits.open(rsp_file, ignore_missing_simple=True)

            try:
                self.matExt = rsp_hdu["SPECRESP MATRIX"]
            except KeyError:
                self.matExt = rsp_hdu["MATRIX"]
            self.ebouExt = rsp_hdu['EBOUNDS']

            self.matHeader = self.matExt.header
            self.ebouHeader = self.ebouExt.header

            self.numEnerBins = self.matHeader['NAXIS2']
            self.numDetChans = self.ebouHeader['NAXIS2']

            self.matData = self.matExt.data
            self.ebouData = self.ebouExt.data

            rsp_hdu.close()

        elif type(rsp_file) is list and len(rsp_file) == 2:
            self.rsp_type = 'rmf&arf'

            if isinstance(rsp_file[0], BytesIO):
                rmf_file = deepcopy(rsp_file[0])
                self.rmf_abspath = rmf_file.name
                self.rmf_file_name = rmf_file.name
            else:
                rmf_file = rsp_file[0]
                self.rmf_abspath = os.path.abspath(rmf_file)
                self.rmf_file_name = rmf_file.split('/')[-1]

            if isinstance(rsp_file[1], BytesIO):
                arf_file = deepcopy(rsp_file[1])
                self.arf_abspath = arf_file.name
                self.arf_file_name = arf_file.name
            else:
                arf_file = rsp_file[1]
                self.arf_abspath = os.path.abspath(arf_file)
                self.arf_file_name = arf_file.split('/')[-1]

            rmf_hdu = fits.open(rmf_file, ignore_missing_simple=True)
            arf_hdu = fits.open(arf_file, ignore_missing_simple=True)

            try:
                self.matExt = rmf_hdu["SPECRESP MATRIX"]
            except KeyError:
                self.matExt = rmf_hdu["MATRIX"]
            self.srpExt = arf_hdu['SPECRESP']
            self.ebouExt = rmf_hdu['EBOUNDS']

            self.matHeader = self.matExt.header
            self.srpHeader = self.srpExt.header
            self.ebouHeader = self.ebouExt.header

            self.numEnerBins = self.matHeader['NAXIS2']
            self.numEnerBins_ = self.srpHeader['NAXIS2']
            self.numDetChans = self.ebouHeader['NAXIS2']
            assert self.numEnerBins == self.numEnerBins_, 'numEnerBins of rmf and arf should be same'

            self.matData = self.matExt.data
            self.ebouData = self.ebouExt.data
            self.srpData = self.srpExt.data

            rmf_hdu.close()
            arf_hdu.close()

        else:
            raise ValueError('rsp or [rmf, arf]')

        self.ChanIndex = self.ebouData.field(0).astype(int)
        self.minDetChans = self.ChanIndex[0]
        self.maxDetChans = self.ChanIndex[-1]

        self.ChanBins = list(zip(self.ebouData.field(1), self.ebouData.field(2)))
        self.ChanMin = [cbin[0] for cbin in self.ChanBins]
        self.ChanMax = [cbin[1] for cbin in self.ChanBins]
        self.ChanWidth = [cbin[1] - cbin[0] for cbin in self.ChanBins]
        self.ChanCenter = [np.sqrt(cbin[0] * cbin[1]) for cbin in self.ChanBins]

        self.EnerBins = list(zip(self.matData.field(0), self.matData.field(1)))
        self.EnerLow = [ebin[0] for ebin in self.EnerBins]
        self.EnerHigh = [ebin[1] for ebin in self.EnerBins]
        self.EnerWidth = [ebin[1] - ebin[0] for ebin in self.EnerBins]
        self.EnerCenter = [np.mean(ebin) for ebin in self.EnerBins]

        if specT is None:
            try:
                self.specT = self.matHeader['SPECT']
            except KeyError:
                self.specT = None
        else:
            self.specT = specT

        if self.matHeader['TFORM4'][0] == 'P':
            self.fchan = [fc for fc in self.matData.field(3)]
        else:
            self.fchan = [[fc] for fc in self.matData.field(3)]
        
        if self.matHeader['TFORM5'][0] == 'P':
            self.nchan = [nc for nc in self.matData.field(4)]
        else:
            self.nchan = [[nc] for nc in self.matData.field(4)]
            
        min_fchan = min([fc for fcs in self.fchan for fc in fcs])
        self.chan_off = int(min_fchan - self.minDetChans)

        # self.fchan = self.matData.field(3).astype(int)
        # self.nchan = self.matData.field(4).astype(int)

        self.matrix = self.matData.field(5)

        try:
            self.specresp = self.srpData['SPECRESP'].astype(float)
        except AttributeError:
            self.specresp = np.ones(self.numEnerBins)

        self.drm = np.zeros([self.numEnerBins, self.numDetChans])
        self.construct_drm()

        self.Eval_Energy = []
        self.Eval_Level = [0]
        self.Eval_dE = []
        self.create_eval_energy()


    # def construct_drm(self):
    #     for fc, nc, i in zip(self.fchan, self.nchan, range(self.numEnerBins)):
    #         assert (fc + nc -1) <= self.numDetChans, \
    #             'The index of channel will overflow!'
    #         if fc == 0:
    #             index = [False] * len(self.ChanIndex); index[fc: fc + nc] = [True] * nc
    #         else:
    #             index = [False] * len(self.ChanIndex); index[fc - 1: fc + nc - 1] = [True] * nc

    #         self.drm[i, index] = self.matrix[i][0 : nc]
    #     self.drm = np.float128(self.drm) * self.specresp.reshape([-1, 1])


    def construct_drm(self):
        for fc, nc, i in zip(self.fchan, self.nchan, range(self.numEnerBins)):
            idx = []
            for fc_i, nc_i in zip(fc, nc):
                fc_i = int(fc_i - self.chan_off)
                nc_i = int(nc_i)
                tc_i = fc_i + nc_i
                
                idx_i = np.where((self.ChanIndex >= fc_i) & (self.ChanIndex < tc_i))[0].tolist()
                idx = idx + idx_i

            self.drm[i, idx] = self.matrix[i][:]
        self.drm = np.float128(self.drm) * self.specresp.reshape([-1, 1])


    def qualifying(self, Qual, Notc):
        # quality flag:
        # Qual = 0 if the data quality is good
        # Qual != 0 if the data quality is bad
        self.Qual = np.array(Qual)
        self.qualified_id = list(np.array(self.Qual) == 0)
        if (self.Qual == 0).all():
            self.qualified = [[min(self.ChanMin), max(self.ChanMax)]]
        else:
            self.qualified = []
            for i, flag in enumerate(self.Qual):
                if flag == 0:
                    self.qualified.append(list(self.ChanBins[i]))
            self.qualified = union(self.qualified)

        self.Notc = Notc
        if self.Notc is None:
            self.notice = [[min(self.ChanMin), max(self.ChanMax)]]
        elif type(self.Notc[0]) is not list:
            self.notice = [self.Notc]
        else:
            self.notice = self.Notc

        self.notice_id = [False] * len(self.ChanIndex)
        for i, (low, upp) in enumerate(self.notice):
            notice_id_i = list(map(lambda l, u: low <= l and upp >= u, self.ChanMin, self.ChanMax))
            self.notice_id = [pre or now for pre, now in zip(self.notice_id, notice_id_i)]

        self.Qualified_Notice = intersection(self.qualified, self.notice)
        self.Qualified_Notice_ID = list(np.array(self.qualified_id) & np.array(self.notice_id))


    def grouping(self, Grpg):
        # grouping flag:
        # Grpg = 0 if the channel is not allowed to group, including the not qualified noticed channels
        # Grpg = +1 if the channel is the start of a new bin
        # Grpg = -1 if the channel is part of a continuing bin
        self.Grpg = np.array(Grpg)
        if (self.Grpg == 0).all():
            pass
        else:
            NewChIdx = 0
            self.New_ChanIndex = []
            self.New_ChanBins = []
            self.New_Drm = []
            self.New_Qualified_Notice_ID = []
            for i, flag in enumerate(self.Grpg):
                if (NewChIdx == 0 and flag == -1) or (flag == 0):
                    continue
                if flag == 1:
                    NewChIdx += 1
                    self.New_ChanIndex.append(NewChIdx)
                    self.New_ChanBins.append(list(self.ChanBins[i]))
                    self.New_Drm.append(self.drm[:, i])
                    self.New_Qualified_Notice_ID.append([self.Qualified_Notice_ID[i]])
                elif flag == -1:
                    self.New_ChanBins[-1][-1] = self.ChanBins[i][-1]
                    self.New_Drm[-1] += self.drm[:, i]
                    self.New_Qualified_Notice_ID[-1] += [self.Qualified_Notice_ID[i]]
            self.ChanIndex = np.array(self.New_ChanIndex).astype(int)
            self.minDetChans = self.ChanIndex[0]
            self.maxDetChans = self.ChanIndex[-1]

            self.ChanBins = self.New_ChanBins
            self.ChanMin = [cbin[0] for cbin in self.ChanBins]
            self.ChanMax = [cbin[1] for cbin in self.ChanBins]
            self.ChanWidth = [cbin[1] - cbin[0] for cbin in self.ChanBins]
            self.ChanCenter = [np.sqrt(cbin[0] * cbin[1]) for cbin in self.ChanBins]

            self.drm = np.float128(np.column_stack(self.New_Drm))
            self.Qualified_Notice_ID = [np.all(ids) for ids in self.New_Qualified_Notice_ID]


    @property
    def info(self):
        print('+-----------------------------------------------+')
        print(' rsp type: %s' % self.rsp_type)
        print(' There are %d channels from %d to %d.' % (self.numDetChans, self.minDetChans, self.maxDetChans))
        print(' There are %d enery bins.' % self.numEnerBins)
        print('+-----------------------------------------------+\n')


    def create_eval_energy(self):
        resFrac511 = 0.2  # lifted from BATSE!
        resExp = -0.15  # lifted from BATSE!

        self.EnerCenter = np.array(self.EnerCenter, dtype=np.float128)
        self.EnerWidth = np.array(self.EnerWidth, dtype=np.float128)

        resFrac = resFrac511 * (self.EnerCenter / 511.) ** resExp
        resFWHM = self.EnerCenter * resFrac

        low_eval_index = self.EnerWidth < resFWHM / 2.
        med_eval_index = (self.EnerWidth >= resFWHM / 2.) * (self.EnerWidth / 2. < resFWHM / 3.)
        high_eval_index = self.EnerWidth / 2. >= resFWHM / 3.

        for i in range(self.numEnerBins):
            x = self.EnerCenter[i]
            y = self.EnerWidth[i]

            if low_eval_index[i]:
                self.Eval_Energy.extend([x])
                self.Eval_Level.append(1)
                self.Eval_dE.extend([[0.5 * y, 0.5 * y]])
            elif med_eval_index[i]:
                self.Eval_Energy.extend([x - 0.33 * y, x , x + 0.33 * y])
                self.Eval_Level.append(3)
                self.Eval_dE.extend([[0.17 * y, 0.83 * y], [0.5 * y, 0.5 * y], [0.83 * y, 0.17 * y]])
            elif high_eval_index[i]:
                self.Eval_Energy.extend([x - 0.5 * y, x - 0.33 * y, x - 0.17 * y, x,
                                         x + 0.17 * y, x + 0.33 * y, x + 0.5 * y])
                self.Eval_Level.append(7)
                self.Eval_dE.extend([[0, y], [0.17 * y, 0.83 * y], [0.33 * y, 0.67 * y], [0.5 * y, 0.5 * y],
                                          [0.67 * y, 0.33 * y], [0.83 * y, 0.17 * y], [y, 0]])

        self.Eval_Energy = np.array(self.Eval_Energy, dtype=np.float128)
        self.Eval_dE = np.array(self.Eval_dE, dtype=np.float128)
        self.Eval_Start = np.cumsum(self.Eval_Level)[:-1]
        self.Eval_Stop = np.cumsum(self.Eval_Level)[1:]
