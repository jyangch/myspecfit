import os
import re
import json
import warnings
import numpy as np
import matplotlib as mpl
from copy import deepcopy
from Analyse import Analyse
from itertools import chain
import plotly.express as px
from Tools import JsonEncoder
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plotly.subplots import make_subplots
from Tools import oper_model, savetxt, ppsig, pgsig


class Plot(object):

    def __init__(self, aobj):
        assert isinstance(aobj, Analyse)
        self.aobj = aobj
        self.fobj = self.aobj.fobj
        self.path = self.fobj.path
        self.rebin_dict = None

        self.savepath = '%s/Plot/' % self.path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)


    @staticmethod
    def rebin_single(s, b, berr, m, ts, tb, ec, min_sigma=None, max_bin=None, stat=None):
        assert len(s) == len(b)
        assert len(m) == len(s)
        assert len(ec) == len(b)
        assert len(berr) == len(b)

        bvar = np.array(berr) ** 2

        if min_sigma == 0 and max_bin == 1:
            news, newb, newbvar, newm, newec = s, b, bvar, m, ec

        else:
            if min_sigma is None:
                news, newb, newbvar, newm, newec = [], [], [], [], []
                if max_bin is None:
                    news, newb, newbvar, newm, newec = s, b, bvar, m, ec

                else:
                    cp, j, k = 0, 0, 0
                    for i in range(len(s)):
                        cp += 1
                        if i == len(s) - 1 and cp < max_bin:
                            if k >= 1:
                                news[k-1] += np.sum(s[j:])
                                newb[k-1] += np.sum(b[j:])
                                newbvar[k-1] += np.sum(bvar[j:])
                                newm[k-1] += np.sum(m[j:])
                                newec[k-1][2] = ec[-1][2]
                                newec[k - 1][3] += np.sum(ec[n][3] for n in range(j, i + 1))
                            else:
                                news.append(np.sum(s))
                                newb.append(np.sum(b))
                                newbvar.append(np.sum(bvar))
                                newm.append(np.sum(m))
                                newec.append([1, ec[0][1], ec[-1][2], np.sum([ec[n][3] for n in range(len(s))])])

                        if cp == max_bin:
                            news.append(np.sum(s[j:i+1]))
                            newb.append(np.sum(b[j:i+1]))
                            newbvar.append(np.sum(bvar[j:i+1]))
                            newm.append(np.sum(m[j:i+1]))
                            newec.append([k+1, ec[j][1], ec[i][2], np.sum(ec[n][3] for n in range(j, i+1))])

                            cp = 0
                            j = i + 1
                            k += 1

            else:
                news, newb, newbvar, newm, newec = [], [], [], [], []
                sp_, bp, bpvar, cp, j, k = 0, 0, 0, 0, 0, 0
                for i in range(len(s)):
                    si = s[i]
                    bi = b[i]
                    bivar = bvar[i]
                    sp_ += si
                    bp += bi
                    bpvar += bivar
                    cp += 1

                    if stat == 'cstat':
                        if (bp < 0 or sp_ < 0) and (bp != sp_):
                            sigma = 0
                        else:
                            sigma = ppsig(sp_, bp, ts / tb)
                    elif stat == 'pgstat' or stat == 'pgfstat':
                        if sp_ <= 0:
                            sigma = 0
                        elif np.sqrt(bpvar) == 0:
                            sigma = 0
                        else:
                            sigma = pgsig(sp_, bp, np.sqrt(bpvar))
                    else:
                        raise KeyError('It is unavailable kind of stat!')

                    if max_bin is None:
                        if i == len(s) - 1 and sigma < min_sigma:
                            if k >= 1:
                                news[k-1] += np.sum(s[j:])
                                newb[k-1] += np.sum(b[j:])
                                newbvar[k-1] += np.sum(bvar[j:])
                                newm[k-1] += np.sum(m[j:])
                                newec[k-1][2] = ec[-1][2]
                                newec[k - 1][3] += np.sum([ec[n][3] for n in range(j, i+1)])
                            else:
                                news.append(np.sum(s))
                                newb.append(np.sum(b))
                                newbvar.append(np.sum(bvar))
                                newm.append(np.sum(m))
                                newec.append([1, ec[0][1], ec[-1][2], np.sum([ec[n][3] for n in range(len(s))])])

                        if sigma >= min_sigma:
                            news.append(np.sum(s[j:i+1]))
                            newb.append(np.sum(b[j:i+1]))
                            newbvar.append(np.sum(bvar[j:i+1]))
                            newm.append(np.sum(m[j:i+1]))
                            newec.append([k+1, ec[j][1], ec[i][2], np.sum([ec[n][3] for n in range(j, i+1)])])

                            sp_, bp, bpvar, cp = 0, 0, 0, 0
                            j = i + 1
                            k += 1

                    else:
                        if i == len(s) - 1 and sigma < min_sigma and cp < max_bin:
                            if k >= 1:
                                news[k-1] += np.sum(s[j:])
                                newb[k-1] += np.sum(b[j:])
                                newbvar[k-1] += np.sum(bvar[j:])
                                newm[k-1] += np.sum(m[j:])
                                newec[k-1][2] = ec[-1][2]
                                newec[k-1][3] += np.sum([ec[n][3] for n in range(j, i+1)])
                            else:
                                news.append(np.sum(s))
                                newb.append(np.sum(b))
                                newbvar.append(np.sum(bvar))
                                newm.append(np.sum(m))
                                newec.append([1, ec[0][1], ec[-1][2], np.sum([ec[n][3] for n in range(len(s))])])

                        if sigma >= min_sigma or cp == max_bin:
                            news.append(np.sum(s[j:i+1]))
                            newb.append(np.sum(b[j:i+1]))
                            newbvar.append(np.sum(bvar[j:i+1]))
                            newm.append(np.sum(m[j:i+1]))
                            newec.append([k+1, ec[j][1], ec[i][2], np.sum([ec[n][3] for n in range(j, i+1)])])

                            sp_, bp, bpvar, cp = 0, 0, 0, 0
                            j = i + 1
                            k += 1

        news, newb, newberr, newm = np.array(news), np.array(newb), np.sqrt(newbvar), np.array(newm)
        assert abs(np.sum(news) - np.sum(s)) < 0.01, warnings.warn('sum of s is different!')
        assert abs(np.sum(newb) - np.sum(b)) < 0.01, warnings.warn('sum of b is different!')
        assert abs(np.sum(newbvar) - np.sum(bvar)) < 0.01, warnings.warn('sum of bvar is different!')
        assert abs(np.sum(newm) - np.sum(m)) < 0.01, warnings.warn('sum of m is different!')
        return news, newb, newberr, newm, newec


    def rebin(self, rebin_dict=None, save=True):
        self.nspec = self.fobj.nspec
        self.spec_exprs = self.fobj.spec_exprs
        self.NchIndex = self.fobj.NchIndex
        self.stat_exprs = self.fobj.stat_exprs

        self.seff = self.fobj.src_eff
        self.beff = self.fobj.bkg_eff

        self.schCounts = self.fobj.src_NchCounts
        self.bchCounts = self.fobj.bkg_NchCounts
        self.schErr = self.fobj.src_NchErr
        self.bchErr = self.fobj.bkg_NchErr
        self.mchCounts = list(map(lambda x, y: x * y, self.fobj.mo_NchRate, self.seff))

        self.chIndex_ = self.fobj.rsp_chIndex
        self.chMin_ = self.fobj.rsp_chMin
        self.chMax_ = self.fobj.rsp_chMax
        self.chWidth_ = self.fobj.rsp_chWidth
        self.chCenter_ = self.fobj.rsp_chCenter

        self.chIndex = [ch[index] for ch, index in zip(self.chIndex_, self.NchIndex)]
        self.chMin = [ch[index] for ch, index in zip(self.chMin_, self.NchIndex)]
        self.chMax = [ch[index] for ch, index in zip(self.chMax_, self.NchIndex)]
        self.chWidth = [ch[index] for ch, index in zip(self.chWidth_, self.NchIndex)]
        self.chCenter = [ch[index] for ch, index in zip(self.chCenter_, self.NchIndex)]

        self.schRate = list(map(lambda x, y: x / y, self.schCounts, self.seff))
        self.bchRate = list(map(lambda x, y: x / y, self.bchCounts, self.beff))
        self.chNetRate = list(map(lambda x, y: x - y, self.schRate, self.bchRate))
        self.chNetRateErr = list(map(lambda se, be, ts, tb: np.sqrt((se/ts)**2 + (be/tb)**2),
                                     self.schErr, self.bchErr, self.seff, self.beff))
        self.mchRate = self.fobj.mo_NchRate

        self.schCE = list(map(lambda x, y: x / y, self.schRate, self.chWidth))
        self.bchCE = list(map(lambda x, y: x / y, self.bchRate, self.chWidth))
        self.chNetCE = list(map(lambda x, y: x - y, self.schCE, self.bchCE))
        self.mchCE = list(map(lambda x, y: x / y, self.mchRate, self.chWidth))

        self.schReCounts, self.bchReCounts, self.bchReErr, self.mchReCounts = [], [], [], []
        self.chReIndex, self.chReMin, self.chReMax = [], [], []
        self.chReWidth, self.chReCenter, self.chReBins = [], [], []

        self.rebin_dict = {}
        for spex, stex in zip(self.spec_exprs, self.stat_exprs):
            if stex == 'pgstat':
                self.rebin_dict[spex] = {'min_sigma': 3, 'max_bin': 10}
            else:
                self.rebin_dict[spex] = {'min_sigma': 0, 'max_bin': 1}
        if rebin_dict is not None: self.rebin_dict.update(rebin_dict)

        for i in range(self.nspec):
            s, b, berr, m = self.schCounts[i], self.bchCounts[i], self.bchErr[i], self.mchCounts[i]
            ts_, tb_ = self.seff[i], self.beff[i]
            ec = list(zip(self.chIndex[i], self.chMin[i], self.chMax[i], self.chWidth[i]))

            min_sigma = self.rebin_dict[self.spec_exprs[i]]['min_sigma']
            max_bin = self.rebin_dict[self.spec_exprs[i]]['max_bin']

            news, newb, newberr, newm, newec = self.rebin_single(s, b, berr, m, ts_, tb_, ec, min_sigma=min_sigma,
                                                                 max_bin=max_bin, stat=self.stat_exprs[i])

            self.schReCounts.append(news)
            self.bchReCounts.append(newb)
            self.bchReErr.append(newberr)
            self.mchReCounts.append(newm)

            self.chReIndex.append([cbin[0] for cbin in newec])
            self.chReMin.append(np.array([cbin[1] for cbin in newec]))
            self.chReMax.append(np.array([cbin[2] for cbin in newec]))
            self.chReWidth.append(np.array([cbin[3] for cbin in newec]))
            self.chReCenter.append(np.array([np.sqrt(cbin[1] * cbin[2]) for cbin in newec]))
            self.chReBins.append(np.array([[cbin[1], cbin[2]] for cbin in newec]))

        self.schReRate = list(map(lambda x, y: x / y, self.schReCounts, self.seff))
        self.bchReRate = list(map(lambda x, y: x / y, self.bchReCounts, self.beff))
        self.mchReRate = list(map(lambda x, y: x / y, self.mchReCounts, self.seff))

        self.schReCE = list(map(lambda x, y: x / y, self.schReRate, self.chReWidth))
        self.bchReCE = list(map(lambda x, y: x / y, self.bchReRate, self.chReWidth))
        self.chReNetCE = list(map(lambda x, y: x - y, self.schReCE, self.bchReCE))
        self.mchReCE = list(map(lambda x, y: x / y, self.mchReRate, self.chReWidth))

        err_type = 'sqrt'
        if err_type == 'sqrt':
            self.schReErr = list(map(np.sqrt, self.schReCounts))
        elif err_type == 'poiss-1':
            self.schReErr = list(map(lambda x: 1 + np.sqrt(x + 0.75), self.schReCounts))
        elif err_type == 'poiss-2':
            self.schReErr = list(map(lambda x: np.sqrt(x - 0.25), self.schReCounts))
        elif err_type == 'poiss-3':
            self.schReErr = list(map(lambda x: (1+np.sqrt(x+0.75)+np.sqrt(x-0.25))/2, self.schReCounts))
        self.chReNetCEErr = list(map(lambda se, be, ts, tb, w: np.sqrt((se/ts)**2 + (be/tb)**2)/w, self.schReErr,
                                     self.bchReErr, self.seff, self.beff, self.chReWidth))
        self.scalRes = list(map(lambda ni, mi, nei: (ni - mi) / nei, self.chReNetCE, self.mchReCE, self.chReNetCEErr))

        if save:
            # ----- save data -----
            json.dump(self.spec_exprs, open(self.savepath + 'spec_exprs.json', 'w'), indent=4, cls=JsonEncoder)

            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.schCounts)}, 
                    open(self.savepath + 'schCounts.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchCounts)}, 
                    open(self.savepath + 'bchCounts.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchErr)}, 
                    open(self.savepath + 'bchErr.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.mchCounts)}, 
                    open(self.savepath + 'mchCounts.json', 'w'), indent=4, cls=JsonEncoder)
            
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.seff)}, 
                    open(self.savepath + 'seff.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.beff)}, 
                    open(self.savepath + 'beff.json', 'w'), indent=4, cls=JsonEncoder)
            
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.schRate)}, 
                    open(self.savepath + 'schRate.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchRate)}, 
                    open(self.savepath + 'bchRate.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chNetRate)}, 
                    open(self.savepath + 'chNetRate.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chNetRateErr)}, 
                    open(self.savepath + 'chNetRateErr.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.mchRate)}, 
                    open(self.savepath + 'mchRate.json', 'w'), indent=4, cls=JsonEncoder)

            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.NchIndex)}, 
                    open(self.savepath + 'NchIndex.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chIndex)}, 
                    open(self.savepath + 'chIndex.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chMin)}, 
                    open(self.savepath + 'chMin.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chMax)}, 
                    open(self.savepath + 'chMax.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chWidth)}, 
                    open(self.savepath + 'chWidth.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chCenter)}, 
                    open(self.savepath + 'chCenter.json', 'w'), indent=4, cls=JsonEncoder)
            
            json.dump(self.rebin_dict, open(self.savepath + 'rebin_dict.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.stat_exprs, self.stat_exprs)}, 
                    open(self.savepath + 'stat_exprs.json', 'w'), indent=4, cls=JsonEncoder)

            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.schReCounts)}, 
                    open(self.savepath + 'schReCounts.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchReCounts)}, 
                    open(self.savepath + 'bchReCounts.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchReErr)}, 
                    open(self.savepath + 'bchReErr.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.mchReCounts)}, 
                    open(self.savepath + 'mchReCounts.json', 'w'), indent=4, cls=JsonEncoder)

            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReIndex)}, 
                    open(self.savepath + 'chReIndex.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReMin)}, 
                    open(self.savepath + 'chReMin.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReMax)}, 
                    open(self.savepath + 'chReMax.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReWidth)}, 
                    open(self.savepath + 'chReWidth.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReCenter)}, 
                    open(self.savepath + 'chReCenter.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReBins)}, 
                    open(self.savepath + 'chReBins.json', 'w'), indent=4, cls=JsonEncoder)

            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.schReRate)}, 
                    open(self.savepath + 'schReRate.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchReRate)}, 
                    open(self.savepath + 'bchReRate.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.mchReRate)}, 
                    open(self.savepath + 'mchReRate.json', 'w'), indent=4, cls=JsonEncoder)

            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.schReCE)}, 
                    open(self.savepath + 'schReCE.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.bchReCE)}, 
                    open(self.savepath + 'bchReCE.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReNetCE)}, 
                    open(self.savepath + 'chReNetCE.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.mchReCE)}, 
                    open(self.savepath + 'mchReCE.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.schReErr)}, 
                    open(self.savepath + 'schReErr.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chReNetCEErr)}, 
                    open(self.savepath + 'chReNetCEErr.json', 'w'), indent=4, cls=JsonEncoder)
            json.dump({ex: val for ex, val in zip(self.spec_exprs, self.scalRes)}, 
                    open(self.savepath + 'scalRes.json', 'w'), indent=4, cls=JsonEncoder)
            
            savetxt(file=self.savepath + 'spec_exprs.txt', data=[self.spec_exprs], trans=True)
            savetxt(file=self.savepath + 'chCenter.txt', data=self.chCenter, trans=True)
            savetxt(file=self.savepath + 'chNetRate.txt', data=self.chNetRate, trans=True)
            savetxt(file=self.savepath + 'mchRate.txt', data=self.mchRate, trans=True)
            savetxt(file=self.savepath + 'chReCenter.txt', data=self.chReCenter, trans=True)
            savetxt(file=self.savepath + 'chReNetCE.txt', data=self.chReNetCE, trans=True)
            savetxt(file=self.savepath + 'chReNetCEErr.txt', data=self.chReNetCEErr, trans=True)
            savetxt(file=self.savepath + 'chReMin.txt', data=self.chReMin, trans=True)
            savetxt(file=self.savepath + 'chReMax.txt', data=self.chReMax, trans=True)
            savetxt(file=self.savepath + 'mchReCE.txt', data=self.mchReCE, trans=True)
            savetxt(file=self.savepath + 'scalRes.txt', data=self.scalRes, trans=True)
            # ----- save data -----


    def cspec(self, spec_exprs='all', each=True, ploter='matplotlib'):
        if self.rebin_dict is None: self.rebin()
        self.spec_exprs = self.fobj.spec_exprs
        self.spec_enum = {sex:i for i, sex in enumerate(self.spec_exprs)}

        if spec_exprs == 'all':
            spec_exprs = self.spec_exprs
        
        if len(spec_exprs) <= 10:
            self.colors = dict(zip(spec_exprs, px.colors.qualitative.Plotly))
        elif 10 < len(spec_exprs) <= 24:
            self.colors = dict(zip(spec_exprs, px.colors.qualitative.Dark24))
        else:
            self.colors = dict(zip(spec_exprs, mpl.colormaps['rainbow'](np.linspace(0, 1, len(spec_exprs)))))

        if ploter == 'plotly':
            self.cspec_fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25], 
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0)

            for expr in spec_exprs:
                if expr in self.spec_enum: i = self.spec_enum[expr]
                else: warnings.warn('%s is not one of spectrum group names' % expr); continue
                obs = go.Scatter(x=self.chReCenter[i].astype(float), 
                                y=self.chReNetCE[i].astype(float), 
                                mode='markers', 
                                name='%s obs' % expr, 
                                showlegend=False, 
                                error_x=dict(
                                    type='data',
                                    symmetric=False,
                                    array=(self.chReMax[i] - self.chReCenter[i]).astype(float),
                                    arrayminus=(self.chReCenter[i] - self.chReMin[i]).astype(float),
                                    color=self.colors[expr],
                                    thickness=1.5,
                                    width=0),
                                error_y=dict(
                                    type='data',
                                    array=self.chReNetCEErr[i].astype(float),
                                    color=self.colors[expr],
                                    thickness=1.5,
                                    width=0),
                                marker=dict(symbol='cross-thin', size=0, color=self.colors[expr]))
                self.cspec_fig.add_trace(obs, row=1, col=1)

                mo = go.Scatter(x=self.chReCenter[i].astype(float), 
                                y=self.mchReCE[i].astype(float), 
                                name=expr, 
                                showlegend=True, 
                                mode='lines', 
                                line=dict(width=2, color=self.colors[expr]))
                self.cspec_fig.add_trace(mo, row=1, col=1)

                res = go.Scatter(x=self.chReCenter[i].astype(float), 
                                y=self.scalRes[i].astype(float), 
                                name='%s res' % expr, 
                                showlegend=False, 
                                mode='markers', 
                                marker=dict(symbol='cross-thin', size=10, color=self.colors[expr], 
                                            line=dict(width=1.5, color=self.colors[expr])))
                self.cspec_fig.add_trace(res, row=2, col=1)
                
            self.cspec_fig.update_xaxes(title_text='', row=1, col=1, type='log')
            self.cspec_fig.update_xaxes(title_text='Energy (keV)', row=2, col=1, type='log')
            self.cspec_fig.update_yaxes(title_text='C(E) (counts/s/keV)', row=1, col=1, type='log')
            self.cspec_fig.update_yaxes(title_text='Residuals', showgrid=False, range=[-3.5, 3.5], row=2, col=1)

            self.cspec_fig.update_layout(height=800, width=900)

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            # rcParams['font.sans-serif'] = 'Arial'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)

            ax1 = fig.add_subplot(gs[0:3, 0])
            for expr in spec_exprs:
                if expr in self.spec_enum: i = self.spec_enum[expr]
                else: warnings.warn('%s is not one of spectrum group names' % expr); continue
                ax1.errorbar(self.chCenter[i], self.chNetRate[i], yerr=self.chNetRateErr[i], fmt='none',
                            xerr=[self.chCenter[i] - self.chMin[i], self.chMax[i] - self.chCenter[i]],
                            ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr)
                ax1.plot(self.chCenter[i], self.mchRate[i], color=self.colors[expr], lw=1.0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylabel(r'$\rm{cts/s/ch}$')
            ax1.minorticks_on()
            ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(which='major', width=1.0, length=5)
            ax1.tick_params(which='minor', width=1.0, length=3)
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.spines['bottom'].set_linewidth(1.0)
            ax1.spines['top'].set_linewidth(1.0)
            ax1.spines['left'].set_linewidth(1.0)
            ax1.spines['right'].set_linewidth(1.0)
            ax1.legend(frameon=True)

            ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
            for expr in spec_exprs:
                if expr in self.spec_enum: i = self.spec_enum[expr]
                else: warnings.warn('%s is not one of spectrum group names' % expr); continue
                ax2.scatter(self.chCenter[i], self.chNetRate[i] - self.mchRate[i], marker='+',
                            color=self.colors[expr], s=40, linewidths=0.8)
            ax2.axhline(0, c='grey', lw=1, ls='--')
            ax2.set_xlabel('Energy (keV)')
            ax2.set_ylabel('Residuals')
            ax2.set_ylim([-20, 20])
            ax2.minorticks_on()
            ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(which='major', width=1.0, length=5)
            ax2.tick_params(which='minor', width=1.0, length=3)
            ax2.yaxis.set_ticks_position('both')
            ax2.spines['bottom'].set_linewidth(1.0)
            ax2.spines['top'].set_linewidth(1.0)
            ax2.spines['left'].set_linewidth(1.0)
            ax2.spines['right'].set_linewidth(1.0)

            plt.savefig(self.savepath + 'cc_spec.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            print('+-----------------------------------------------+')
            print(' plotted CC spectrum')
            print('+-----------------------------------------------+\n')

            if each:
                for expr in spec_exprs:
                    if expr in self.spec_enum: i = self.spec_enum[expr]
                    else: warnings.warn('%s is not one of spectrum group names' % expr); continue

                    fig = plt.figure(figsize=(6, 6))
                    gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)

                    ax1 = fig.add_subplot(gs[0:3, 0])
                    ax1.errorbar(self.chCenter[i], self.chNetRate[i], yerr=self.chNetRateErr[i], fmt='none',
                                xerr=[self.chCenter[i] - self.chMin[i], self.chMax[i] - self.chCenter[i]],
                                ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr)
                    ax1.plot(self.chCenter[i], self.mchRate[i], color=self.colors[expr], lw=1.0)
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax1.set_ylabel(r'$\rm{cts/s/ch}$')
                    ax1.minorticks_on()
                    ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
                    ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
                    ax1.tick_params(which='major', width=1.0, length=5)
                    ax1.tick_params(which='minor', width=1.0, length=3)
                    ax1.xaxis.set_ticks_position('both')
                    ax1.yaxis.set_ticks_position('both')
                    plt.setp(ax1.get_xticklabels(), visible=False)
                    ax1.spines['bottom'].set_linewidth(1.0)
                    ax1.spines['top'].set_linewidth(1.0)
                    ax1.spines['left'].set_linewidth(1.0)
                    ax1.spines['right'].set_linewidth(1.0)
                    ax1.legend(frameon=True)

                    ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
                    ax2.scatter(self.chCenter[i], self.chNetRate[i] - self.mchRate[i], marker='+',
                                color=self.colors[expr], s=40, linewidths=0.8)
                    ax2.axhline(0, c='grey', lw=1, ls='--')
                    ax2.set_xlabel('Energy (keV)')
                    ax2.set_ylabel('Residuals')
                    ax2.set_ylim([-20, 20])
                    ax2.minorticks_on()
                    ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
                    ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
                    ax2.tick_params(which='major', width=1.0, length=5)
                    ax2.tick_params(which='minor', width=1.0, length=3)
                    ax2.yaxis.set_ticks_position('both')
                    ax2.spines['bottom'].set_linewidth(1.0)
                    ax2.spines['top'].set_linewidth(1.0)
                    ax2.spines['left'].set_linewidth(1.0)
                    ax2.spines['right'].set_linewidth(1.0)

                    plt.savefig(self.savepath + 'cc_spec@%s.pdf'%expr, bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close(fig)

                print('+-----------------------------------------------+')
                print(' plotted CC spectrum for each spec')
                print('+-----------------------------------------------+\n')

            fig = plt.figure(figsize=(6, 8))
            gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)

            ax1 = fig.add_subplot(gs[0:3, 0])
            for expr in spec_exprs:
                if expr in self.spec_enum: i = self.spec_enum[expr]
                else: warnings.warn('%s is not one of spectrum group names' % expr); continue
                ax1.errorbar(self.chReCenter[i], self.chReNetCE[i], yerr=self.chReNetCEErr[i], fmt='none',
                            xerr=[self.chReCenter[i] - self.chReMin[i], self.chReMax[i] - self.chReCenter[i]],
                            ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr, zorder=1)
                ax1.plot(self.chReCenter[i], self.mchReCE[i], color=self.colors[expr], lw=1.0, zorder=2)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylabel(r'$C_E~(\rm{Counts~s^{-1}~keV^{-1}})$')
            ax1.set_ylim(ymin=6e-4)
            ax1.minorticks_on()
            ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(which='major', width=1.0, length=5)
            ax1.tick_params(which='minor', width=1.0, length=3)
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.spines['bottom'].set_linewidth(1.0)
            ax1.spines['top'].set_linewidth(1.0)
            ax1.spines['left'].set_linewidth(1.0)
            ax1.spines['right'].set_linewidth(1.0)
            ax1.legend(frameon=True)

            ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
            for expr in spec_exprs:
                if expr in self.spec_enum: i = self.spec_enum[expr]
                else: warnings.warn('%s is not one of spectrum group names' % expr); continue
                ax2.scatter(self.chReCenter[i], self.scalRes[i], marker='+',
                            color=self.colors[expr], s=40, linewidths=0.8, zorder=2)
            ax2.axhline(0, color='grey', lw=1, ls='--', zorder=1)
            ax2.set_ylim([-3.5, 3.5])
            ax2.set_xlabel('Energy (keV)')
            ax2.set_ylabel('Residuals')
            ax2.minorticks_on()
            ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(which='major', width=1.0, length=5)
            ax2.tick_params(which='minor', width=1.0, length=3)
            ax2.yaxis.set_ticks_position('both')
            ax2.spines['bottom'].set_linewidth(1.0)
            ax2.spines['top'].set_linewidth(1.0)
            ax2.spines['left'].set_linewidth(1.0)
            ax2.spines['right'].set_linewidth(1.0)

            plt.savefig(self.savepath + 'ce_spec.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            print('+-----------------------------------------------+')
            print(' plotted CE spectrum')
            print('+-----------------------------------------------+\n')

            if each:
                for expr in spec_exprs:
                    if expr in self.spec_enum: i = self.spec_enum[expr]
                    else: warnings.warn('%s is not one of spectrum group names' % expr); continue

                    fig = plt.figure(figsize=(6, 8))
                    gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)

                    ax1 = fig.add_subplot(gs[0:3, 0])
                    ax1.errorbar(self.chReCenter[i], self.chReNetCE[i], yerr=self.chReNetCEErr[i], fmt='none',
                                xerr=[self.chReCenter[i] - self.chReMin[i], self.chReMax[i] - self.chReCenter[i]],
                                ecolor=self.colors[expr], elinewidth=0.8, capsize=0, capthick=0, label=expr, zorder=2)
                    ax1.plot(self.chReCenter[i], self.mchReCE[i], color=self.colors[expr], lw=1.0, zorder=1)
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax1.set_ylabel(r'$C_E~(\rm{Counts~s^{-1}~keV^{-1}})$')
                    ax1.set_ylim(ymin=6e-4)
                    ax1.minorticks_on()
                    ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
                    ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
                    ax1.tick_params(which='major', width=1.0, length=5)
                    ax1.tick_params(which='minor', width=1.0, length=3)
                    ax1.xaxis.set_ticks_position('both')
                    ax1.yaxis.set_ticks_position('both')
                    plt.setp(ax1.get_xticklabels(), visible=False)
                    ax1.spines['bottom'].set_linewidth(1.0)
                    ax1.spines['top'].set_linewidth(1.0)
                    ax1.spines['left'].set_linewidth(1.0)
                    ax1.spines['right'].set_linewidth(1.0)
                    ax1.legend(frameon=True)

                    ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
                    ax2.scatter(self.chReCenter[i], self.scalRes[i], marker='+', color=self.colors[expr],
                                s=40, linewidths=0.8, zorder=2)
                    ax2.axhline(0, color='grey', lw=1, ls='--', zorder=1)
                    ax2.set_ylim([-3.5, 3.5])
                    ax2.set_xlabel('Energy (keV)')
                    ax2.set_ylabel('Residuals')
                    ax2.minorticks_on()
                    ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
                    ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
                    ax2.tick_params(which='major', width=1.0, length=5)
                    ax2.tick_params(which='minor', width=1.0, length=3)
                    ax2.yaxis.set_ticks_position('both')
                    ax2.spines['bottom'].set_linewidth(1.0)
                    ax2.spines['top'].set_linewidth(1.0)
                    ax2.spines['left'].set_linewidth(1.0)
                    ax2.spines['right'].set_linewidth(1.0)

                    plt.savefig(self.savepath + 'ce_spec@%s.pdf'%expr, bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close(fig)

                print('+-----------------------------------------------+')
                print(' plotted CE spectrum for each spec')
                print('+-----------------------------------------------+\n')


    def pspec(self, spec_exprs='all', each=True, ploter='matplotlib'):
        self.nspecs = self.fobj.nspecs
        self.specTs = self.fobj.specTs
        self.mo_exprs = self.fobj.mo_exprs
        self.mo_funcs = self.fobj.mo_funcs
        self.spec_exprs = self.fobj.spec_exprs

        self.spec_enum = {}
        for i, j in enumerate(chain.from_iterable([i] * n for i, n in enumerate(self.nspecs))):
            sex, mex = self.spec_exprs[i], self.mo_exprs[j]
            self.spec_enum[sex] = (i, j)

        if spec_exprs == 'all':
            spec_exprs = self.spec_exprs

        log_chE_min = np.log10(min(list(chain.from_iterable(self.chReMin))))
        log_chE_max = np.log10(max(list(chain.from_iterable(self.chReMax))))
        self.specEr = np.vstack((np.logspace(log_chE_min, log_chE_max, 300)[:-1],
                                 np.logspace(log_chE_min, log_chE_max, 300)[1:])).T
        self.specE = np.array([np.sqrt(Eri[0] * Eri[1]) for Eri in self.specEr])

        self.specNE, self.specENE, self.specE2NE, self.chNE = {}, {}, {}, []

        for i, j in enumerate(chain.from_iterable([i] * n for i, n in enumerate(self.nspecs))):
            sex, mex = self.spec_exprs[i], self.mo_exprs[j]

            specT = self.specTs[i] * np.ones(len(self.specE)) if self.specTs[i] is not None else self.specTs[i]
            ne, ene, e2ne = self.mo_funcs[j](self.specEr, specT, mex)
            self.specNE[sex] = ne; self.specENE[sex] = ene; self.specE2NE[sex] = e2ne

            specT = self.specTs[i] * np.ones(len(self.chReBins[i])) if self.specTs[i] is not None else self.specTs[i]
            self.chNE.append(self.mo_funcs[j](self.chReBins[i], specT, self.mo_exprs[j])[0][self.mo_exprs[j]])

        self.chScale = [(self.chReNetCE[i] - self.mchReCE[i]) / self.mchReCE[i] for i in range(self.nspec)]
        self.pspecCE = [self.chNE[i] * self.chScale[i] + self.chNE[i] for i in range(self.nspec)]
        self.chScaleErr = [self.chReNetCEErr[i] / self.chReNetCE[i] for i in range(self.nspec)]
        self.pspecCEErr = [self.pspecCE[i] * self.chScaleErr[i] for i in range(self.nspec)]

        if len(spec_exprs) <= 10:
            self.colors = dict(zip(spec_exprs, px.colors.qualitative.Plotly))
        elif 10 < len(spec_exprs) <= 24:
            self.colors = dict(zip(spec_exprs, px.colors.qualitative.Dark24))
        else:
            self.colors = dict(zip(spec_exprs, mpl.colormaps['rainbow'](np.linspace(0, 1, len(spec_exprs)))))

        if ploter == 'plotly':
            self.pspec_fig = go.Figure()
            for sex in spec_exprs:
                if sex in self.spec_enum: i, j = self.spec_enum[sex]; mex = self.mo_exprs[j]
                else: warnings.warn('%s is not one of spectrum group names' % sex); continue

                obs = go.Scatter(x=self.chReCenter[i].astype(float), 
                                y=self.pspecCE[i].astype(float), 
                                mode='markers', 
                                name=sex, 
                                showlegend=False, 
                                error_x=dict(
                                    type='data',
                                    symmetric=False,
                                    array=(self.chReMax[i] - self.chReCenter[i]).astype(float),
                                    arrayminus=(self.chReCenter[i] - self.chReMin[i]).astype(float),
                                    color=self.colors[sex],
                                    thickness=1.5,
                                    width=0),
                                error_y=dict(
                                    type='data',
                                    array=self.pspecCEErr[i].astype(float),
                                    color=self.colors[sex],
                                    thickness=1.5,
                                    width=0),
                                marker=dict(symbol='cross-thin', size=0, color=self.colors[sex]))
                self.pspec_fig.add_trace(obs)

                mo = go.Scatter(x=self.specE.astype(float), 
                                y=self.specNE[sex][mex].astype(float), 
                                name='%s@%s'%(mex, sex), 
                                showlegend=True, 
                                mode='lines', 
                                line=dict(width=2, color=self.colors[sex]))
                self.pspec_fig.add_trace(mo)
                
            self.pspec_fig.update_xaxes(title_text='Energy (keV)', type='log')
            self.pspec_fig.update_yaxes(title_text='N(E) (photons/cm2/s/keV)', type='log')

            self.pspec_fig.update_layout(height=600, width=800)

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            # rcParams['font.sans-serif'] = 'Arial'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            for sex in spec_exprs:
                if sex in self.spec_enum: i, j = self.spec_enum[sex]; mex = self.mo_exprs[j]
                else: warnings.warn('%s is not one of spectrum group names' % sex); continue
                ax.plot(self.specE, self.specNE[sex][mex], color=self.colors[sex], lw=1.0, label='%s@%s'%(mex, sex), zorder=2)
                ax.errorbar(self.chReCenter[i], self.pspecCE[i], yerr=self.pspecCEErr[i], fmt='none',
                            xerr=[self.chReCenter[i] - self.chReMin[i], self.chReMax[i] - self.chReCenter[i]],
                            ecolor=self.colors[sex], elinewidth=0.8, capsize=0, capthick=0, zorder=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(r'$N_E~(\rm{Photons~cm^{-2}~s^{-1}~keV^{-1}})$')
            ax.set_xlim([10 ** log_chE_min, 10 ** log_chE_max])
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.legend(frameon=True)
            
            plt.savefig(self.savepath + 'ne_spec.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            self.pspec_fig = deepcopy(fig)
            plt.close(fig)

            print('+-----------------------------------------------+')
            print(' plotted NE spectrum')
            print('+-----------------------------------------------+\n')

            if each:
                for sex in spec_exprs:
                    if sex in self.spec_enum: i, j = self.spec_enum[sex]; mex = self.mo_exprs[j]
                    else: warnings.warn('%s is not one of spectrum group names' % sex); continue

                    fig = plt.figure(figsize=(6, 8))
                    gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)

                    ax1 = fig.add_subplot(gs[0:3, 0])
                    ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)

                    ax1.plot(self.specE, self.specNE[sex][mex], color=self.colors[sex], lw=1.0, label='%s@%s'%(mex, sex))
                    ax1.errorbar(self.chReCenter[i], self.pspecCE[i], yerr=self.pspecCEErr[i], fmt='none',
                                xerr=[self.chReCenter[i] - self.chReMin[i], self.chReMax[i] - self.chReCenter[i]],
                                ecolor=self.colors[sex], elinewidth=0.8, capsize=0, capthick=0)
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax1.set_ylabel(r'$N_E~(\rm{Photons~cm^{-2}~s^{-1}~keV^{-1}})$')
                    ax1.set_xlim([10 ** log_chE_min, 10 ** log_chE_max])
                    ax1.minorticks_on()
                    ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
                    ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
                    ax1.tick_params(which='major', width=1.0, length=5)
                    ax1.tick_params(which='minor', width=1.0, length=3)
                    ax1.xaxis.set_ticks_position('both')
                    ax1.yaxis.set_ticks_position('both')
                    plt.setp(ax1.get_xticklabels(), visible=False)
                    ax1.spines['bottom'].set_linewidth(1.0)
                    ax1.spines['top'].set_linewidth(1.0)
                    ax1.spines['left'].set_linewidth(1.0)
                    ax1.spines['right'].set_linewidth(1.0)
                    ax1.legend(frameon=True)

                    ax2.scatter(self.chReCenter[i], self.scalRes[i], marker='+', color=self.colors[sex], s=40, linewidths=0.8)
                    ax2.axhline(0, color='grey', lw=1, ls='--')
                    ax2.set_ylim([-3.5, 3.5])
                    ax2.set_xlabel('Energy (keV)')
                    ax2.set_ylabel('Residuals')
                    ax2.minorticks_on()
                    ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
                    ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
                    ax2.tick_params(which='major', width=1.0, length=5)
                    ax2.tick_params(which='minor', width=1.0, length=3)
                    ax2.yaxis.set_ticks_position('both')
                    ax2.spines['bottom'].set_linewidth(1.0)
                    ax2.spines['top'].set_linewidth(1.0)
                    ax2.spines['left'].set_linewidth(1.0)
                    ax2.spines['right'].set_linewidth(1.0)

                    plt.savefig(self.savepath + 'ne_spec@%s.pdf'%sex, bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close(fig)

                print('+-----------------------------------------------+')
                print(' plotted NE spectrum for each spec')
                print('+-----------------------------------------------+\n')

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            for sex in spec_exprs:
                if sex in self.spec_enum: i, j = self.spec_enum[sex]; mex = self.mo_exprs[j]
                else: warnings.warn('%s is not one of spectrum group names' % sex); continue
                ax.plot(self.specE, self.specENE[sex][mex], color=self.colors[sex], lw=1.0, label='%s@%s'%(mex, sex))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(r'$EN_E~(\rm{erg~cm^{-2}~s^{-1}~keV^{-1}})$')
            ax.set_xlim([10 ** log_chE_min, 10 ** log_chE_max])
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.legend(frameon=True)

            plt.savefig(self.savepath + 'ene_spec.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            print('+-----------------------------------------------+')
            print(' plotted ENE or Fv spectrum')
            print('+-----------------------------------------------+\n')

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            for sex in spec_exprs:
                if sex in self.spec_enum: i, j = self.spec_enum[sex]; mex = self.mo_exprs[j]
                else: warnings.warn('%s is not one of spectrum group names' % sex); continue
                ax.plot(self.specE, self.specE2NE[sex][mex], color=self.colors[sex], lw=1.0, label='%s@%s'%(mex, sex))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(r'$E^2N_E~(\rm{erg~cm^{-2}~s^{-1}})$')
            ax.set_xlim([10 ** log_chE_min, 10 ** log_chE_max])
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.legend(frameon=True)

            plt.savefig(self.savepath + 'eene_spec.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            print('+-----------------------------------------------+')
            print(' plotted E^2NE or vFv spectrum')
            print('+-----------------------------------------------+\n')

        # ----- save data -----
        np.savetxt(self.savepath + 'specE.txt', self.specE)
        np.savetxt(self.savepath + 'specEr.txt', self.specEr)
        savetxt(file=self.savepath + 'pspecCE.txt', data=self.pspecCE, trans=True)
        savetxt(file=self.savepath + 'pspecCEErr.txt', data=self.pspecCEErr, trans=True)

        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chNE)}, 
                  open(self.savepath + 'chNE.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chScale)}, 
                  open(self.savepath + 'chScale.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.pspecCE)}, 
                  open(self.savepath + 'pspecCE.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.chScaleErr)}, 
                  open(self.savepath + 'chScaleErr.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.pspecCEErr)}, 
                  open(self.savepath + 'pspecCEErr.json', 'w'), indent=4, cls=JsonEncoder)

        json.dump(self.specNE, open(self.savepath + 'specNE.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.specENE, open(self.savepath + 'specENE.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.specE2NE, open(self.savepath + 'specE2NE.json', 'w'), indent=4, cls=JsonEncoder)
        # ----- save data -----


    def mspec(self, mo_spec, ploter='matplotlib'):
        self.mo_exprs_sp = self.fobj.mo_exprs_sp
        self.mo_funcs_sp = self.fobj.mo_funcs_sp
        self.mo_spec = mo_spec

        for info in self.mo_spec:
            assert 'expr' in info
            expr = info['expr']
            expr = re.sub('\s*', '', expr)
            info['expr'] = expr
            expr_sp = re.split(r"[(+\-*/)]", expr)
            expr_sp = [ex for ex in expr_sp if ex != '']
            assert set(expr_sp) <= set(self.mo_exprs_sp), expr + ' is not present!'

            if 'Emin' not in info:
                Emin = min(list(chain.from_iterable(self.chReMin)))
                info['Emin'] = Emin
            else:
                Emin = info['Emin']

            if 'Emax' not in info:
                Emax = max(list(chain.from_iterable(self.chReMax)))
                info['Emax'] = Emax
            else:
                Emax = info['Emax']

            moEr = np.vstack((np.logspace(np.log10(Emin), np.log10(Emax), 300)[:-1],
                              np.logspace(np.log10(Emin), np.log10(Emax), 300)[1:])).T
            moE = np.array([np.sqrt(Er[0] * Er[1]) for Er in moEr])
            info['moE'] = moE

            if 'T' not in info:
                moT = None
            else:
                if info['T'] is None:
                    moT = None
                else:
                    moT = info['T'] * np.ones_like(moE)
            info['moT'] = moT

            NE, ENE, E2NE = {}, {}, {}
            for ex in expr_sp:
                if 'phabs' in ex or 'tbabs' in ex:
                    # NE[ex] = np.array(self.mo_funcs_sp[ex](moEr, moT))
                    NE[ex] = np.array(self.mo_funcs_sp[ex](moE, moT))
                    ENE[ex] = NE[ex]; E2NE[ex] = NE[ex]
                else:
                    NE[ex] = np.array(self.mo_funcs_sp[ex](moE, moT))
                    ENE[ex] = 1.60218e-9 * moE * NE[ex]
                    E2NE[ex] = 1.60218e-9 * moE * moE * NE[ex]

            NE.update({expr: oper_model(expr, NE)})
            ENE.update({expr: oper_model(expr, ENE)})
            E2NE.update({expr: oper_model(expr, E2NE)})

            info['NE'] = NE; info['ENE'] = ENE; info['E2NE'] = E2NE

        if len(self.mo_spec) <= 10:
            colors = px.colors.qualitative.Plotly
        elif 10 < len(self.mo_spec) <= 24:
            colors = px.colors.qualitative.Dark24
        else:
            colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(self.mo_spec)))

        if ploter == 'plotly':
            self.mspec_fig = go.Figure()

            for i, info in enumerate(self.mo_spec):
                expr, moE, moT = info['expr'], info['moE'], info['moT']
                mo = go.Scatter(x=moE.astype(float), 
                                y=info['NE'][expr].astype(float), 
                                name=('%s'%expr if moT is None else '%s@%.3f'%(expr, info['T'])), 
                                showlegend=True, 
                                mode='lines', 
                                line=dict(width=2, color=colors[i]))
                self.mspec_fig.add_trace(mo)

            self.mspec_fig.update_xaxes(title_text='Energy (keV)', type='log')
            self.mspec_fig.update_yaxes(title_text='N(E) (photons/cm2/s/keV)', type='log')

            self.mspec_fig.update_layout(height=600, width=800)

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            # rcParams['font.sans-serif'] = 'Arial'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            for i, info in enumerate(self.mo_spec):
                expr, moE, moT = info['expr'], info['moE'], info['moT']
                ax.plot(moE, info['NE'][expr], color=colors[i], lw=1.0, 
                        label=('%s'%expr if moT is None else '%s@%.3f'%(expr, info['T'])))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(r'$N_E~(\rm{Photons~cm^{-2}~s^{-1}~keV^{-1}})$')
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.legend(frameon=True)

            plt.savefig(self.savepath + 'ne_mo.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            for i, info in enumerate(self.mo_spec):
                expr, moE, moT = info['expr'], info['moE'], info['moT']
                ax.plot(moE, info['ENE'][expr], color=colors[i], lw=1.0, 
                        label=('%s'%expr if moT is None else '%s@%.3f'%(expr, info['T'])))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(r'$EN_E~(\rm{erg~cm^{-2}~s^{-1}~keV^{-1}})$')
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.legend(frameon=True)

            plt.savefig(self.savepath + 'ene_mo.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            for i, info in enumerate(self.mo_spec):
                expr, moE, moT = info['expr'], info['moE'], info['moT']
                ax.plot(moE, info['E2NE'][expr], color=colors[i], lw=1.0, 
                        label=('%s'%expr if moT is None else '%s@%.3f'%(expr, info['T'])))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel(r'$E^2N_E~(\rm{erg~cm^{-2}~s^{-1}})$')
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.legend(frameon=True)

            plt.savefig(self.savepath + 'eene_mo.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            print('+-----------------------------------------------+')
            print(' plotted combinated models')
            print('+-----------------------------------------------+\n')

        # ----- save data -----
        json.dump(self.mo_spec, open(self.savepath + 'mo_spec.json', 'w'), indent=4, cls=JsonEncoder)
        # ----- save data -----
