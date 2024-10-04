import os
import re
import json
import numpy as np
from Analyse import Analyse
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import simps
from Tools import JsonEncoder, quantile, oper_model


class Calculate(object):

    def __init__(self, aobj):
        assert isinstance(aobj, Analyse)
        self.aobj = aobj
        self.fobj = aobj.fobj
        self.path = self.fobj.path

        self.savepath = '%s/Calculate/' % self.path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)


    def flux(self, expr, T=None, e1=10, e2=1000):
        self.nparams = self.fobj.nparams
        self.params = self.fobj.params
        self.pids = self.fobj.pids
        self.edata = self.fobj.equal_weighted_samples
        self.post_values = self.aobj.post_values
        self.mo_exprs_sp = self.fobj.mo_exprs_sp
        self.mo_funcs_sp = self.fobj.mo_funcs_sp

        self.Er = np.vstack((np.logspace(np.log10(e1), np.log10(e2), 1000)[:-1],
                             np.logspace(np.log10(e1), np.log10(e2), 1000)[1:])).T
        self.E = np.array([np.sqrt(Eri[0] * Eri[1]) for Eri in self.Er])
        self.T = T * np.ones_like(self.E) if T is not None else T

        self.ergfluxs = np.zeros(len(self.edata))
        self.ergflux_mp = {}
        self.phtfluxs = np.zeros(len(self.edata))
        self.phtflux_mp = {}

        self.expr = re.sub('\s*', '', expr)
        expr_sp = re.split(r"[(+\-*/)]", self.expr)
        self.expr_sp = [ex for ex in expr_sp if ex != '']
        assert set(self.expr_sp) <= set(self.mo_exprs_sp), expr + ' is not present!'

        for k in range(len(self.edata)):
            pvs = self.edata[k, 0:self.nparams]
            for pi, pv in zip(self.pids, pvs): self.params[pi]['param'].val = pv
            self.phtfluxs[k], self.ergfluxs[k] = self.calf()
        pfv_lo, pfv_hi = quantile(self.phtfluxs, [0.16, 0.84])
        efv_lo, efv_hi = quantile(self.ergfluxs, [0.16, 0.84])

        pvs = self.post_values
        for pi, pv in zip(self.pids, pvs): self.params[pi]['param'].val = pv
        pfv, efv = self.calf()
        self.phtflux_mp['value'] = pfv
        self.ergflux_mp['value'] = efv

        if efv_lo > efv: efv_lo = efv
        if efv_hi < efv: efv_hi = efv
        if pfv_lo > pfv: pfv_lo = pfv
        if pfv_hi < pfv: pfv_hi = pfv

        self.phtflux_mp['1sigma_err'] = [pfv - pfv_lo, pfv_hi - pfv]
        self.ergflux_mp['1sigma_err'] = [efv - efv_lo, efv_hi - efv]
        
        rcParams['font.family'] = 'sans-serif'
        # rcParams['font.sans-serif'] = 'Arial'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        bins = np.logspace(min(np.log10(self.phtfluxs)), max(np.log10(self.phtfluxs)), 30)
        histvalue, histbin = np.histogram(self.phtfluxs, bins=bins)
        xx = (histbin[1:] + histbin[:-1]) / 2
        yy = histvalue
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.plot(xx, yy, ds='steps-mid', color='blue')
        ax.axvline(pfv, c='grey')
        ax.axvline(pfv_lo, c='grey', ls='--')
        ax.axvline(pfv_hi, c='grey', ls='--')
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
        plt.savefig(self.savepath + 'phtflux_pdf.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)

        bins = np.logspace(min(np.log10(self.ergfluxs)), max(np.log10(self.ergfluxs)), 30)
        histvalue, histbin = np.histogram(self.ergfluxs, bins=bins)
        xx = (histbin[1:] + histbin[:-1]) / 2
        yy = histvalue
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.plot(xx, yy, ds='steps-mid', color='blue')
        ax.axvline(efv, c='grey')
        ax.axvline(efv_lo, c='grey', ls='--')
        ax.axvline(efv_hi, c='grey', ls='--')
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
        plt.savefig(self.savepath + 'ergflux_pdf.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)

        print('+-----------------------------------------------+')
        print(' calculated flux within [%.2f, %.2f]' % (e1, e2))
        print('+-----------------------------------------------+\n')

        # ----- save data -----
        np.savetxt(self.savepath + 'E.txt', self.E)
        np.savetxt(self.savepath + 'Er.txt', self.Er)
        np.savetxt(self.savepath + 'phtflux_post.txt', self.phtfluxs)
        np.savetxt(self.savepath + 'ergflux_post.txt', self.ergfluxs)
        json.dump({'expr': expr, 'T': T, 'e1': e1, 'e2': e2}, 
                  open(self.savepath + 'flux_params.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.phtflux_mp, open(self.savepath + 'phtflux.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.ergflux_mp, open(self.savepath + 'ergflux.json', 'w'), indent=4, cls=JsonEncoder)
        # ----- save data -----


    def calf(self):
        NE_ = {}
        for ex in self.expr_sp:
            if 'phabs' in ex or 'tbabs' in ex:
                assert len(self.expr_sp) != 1, 'can not calculate flux only for abs model!'
                # NE_[ex] = np.array(self.mo_funcs_sp[ex](self.Er, self.T))
                NE_[ex] = np.array(self.mo_funcs_sp[ex](self.E, self.T))
            else:
                NE_[ex] = np.array(self.mo_funcs_sp[ex](self.E, self.T))

        NE = oper_model(self.expr, NE_)
        ENE = self.E * NE
        pf = simps(NE, self.E)
        ef = 1.60218e-9 * simps(ENE, self.E)
        return pf, ef
