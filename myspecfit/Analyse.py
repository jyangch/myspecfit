import os
import json
import copy
import corner
import warnings
import numpy as np
from Fit import Fit
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import OrderedDict
from Tools import savetxt, JsonEncoder


class Analyse(object):

    def __init__(self, fobj):
        assert isinstance(fobj, Fit)
        self.fobj = fobj
        self.path = self.fobj.path
        self.config = None

        self.savepath = '%s/Analyse/' % self.path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)


    def set(self, err_level='1sigma', post_style='maxl', post_level='1sigma'):
        """
        Parameters
        ----------
        err_level: ['1sigma', '2sigma', '3sigma']
        post_style: ['maxl', 'midv']
        post_level: : ['nsigma', '1sigma', '2sigma', '3sigma']
        ----------
        """

        print('\n+-----------------------------------------------+')
        print(' the asnlysis settings:')
        print('+-----------------------------------------------+')
        print(' %-16s%-16s%-16s' % ('err_level', 'post_style', 'post_level'))
        print('+-----------------------------------------------+')
        print(' %-16s%-16s%-16s' % (err_level, post_style, post_level))
        print('+-----------------------------------------------+\n')

        self.config = {'err_level': err_level, 'post_style': post_style, 'post_level': post_level}
        self.err_level, self.post_style, self.post_level = err_level, post_style, post_level


    def post(self, pdf=True):
        if self.config is None: self.set()

        self.spec_exprs = self.fobj.spec_exprs
        self.argsort_ll_samples = np.argsort(self.fobj.equal_weighted_samples[: ,-1])[::-1]
        self.sort_ll_samples = self.fobj.equal_weighted_samples[: ,-1][self.argsort_ll_samples]
        self.sort_pv_samples = self.fobj.equal_weighted_samples[: ,0:-1][self.argsort_ll_samples]
        self.critical_lnL = np.percentile(self.sort_ll_samples, 19)

        self.post_values = None
        self.post_lnL = None
        self.post_fit = OrderedDict()

        if self.post_style == 'maxl':
            if self.post_level == 'nsigma':
                self.post_values = self.sort_pv_samples[0]
            else:
                if self.post_level not in ['1sigma', '2sigma', '3sigma']:
                    raise ValueError('invalid value for post_level!')
                for pvs, lls in zip(self.sort_pv_samples, self.sort_ll_samples):
                    if np.array([True if (st[self.post_level][0] <= pvs[i] <= st[self.post_level][1]) else False
                                    for i, st in enumerate(self.fobj.stats)]).all():
                        self.post_values = pvs
                        self.post_lnL = lls
                        break
                if self.post_values is None and self.post_lnL is None:
                    self.post_values = self.sort_pv_samples[0]
                    warnings.warn('no sample in %s region, back to global best fit!' % self.post_level)
                elif self.post_lnL < self.critical_lnL:
                    self.post_values = self.sort_pv_samples[0]
                    warnings.warn('no sample with a logL larger than critical value %.4f!' % self.critical_lnL)
                else: pass
        elif self.post_style == 'midv':
            self.post_values = np.array([st['median'] for st in self.fobj.stats])
        else:
            raise ValueError('invalid value for post_style!')

        for i, (pi, pl, st) in enumerate(zip(self.fobj.pids, self.fobj.plabels, self.fobj.stats)):
            pv = self.post_values[i]
            lo1, hi1 = st['1sigma']; lo2, hi2 = st['2sigma']; lo3, hi3 = st['3sigma']
            self.post_fit[pi] = {'label': pl, 'value': pv, '1sigma_err': [max(0, (pv - lo1)), max(0, (hi1 - pv))],
                                 '2sigma_err': [max(0, (pv - lo2)), max(0, (hi2 - pv))],
                                 '3sigma_err': [max(0, (pv - lo3)), max(0, (hi3 - pv))]}

        if self.fobj.engine == 'multinest':
            self.post_lnL = self.fobj.log_ll_multinest(cube=self.post_values, ndim=self.fobj.nparams, nparams=self.fobj.nparams)
        elif self.fobj.engine == 'emcee':
            self.post_lnL = self.fobj.log_ll_emcee(self.post_values)
        else: raise ValueError('invalid value for fitting engine!')

        self.post_pdicts = copy.deepcopy(self.fobj.pdicts)
        self.post_params = copy.deepcopy(self.fobj.params)
        self.post_mo_params = copy.deepcopy(self.fobj.mo_params)

        self.post_stat, self.post_stat_list = self.fobj.stat, self.fobj.stat_list
        self.post_nps, self.post_nps_list = self.fobj.nps, self.fobj.nps_list
        self.post_nparams, self.post_dof = self.fobj.nparams, self.fobj.dof
        self.post_lnL, self.post_lnL_list = self.fobj.ll, self.fobj.ll_list
        self.post_bic, self.post_aic, self.post_aicc = self.fobj.bic, self.fobj.aic, self.fobj.aicc
        self.post_lnZ = self.fobj.lnZ

        self.post_goodness = {'lnL': self.post_lnL, 'stat': self.post_stat, 'npoints': self.post_nps,
                              'nparams': self.post_nparams, 'dof': self.post_dof, 'bic': self.post_bic,
                              'aic': self.post_aic, 'aicc': self.post_aicc, 'lnZ': self.post_lnZ}

        self.post_spec_goodness = OrderedDict()
        for i, expr in enumerate(self.spec_exprs):
            self.post_spec_goodness[expr] = {'lnL': self.post_lnL_list[i], 'stat': self.post_stat_list[i],
                                             'npoints': self.post_nps_list[i]}

        self.Post_Stats = {'goodness': self.post_goodness, 'parameter': self.post_fit,
                           'spec_goodness': self.post_spec_goodness}

        self.Post_Fit = []
        for pi, pl in zip(self.fobj.pids, self.fobj.plabels):
            self.Post_Fit.append([pi, pl, self.post_fit[pi]['value']] + self.post_fit[pi]['%s_err'%self.err_level])

        self.Post_Goodness = []
        self.Post_Goodness.append(['lnL', self.post_lnL])
        self.Post_Goodness.append(['STAT', self.post_stat])
        self.Post_Goodness.append(['npoints', self.post_nps])
        self.Post_Goodness.append(['nparams', self.post_nparams])
        self.Post_Goodness.append(['DOF', self.post_dof])
        self.Post_Goodness.append(['BIC', self.post_bic])
        self.Post_Goodness.append(['AIC', self.post_aic])
        self.Post_Goodness.append(['AICc', self.post_aicc])
        self.Post_Goodness.append(['lnZ', self.post_lnZ])

        print('+--------------------------------------------------------------------+')
        print(' posterior paramenter distribution analyse: ')
        print('+--------------------------------------------------------------------+')
        print(' %-5s%-20s%-20s%-25s' % ('pid', 'expr', 'label', 'value'))
        print('+--------------------------------------------------------------------+')
        for pi, ex, pl, pv in zip(self.fobj.pids, self.fobj.pexprs, self.fobj.plabels, self.Post_Fit):
            pv_ = '%.3f_{-%.3f}^{+%.3f}'%(pv[2], pv[3], pv[4])
            print(' %-5s%-20s%-20s%-25s' % (pi, ex, pl, pv_))
        print('+--------------------------------------------------------------------+\n')

        print('+--------------------------------------------------------------------+')
        print(' posterior model fitting goodness: ')
        print('+--------------------------------------------------------------------+')
        print(' %-12s%-12s%-12s%-12s%-12s%-12s' % ('lnL', 'STAT', 'dof', 'BIC', 'AIC', 'AICc'))
        print('+--------------------------------------------------------------------+')
        print(' %-12.2f%-12.2f%-12d%-12.2f%-12.2f%-12.2f' % 
              (self.post_lnL, self.post_stat, self.post_dof, self.post_bic, self.post_aic, self.post_aicc))
        print('+--------------------------------------------------------------------+\n')

        if pdf:
            rcParams['font.family'] = 'sans-serif'
            # rcParams['font.sans-serif'] = 'Arial'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(7, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(self.fobj.equal_weighted_samples[:, -1], color='b', lw=1.0)
            ax.axhline(self.post_lnL, color='r', lw=1.0)
            ax.axhline(self.critical_lnL, color='c', lw=1.0)
            ax.set_xlabel('Step number')
            ax.set_ylabel('ln$L$')
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
            plt.savefig(self.savepath + 'sampler_lnL.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)

            fig = plt.figure(figsize=(7, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])
            ax.hist(self.sort_ll_samples, bins=30, density=False, histtype='step', color='w',
                    edgecolor='b', fill=True, linewidth=1.5)
            ax.axvline(self.post_lnL, color='r', lw=1.0)
            ax.axvline(self.critical_lnL, color='c', lw=1.0)
            ax.set_xlabel('ln$L$')
            ax.set_ylabel('Number')
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
            plt.savefig(self.savepath + 'sampler_lnL_pdf.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)

        # ----- save data -----
        savetxt(file=self.savepath + 'post_fit.txt', data=self.Post_Fit)
        savetxt(file=self.savepath + 'post_goodness.txt', data=self.Post_Goodness)
        np.savetxt(self.savepath + 'sort_ll_samples.txt', self.sort_ll_samples)
        np.savetxt(self.savepath + 'sort_pv_samples.txt', self.sort_pv_samples)
    
        json.dump(self.spec_exprs, open(self.savepath + 'spec_exprs.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.config, open(self.savepath + 'config.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.critical_lnL, open(self.savepath + 'critical_logL.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.post_pdicts, open(self.savepath + 'post_pdicts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.post_params, open(self.savepath + 'post_params.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.post_mo_params, open(self.savepath + 'post_mo_params.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.Post_Stats, open(self.savepath + 'post_stats.json', 'w'), indent=4, cls=JsonEncoder)

        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.src_NchCounts)}, 
                  open(self.savepath + 'src_NchCounts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.bkg_NchCounts)}, 
                  open(self.savepath + 'bkg_NchCounts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.mo_NchRate)}, 
                  open(self.savepath + 'mo_NchRate.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.src_eff)}, 
                  open(self.savepath + 'src_eff.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.bkg_eff)}, 
                  open(self.savepath + 'bkg_eff.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.bkg_NchErr)}, 
                  open(self.savepath + 'bkg_NchErr.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.fobj.NchIndex)}, 
                  open(self.savepath + 'NchIndex.json', 'w'), indent=4, cls=JsonEncoder)
        # ----- save data -----


    def corner(self, smooth=2, level=[1, 1.5, 2]):
        self.engine = self.fobj.engine
        self.nparams = self.fobj.nparams
        self.plabels = self.fobj.plabels
        self.weights = self.fobj.weights

        if self.engine == 'multinest':
            mask = self.weights > 1e-5
            self.weights = self.weights[mask]
            self.pdata = self.fobj.weighted_samples[mask]
        else:
            self.pdata = self.fobj.equal_weighted_samples

        rcParams['font.family'] = 'sans-serif'
        # rcParams['font.sans-serif'] = 'Arial'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        self.corner_fig = corner.corner(self.pdata, bins=30, weights=self.weights, color='blue', smooth=smooth, smooth1d=smooth,
                                        labels=self.plabels, show_titles=True, levels=1.0-np.exp(-0.5*np.array(level)**2))
        axes = np.array(self.corner_fig.axes).reshape((self.nparams, self.nparams))
        for i in range(self.nparams):
            ax = axes[i, i]
            fmt = '%s = $%.2f_{-%.2f}^{+%.2f}$'
            title = (self.plabels[i], self.Post_Fit[i][2], self.Post_Fit[i][3], self.Post_Fit[i][4])
            ax.set_title(fmt % title)
            ax.errorbar(self.Post_Fit[i][2], 0.005, xerr=[[self.Post_Fit[i][3]], [self.Post_Fit[i][4]]], 
                        fmt='or', ms=2, ecolor='r', elinewidth=1)
        for yi in range(self.nparams):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.errorbar(self.Post_Fit[xi][2], self.Post_Fit[yi][2],
                            xerr=[[self.Post_Fit[xi][3]], [self.Post_Fit[xi][4]]],
                            yerr=[[self.Post_Fit[yi][3]], [self.Post_Fit[yi][4]]],
                            fmt='or', ms=2, ecolor='r', elinewidth=1)
        plt.savefig(self.savepath + 'corner.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

        print('+-----------------------------------------------+')
        print(' plotted corner')
        print('+-----------------------------------------------+\n')

        # ----- save data -----
        np.savetxt(self.savepath + 'pdata.txt', self.pdata)
        np.savetxt(self.savepath + 'weights.txt', self.weights)
        savetxt(file=self.savepath + 'nparams.txt', data=[[self.nparams]])
        savetxt(file=self.savepath + 'plabels.txt', data=[self.plabels], trans=True)
        json.dump(self.engine, open(self.savepath + 'engine.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.nparams, open(self.savepath + 'nparams.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.plabels, open(self.savepath + 'plabels.json', 'w'), indent=4, cls=JsonEncoder)
        # ----- save data -----