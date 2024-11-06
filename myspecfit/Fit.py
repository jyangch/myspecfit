import os
import json
import emcee
import warnings
import pymultinest
import numpy as np
from Model import Model
from itertools import chain
from Tools import JsonEncoder
from Spectrum import Spectrum
import matplotlib.pyplot as plt
from collections import Counter
from collections import OrderedDict
from scipy.optimize import minimize
from os.path import dirname, abspath
docs_path = dirname(abspath(__file__)) + '/docs'


class Fit(object):

    def __init__(self):
        self.clear()


    def clear(self):
        self.Spectrum = list()
        self.Model = list()
        self.check_status = False

    
    def set(self, spectrum, model):
        assert isinstance(spectrum, Spectrum), 'spectrum parameter should be Spectrum type!'
        self.Spectrum.append(spectrum)

        assert isinstance(model, Model), 'model parameter should be Model type!'
        self.Model.append(model)

        spectrum.fit_with(model)
        model.fit_to(spectrum)


    def _with(self, model):
        assert isinstance(model, Model), 'model parameter should be Model type!'
        self.Model.append(model)
        
        assert model.spectrum is not None, 'to which spectrum the model will fit?'
        self.Spectrum.append(model.spectrum)


    def _to(self, spectrum):
        assert isinstance(spectrum, Spectrum), 'spectrum parameter should be Spectrum type!'
        self.Spectrum.append(spectrum)
        
        assert spectrum.model is not None, 'with which model the spectrum will be fitted?'
        self.Model.append(spectrum.model)


    def check(self):
        for spec in self.Spectrum: spec.check()

        self.srcs = [src for spec in self.Spectrum for src in spec.src_info]
        self.bkgs = [bkg for spec in self.Spectrum for bkg in spec.bkg_info]
        self.rsps = [rsp for spec in self.Spectrum for rsp in spec.rsp_info]

        self.spec_exprs = [expr for spec in self.Spectrum for expr in spec.exprs]
        self.specTs = [specT for spec in self.Spectrum for specT in spec.specTs]
        self.spec_weights = [wt for spec in self.Spectrum for wt in spec.wts]
        self.nspecs = [spec.nspec for spec in self.Spectrum]
        self.nspec = sum(self.nspecs)

        assert_info = 'sorry for prohibiting the use of the same group name!'
        assert len(set(self.spec_exprs)) == self.nspec, assert_info

        self.spec_params = [pd for spec in self.Spectrum for pd in spec.params]
        self.spec_nparams = [len(pd) for pd in self.spec_params]
        self.spec_pdicts = {key: value for pr in self.spec_params for key, value in pr.items()}

        self.src_chIndex = [src.ChanIndex for src in self.srcs]
        self.src_chCounts = [src.SrcCounts for src in self.srcs]
        self.src_chErr = [src.SrcErr for src in self.srcs]
        self.src_expo_ = [src.SrcExpo for src in self.srcs]

        self.bkg_chIndex = [bkg.ChanIndex for bkg in self.bkgs]
        self.bkg_chCounts = [bkg.BkgCounts for bkg in self.bkgs]
        self.bkg_chErr_ = [bkg.BkgErr for bkg in self.bkgs]
        self.bkg_expo_ = [bkg.BkgExpo for bkg in self.bkgs]

        self.rsp_chIndex = [rsp.ChanIndex for rsp in self.rsps]
        self.rsp_chMin = [np.array(rsp.ChanMin) for rsp in self.rsps]
        self.rsp_chMax = [np.array(rsp.ChanMax) for rsp in self.rsps]
        self.rsp_chWidth = [np.array(rsp.ChanWidth) for rsp in self.rsps]
        self.rsp_chCenter = [np.array(rsp.ChanCenter) for rsp in self.rsps]

        self.Nranges = [rsp.Qualified_Notice for rsp in self.rsps]
        self.NchIndex = [rsp.Qualified_Notice_ID for rsp in self.rsps]

        self.stat_exprs = [stat for spec in self.Spectrum for stat in spec.stats]
        self.stat_funcs = [func for spec in self.Spectrum for func in spec.stat_funcs]

        self.mos = [mo for mo in self.Model]
        self.mo_exprs = [mo.expr for mo in self.Model]
        self.mo_params = [mo.pdicts for mo in self.Model]
        self.mo_nparams = [len(pd) for pd in self.mo_params]
        self.mo_funcs = [mo.func for mo in self.Model]

        self.mo_exprs_sp = list(set(chain.from_iterable([mo.expr_sp for mo in self.Model])))
        self.mo_funcs_sp = {key: value for mo in self.Model for key, value in mo.fdicts.items()}

        print('+-----------------------------------------------+')
        print(' %-15s%-10s%-25s' % ('Spec', 'Stat', 'Model'))
        print('+-----------------------------------------------+')
        for i, j in enumerate(chain.from_iterable([i] * n for i, n in enumerate(self.nspecs))):
            print(' %-15s%-10s%-25s' % (self.spec_exprs[i], self.stat_exprs[i], self.mo_exprs[j]))
        print('+-----------------------------------------------+\n')

        self.check_status = True
        self.check_pdict()
        self.check_param()

        self.engine_dict = {'multinest': self.run_multinest, 
                            'emcee': self.run_emcee, 
                            'som': self.run_som}


    def check_pdict(self):
        pid_int = 0
        self.idpid = OrderedDict()
        self.pdicts = OrderedDict()

        for mo in self.Model:
            for pl, pr in mo.pdicts.items():
                pid_int += 1; pid = 'p%02d' % pid_int
                self.idpid[pid] = id(pr); self.idpid[str(id(pr))] = pid
                self.pdicts[pid] = {'expr': mo.expr, 'label': pl, 'param': pr, 'frozen': pr.frozen, 'mates': set()}

        for pl, pr in self.spec_pdicts.items():
            pid_int += 1; pid = 'p%02d' % pid_int
            self.idpid[pid] = id(pr); self.idpid[str(id(pr))] = pid
            self.pdicts[pid] = {'expr': pl.split('@')[1], 'label': pl, 'param': pr, 'frozen': pr.frozen, 'mates': set()}

        # if some parameters already links with others before fit, need add pid into mates
        for ip, pv in self.pdicts.items():
            for mate in pv['param'].mates:
                jp = self.idpid[str(id(mate))]
                if jp != ip: self.pdicts[ip]['mates'].add(jp)

        # two cases: 1, same mini-model in different Model, 
        #               the same-label parameters are really same, no need for link
        #            2, different mini-model in different Model, but label is same, need link
        # for both caces, need add pid into mates
        pls = [pv['label'] for pv in self.pdicts.values()]
        for pl, n in Counter(pls).items():
            if n > 1: self.link(['p%02d' % (i + 1) for i, pi in enumerate(pls) if pi == pl], False)


    def check_param(self):
        self.linked = set()
        self.params = OrderedDict()
        
        for pid, pv in self.pdicts.items():
            pr = pv['param']
            if pr.frozen:
                if len(pv['mates']) != 0: 
                    self.linked.update(pv['mates'])
                    warnings.warn('frozen parameter %s links with other parameters %s' % (pid, pv['mates']))
            else:
                if pid not in self.linked:
                    self.params[pid] = pv
                    self.linked.update(pv['mates'])
                else:
                    self.linked.update(set(mate for mate in pv['mates'] if mate not in self.params))

        self.pids = list(self.params.keys())
        self.pexprs = [pv['expr'] for pv in self.params.values()]
        self.plabels = [pv['label'] for pv in self.params.values()]
        self.pranges = [pv['param'].range for pv in self.params.values()]
        self.pinits = [pv['param'].val for pv in self.params.values()]
        self.nparams = len(self.plabels)
        self.info


    def link(self, pids, check_param=True):
        assert self.check_status, 'please run check method first!'

        for i, ip in enumerate(pids):
            for j, jp in enumerate(pids):
                if j > i:
                    self.pdicts[ip]['mates'].add(jp); self.pdicts[jp]['mates'].add(ip)
                    if id(self.pdicts[ip]['param']) != id(self.pdicts[jp]['param']):
                        self.pdicts[ip]['param'].link(self.pdicts[jp]['param'])

        print('+-----------------------------------------------+')
        print(' parameter link note: ')
        print('+-----------------------------------------------+')
        print(' %-6s%-22s%-22s' % ('pid', 'expr', 'label'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-6s%-22s%-22s' % (pi, self.pdicts[pi]['expr'], self.pdicts[pi]['label']) for pi in pids]))
        print('+-----------------------------------------------+\n')

        if check_param: self.check_param()


    @property
    def info(self):
        assert self.check_status, 'please run check method first!'

        print('+--------------------------------------------------------------------+')
        print(' %-5s%-20s%-20s%-8s%-17s' % ('pid', 'label', 'range', 'frozen', 'mates'))
        for i, (pid, pv) in enumerate(self.pdicts.items()):
            if i in np.cumsum([0] + self.mo_nparams + self.spec_nparams[:-1]):
                print('+--------------------------------------------------------------------+')
                print('{:^70s}'.format('<<< ' + pv['expr'] + ' >>>'))
                print('+--------------------------------------------------------------------+')
            label, range_, frozen = pv['label'], pv['param'].range, pv['frozen']
            mates = pv['mates'] if len(pv['mates']) != 0 else ''
            print(' %-5s%-20s%-20s%-8s%-17s' % (pid, label, range_, frozen, mates))
        print('+--------------------------------------------------------------------+\n')

        print('+--------------------------------------------------------------------+')
        print(' below parameters will be used for fitting')
        print('+--------------------------------------------------------------------+')
        print(' %-10s%-20s%-20s%-20s' % ('pid', 'expr', 'label', 'range'))
        print('+--------------------------------------------------------------------+')
        for pi, ex, pl, pr in zip(self.pids, self.pexprs, self.plabels, self.pranges):
            print(' %-10s%-20s%-20s%-20s' % (pi, ex, pl, pr))
        print('+--------------------------------------------------------------------+\n')


    def run(self, engine, path, nlive=500, nstep=2000, discard=100, resume=True):
        if not self.check_status: self.check()

        self.engine = engine
        assert engine in self.engine_dict.keys(), engine + ' is not one of engines!'

        self.path = path

        for i, (spec, mo) in enumerate(zip(self.Spectrum, self.Model)):
            spec.save('%s/Spectrum/spec-%d/' % (self.path, i+1))
            mo.save('%s/Model/mo-%d/' % (self.path, i+1))

        self.savepath = '%s/Fit/' % self.path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.nlive = nlive
        self.nstep = nstep
        self.discard = discard
        self.resume = resume

        self.engine_dict[self.engine]()

        # ----- save data -----
        json.dump(self.nspec, open(self.savepath + 'nspec.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.nspecs, open(self.savepath + 'nspecs.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.spec_exprs, open(self.savepath + 'spec_exprs.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.specTs)}, 
                  open(self.savepath + 'specTs.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.spec_weights)}, 
                  open(self.savepath + 'spec_weights.json', 'w'), indent=4, cls=JsonEncoder)

        json.dump(self.spec_nparams, open(self.savepath + 'spec_nparams.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.spec_pdicts, open(self.savepath + 'spec_pdicts.json', 'w'), indent=4, cls=JsonEncoder) 
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.spec_params)}, 
                  open(self.savepath + 'spec_params.json', 'w'), indent=4, cls=JsonEncoder)
        
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.Nranges)}, 
                  open(self.savepath + 'Nranges.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.NchIndex)}, 
                  open(self.savepath + 'NchIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.spec_exprs, self.stat_exprs)}, 
                  open(self.savepath + 'stat_exprs.json', 'w'), indent=4, cls=JsonEncoder)

        json.dump(self.mo_exprs, open(self.savepath + 'mo_exprs.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.mo_nparams, open(self.savepath + 'mo_nparams.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.mo_exprs_sp, open(self.savepath + 'mo_exprs_sp.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: val for ex, val in zip(self.mo_exprs, self.mo_params)}, 
                  open(self.savepath + 'mo_params.json', 'w'), indent=4, cls=JsonEncoder)

        json.dump(self.idpid, open(self.savepath + 'idpid.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.pdicts, open(self.savepath + 'pdicts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.params, open(self.savepath + 'params.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.linked, open(self.savepath + 'linked.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.nparams, open(self.savepath + 'nparams.json', 'w'), indent=4, cls=JsonEncoder)

        json.dump(self.engine, open(self.savepath + 'engine.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.nlive, open(self.savepath + 'nlive.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.nstep, open(self.savepath + 'nstep.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.discard, open(self.savepath + 'discard.json', 'w'), indent=4, cls=JsonEncoder)
        # ----- save data -----


    # ---------------------------------------
    # below for multinest!
    # ---------------------------------------
    def prior_multinest(self, cube, ndim, nparams):
        for i in range(ndim):
            cube[i] = (self.pranges[i][1] - self.pranges[i][0]) * cube[i] + self.pranges[i][0]


    def log_ll_multinest(self, cube, ndim, nparams):
        cube = np.array([cube[i] for i in range(ndim)], dtype=np.float64)
        for pi, pv in zip(self.pids, cube): self.params[pi]['param'].val = pv
        self.mo_chRate = [rate for mo in self.mos for rate in mo.conv_drm()]

        self.src_NchCounts = [count[index] for count, index in zip(self.src_chCounts, self.NchIndex)]
        self.bkg_NchCounts = [count[index] for count, index in zip(self.bkg_chCounts, self.NchIndex)]
        self.mo_NchRate = [rate[index] for rate, index in zip(self.mo_chRate, self.NchIndex)]
        self.src_NchErr = [err[index] for err, index in zip(self.src_chErr, self.NchIndex)]
        self.bkg_NchErr_ = [err[index] for err, index in zip(self.bkg_chErr_, self.NchIndex)]

        if True in [np.isnan(m).any() for m in self.mo_NchRate]: return -np.inf
        if True in [np.isinf(m).any() for m in self.mo_NchRate]: return -np.inf

        self.src_expo = [po * self.spec_pdicts['sf@%s'%ex].value for po, ex in zip(self.src_expo_, self.spec_exprs)]
        self.bkg_expo = [po * self.spec_pdicts['bf@%s'%ex].value for po, ex in zip(self.bkg_expo_, self.spec_exprs)]

        f = lambda bi, bierr, ex: np.sqrt(bierr ** 2 + self.spec_pdicts['bvf@%s'%ex].value ** 2 * bi ** 2)
        self.bkg_NchErr = list(map(f, self.bkg_NchCounts, self.bkg_NchErr_, self.spec_exprs))

        self.stat_list = np.array(list(map(lambda s, b, m, ts, tb, berr, func: func(s, b, m, ts, tb, berr),
                                           self.src_NchCounts, self.bkg_NchCounts, self.mo_NchRate, self.src_expo,
                                           self.bkg_expo, self.bkg_NchErr, self.stat_funcs))).astype(float)

        self.stat = np.sum([s * w for s, w in zip(self.stat_list, self.spec_weights)])

        self.nps_list = np.array([len(rate) for rate in self.mo_NchRate]).astype(int)
        self.nps = np.sum(self.nps_list)

        self.dof = self.nps - self.nparams
        if self.dof <= 0: warnings.warn('dof should not be less than zero!')

        self.ll = -0.5 * self.stat
        self.ll_list = -0.5 * self.stat_list

        self.bic = self.stat + self.nparams * np.log(self.nps)
        self.aic = self.stat + 2 * self.nparams
        self.aicc = self.stat + 2 * self.nparams + 2 * self.nparams * (self.nparams + 1) / (self.nps - self.nparams - 1)

        return float(self.ll)


    def run_multinest(self):
        self.prefix = self.savepath + '1-'

        pymultinest.run(self.log_ll_multinest, self.prior_multinest, self.nparams, resume=self.resume, 
                        verbose=True, n_live_points=self.nlive, outputfiles_basename=self.prefix, 
                        sampling_efficiency=0.8, importance_nested_sampling=True, multimodal=True)

        self.Analyzer = pymultinest.Analyzer(outputfiles_basename=self.prefix, n_params=self.nparams)
        self.Data = self.Analyzer.get_data()
        self.Stats = self.Analyzer.get_stats()
        self.Best_Fit = self.Analyzer.get_best_fit()

        self.stats = self.Stats['marginals']
        self.weights = self.Data[:, 0]
        self.weighted_samples = self.Data[:, 2:]
        self.equal_weighted_samples = self.Analyzer.get_equal_weighted_posterior()

        self.lnZ = self.Stats['nested importance sampling global log-evidence']

        json.dump(self.Stats, open(self.prefix + 'stats.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.Best_Fit, open(self.prefix + 'best_fit.json', 'w'), indent=4, cls=JsonEncoder)


    # ---------------------------------------
    # below for emcee!
    # ---------------------------------------
    def log_prior_emcee(self, theta):
        for i in range(len(theta)):
            if not (self.pranges[i][0] <= theta[i] <= self.pranges[i][1]):
                return -np.inf
        return 0.0


    def log_ll_emcee(self, theta):
        theta = np.array(theta, dtype=np.float64)
        for pi, pv in zip(self.pids, theta): self.params[pi]['param'].val = pv
        self.mo_chRate = [rate for mo in self.mos for rate in mo.conv_drm()]

        self.src_NchCounts = [count[index] for count, index in zip(self.src_chCounts, self.NchIndex)]
        self.bkg_NchCounts = [count[index] for count, index in zip(self.bkg_chCounts, self.NchIndex)]
        self.mo_NchRate = [rate[index] for rate, index in zip(self.mo_chRate, self.NchIndex)]
        self.src_NchErr = [err[index] for err, index in zip(self.src_chErr, self.NchIndex)]
        self.bkg_NchErr_ = [err[index] for err, index in zip(self.bkg_chErr_, self.NchIndex)]

        if True in [np.isnan(m).any() for m in self.mo_NchRate]: return -np.inf
        if True in [np.isinf(m).any() for m in self.mo_NchRate]: return -np.inf

        self.src_expo = [po * self.spec_pdicts['sf@%s'%ex].value for po, ex in zip(self.src_expo_, self.spec_exprs)]
        self.bkg_expo = [po * self.spec_pdicts['bf@%s'%ex].value for po, ex in zip(self.bkg_expo_, self.spec_exprs)]

        f = lambda bi, bierr, ex: np.sqrt(bierr ** 2 + self.spec_pdicts['bvf@%s'%ex].value ** 2 * bi ** 2)
        self.bkg_NchErr = list(map(f, self.bkg_NchCounts, self.bkg_NchErr_, self.spec_exprs))

        self.stat_list = np.array(list(map(lambda s, b, m, ts, tb, berr, func: func(s, b, m, ts, tb, berr),
                                           self.src_NchCounts, self.bkg_NchCounts, self.mo_NchRate, self.src_expo,
                                           self.bkg_expo, self.bkg_NchErr, self.stat_funcs))).astype(float)

        self.stat = np.sum([s * w for s, w in zip(self.stat_list, self.spec_weights)])

        self.nps_list = np.array([len(rate) for rate in self.mo_NchRate]).astype(int)
        self.nps = np.sum(self.nps_list)

        self.dof = self.nps - self.nparams
        if self.dof <= 0: warnings.warn('dof should not be less than zero!')

        self.ll = -0.5 * self.stat
        self.ll_list = -0.5 * self.stat_list

        self.bic = self.stat + self.nparams * np.log(self.nps)
        self.aic = self.stat + 2 * self.nparams
        self.aicc = self.stat + 2 * self.nparams + 2 * self.nparams * (self.nparams + 1) / (self.nps - self.nparams - 1)

        return float(self.ll)


    def log_prob_emcee(self, theta):
        lp = self.log_prior_emcee(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_ll_emcee(theta)


    def run_som(self):
        self.prefix = self.savepath + '1-'

        np.random.seed(42)
        nll = lambda *args: -self.log_ll_emcee(*args)
        pos = self.pinits + 1e-4 * np.random.randn(self.nparams)
        soln = minimize(nll, pos)

        print('+-----------------------------------------------+')
        print(" Maximum likelihood estimates:")
        print(" log_likelihood: {0:.3f}".format(-nll(soln.x)))
        for i in range(len(self.plabels)):
            print('\t' + self.plabels[i] + ' = {0:.3f}'.format(soln.x[i]))
        print('+-----------------------------------------------+\n')


    def run_emcee(self):
        self.prefix = self.savepath + '1-'

        np.random.seed(42)
        ndim = self.nparams
        nwalkers = 32 if 2 * ndim < 32 else 2 * ndim
        pos = self.pinits + 1e-4 * np.random.randn(nwalkers, ndim)

        if not self.resume:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_emcee)
            self.sampler.run_mcmc(pos, self.nstep, progress=True)
            self.samples = self.sampler.get_chain()
            self.log_prob = self.sampler.get_log_prob(flat=True)
            self.flat_samples = self.sampler.get_chain(flat=True)
            self.Data = np.hstack((np.reshape(self.log_prob, (-1, 1)), self.flat_samples))
            np.savetxt(self.prefix + '.txt', self.Data)

            fig, axes = plt.subplots(self.nparams, figsize=(10, 2 * self.nparams), sharex='all')
            for i in range(ndim):
                ax = axes[i]
                ax.plot(self.samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(self.samples))
                ax.set_ylabel(self.plabels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
            plt.savefig(self.savepath + 'sampler_walker.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
        
        self.Data = np.loadtxt(self.prefix + '.txt')
        self.log_prob = self.Data[self.discard:, 0]
        self.flat_samples = self.Data[self.discard:, 1:]
        self.weights = np.ones(len(self.flat_samples)) / len(self.flat_samples)
        self.equal_weighted_samples = np.hstack((self.flat_samples, np.reshape(self.log_prob, (-1, 1))))

        self.stats = []
        for i, p in enumerate(self.plabels):
            sigma123 = [50-99.73/2, 50-95.45/2, 50-68.27/2, 50, 50+68.27/2, 50+95.45/2, 50+99.73/2]
            lo3, lo2, lo1, med, hi1, hi2, hi3 = np.percentile(self.flat_samples[:, i], sigma123)
            self.stats.append({'median': med, '1sigma': [lo1, hi1], '2sigma': [lo2, hi2],
                               '3sigma': [lo3, hi3]})

        self.lnZ = None

        np.savetxt(self.prefix + 'post_equal_weights.dat', self.equal_weighted_samples)
        json.dump(self.stats, open(self.prefix + 'stats.json', 'w'), indent=4, cls=JsonEncoder)
