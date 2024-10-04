from myspecfit import *

group1 = {'src': './leia/grb.src', 
          'bkg': './leia/grb.bkg', 
          'rmf': './leia/grb.rmf', 
          'arf': './leia/grb.arf'}
spec1 = Spectrum()
spec1.set('LEIA', group1, nt=[0.5, 4.0], stat='cstat', gr={'min_evt': 10})

group2 = {'src': './gbm/na_pha.fits', 
          'bkg': './gbm/na_bak.fits', 
          'rsp': './gbm/na_resp.rsp'}
group3 = {'src': './gbm/b1_pha.fits', 
          'bkg': './gbm/b1_bak.fits', 
          'rsp': './gbm/b1_resp.rsp'}
group4 = {'src': './gecam/b/hg_pha.fits', 
          'bkg': './gecam/b/hg_bak.fits', 
          'rsp': './gecam/b/hg_resp.rsp'}
group5 = {'src': './gecam/b/lg_pha.fits', 
          'bkg': './gecam/b/lg_bak.fits', 
          'rsp': './gecam/b/lg_resp.rsp'}
group6 = {'src': './gecam/c/gcg.src', 
          'bkg': './gecam/c/gcg.bkg', 
          'rsp': './gecam/c/gcg.rsp'}
spec2 = Spectrum()
spec2.set('GBM-na', group2, nt=[8, 900], stat='pgstat', gr={'min_sigma': 3, 'max_bin': 10})
spec2.set('GBM-b1', group3, nt=[300, 38000], stat='pgstat', gr={'min_sigma': 3, 'max_bin': 10})
spec2.set('GECAM-BHG', group4, nt=[40, 350], stat='pgstat', gr={'min_sigma': 3, 'max_bin': 10})
spec2.set('GECAM-BLG', group5, nt=[700, 6000], stat='pgstat', gr={'min_sigma': 3, 'max_bin': 10})
spec2.set('GECAM-CHG', group6, nt=[[15, 35], [42, 100]], stat='pgstat', sii=4, bii=4, gr={'min_sigma': 3, 'max_bin': 10})

tbabs_ = tbabs()
tbabs_.expr = 'tbabs'
tbabs_.redshift = 0

ztbabs_ = tbabs()
ztbabs_.expr = 'ztbabs'
ztbabs_.redshift = 0.065

ppl_ = ppl()

mo1 = Model()
mo1.set(expr='tbabs * ztbabs * ppl', obje=[tbabs_, ztbabs_, ppl_])
mo1.frozen('tbabs.$n_H$', 0.126)
mo1.frozen('ztbabs.$n_H$', 0.392)

mo2 = Model()
mo2.set(obje=ppl_)

fit = Fit()
fit.set(spec1, mo1)
fit.set(spec2, mo2)
fit.run('multinest', './2s2m', nlive=300)

ana = Analyse(fit)
ana.post()
ana.corner()

plot = Plot(ana)
plot.rebin()
plot.cspec()
plot.pspec()
plot.mspec([{'expr': 'ppl'}, {'expr': 'tbabs * ztbabs * ppl'}])

cal = Calculate(ana)
cal.flux('ppl')
