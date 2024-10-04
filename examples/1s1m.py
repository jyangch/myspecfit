from myspecfit import *

group1 = {'src': './gbm/na_pha.fits', 'bkg': './gbm/na_bak.fits', 'rsp': './gbm/na_resp.rsp'}
group2 = {'src': './gbm/b1_pha.fits', 'bkg': './gbm/b1_bak.fits', 'rsp': './gbm/b1_resp.rsp'}

spec = Spectrum()
spec.set('GBM-na', group1, nt=[8, 900], stat='pgstat')
spec.set('GBM-b1', group2, nt=[300, 38000], stat='pgstat')

mo = Model()
mo.set(expr='ppl')

fit = Fit()
fit.set(spec, mo)
fit.run('multinest', './1s1m', nlive=400)

ana = Analyse(fit)
ana.post()
ana.corner()

plot = Plot(ana)
plot.rebin()
plot.cspec()
plot.pspec()
plot.mspec([{'expr': 'ppl'}])

cal = Calculate(ana)
cal.flux('ppl')
