import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


# # using sherpa.xspec
# from sherpa.astro import xspec


# class tbabs(model):

#     def __init__(self):
#         self.expr = 'tbabs'
#         self.redshift = 0
#         self.comment = 'Wilms absorption model'

#         self.pdicts = OrderedDict()
#         self.pdicts['$n_H$'] = Parameter(1, 0, 10)  # in 10^22 cm^-2

#         xspec.set_xsabund('wilm')
#         xspec.set_xsxsect('vern')
#         self.abund = xspec.get_xsabund()
#         self.xsect = xspec.get_xsxsect()
#         self.TBabs = xspec.XSTBabs()
#         self.zTBabs = xspec.XSzTBabs()
#         self.TBabs.integrate = False
#         self.zTBabs.integrate = False


#     def func(self, Er, T=None):
#         nH = self.pdicts['$n_H$'].value
#         z = self.redshift

#         assert self.abund == 'wilm', 'abundance table need set to be wilm.'
#         assert self.xsect == 'vern', 'cross-section table need set to be vern.'

#         elo, ehi = Er[:, 0], Er[:, 1]

#         if z == 0:
#             self.TBabs.nH = nH
#             frac = self.TBabs(elo, ehi)
#         else:
#             self.zTBabs.nH = nH
#             self.zTBabs.redshift = z
#             frac = self.zTBabs(elo, ehi)
#         return frac



# using astromodels.TbAbs
from astromodels import TbAbs


class tbabs(model):

    def __init__(self):
        self.expr = 'tbabs'
        self.redshift = 0
        self.comment = 'Wilms absorption model'

        self.pdicts = OrderedDict()
        self.pdicts['$n_H$'] = Parameter(1, 0.0001, 10)  # in 10^22 cm^-2

        self.TbAbs = TbAbs()


    def func(self, E, T=None):
        nH = self.pdicts['$n_H$'].value
        z = self.redshift

        self.TbAbs.NH = nH
        self.TbAbs.redshift = z

        frac = self.TbAbs(np.array(E, dtype=float))
        return frac
