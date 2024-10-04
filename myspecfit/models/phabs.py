import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


# # using sherpa.xspec
# from sherpa.astro import xspec


# class phabs(model):

#     def __init__(self):
#         self.expr = 'phabs'
#         self.redshift = 0
#         self.comment = 'photon-electron absorption model'

#         self.pdicts = OrderedDict()
#         self.pdicts['$n_H$'] = Parameter(1, 0, 3)  # in 10^22 cm^-2

#         xspec.set_xsabund('angr')
#         xspec.set_xsxsect('vern')
#         self.abund = xspec.get_xsabund()
#         self.xsect = xspec.get_xsxsect()
#         self.PHabs = xspec.XSphabs()
#         self.zPHabs = xspec.XSzphabs()
#         self.PHabs.integrate = False
#         self.zPHabs.integrate = False


#     def func(self, Er, T=None):
#         nH = self.pdicts['$n_H$'].value
#         z = self.redshift

#         assert self.abund == 'angr', 'abundance table need set to be angr.'
#         assert self.xsect == 'vern', 'cross-section table need set to be vern.'

#         elo, ehi = Er[:, 0], Er[:, 1]

#         if z == 0:
#             self.PHabs.nH = nH
#             frac = self.PHabs(elo, ehi)
#         else:
#             self.zPHabs.nH = nH
#             self.zPHabs.redshift = z
#             frac = self.zPHabs(elo, ehi)
#         return frac



# using astromodels.PhAbs
from astromodels import PhAbs


class phabs(model):

    def __init__(self):
        self.expr = 'phabs'
        self.redshift = 0
        self.comment = 'photon-electron absorption model'

        self.pdicts = OrderedDict()
        self.pdicts['$n_H$'] = Parameter(1, 0, 3)  # in 10^22 cm^-2

        self.PhAbs = PhAbs()


    def func(self, E, T=None):
        nH = self.pdicts['$n_H$'].value
        z = self.redshift

        self.PhAbs.NH = nH
        self.PhAbs.redshift = z

        frac = self.PhAbs(np.array(E, dtype=float))
        return frac
