import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
from scipy.special import gamma, zeta


class mbb(model):

    def __init__(self):
        self.expr = 'mbb'
        self.redshift = 0
        self.comment = 'multi-color black-body model'

        self.pdicts = OrderedDict()
        self.pdicts['log$kT_{min}$'] = Parameter(1, 0, 2)
        self.pdicts['log$kT_{max}$'] = Parameter(3, 1, 4)
        self.pdicts['$m$'] = Parameter(0, -2, 2)
        self.pdicts['log$A$'] = Parameter(0, -10, 10)


    def func(self, E, T=None):
        logkTmin = self.pdicts['log$kT_{min}$'].value
        logkTmax = self.pdicts['log$kT_{max}$'].value
        m = self.pdicts['$m$'].value
        logA = self.pdicts['log$A$'].value

        kTmin = 10 ** logkTmin
        kTmax = 10 ** logkTmax
        A = 10 ** logA

        zi = 1 + self.redshift
        E = E * zi

        item1 = 1 / (2 - m) * (E / kTmin)
        item2 = gamma(3 - m) * zeta(3 - m) * (E / kTmin) ** (m - 1)
        item3 = (kTmin / kTmax) ** (2 - m) * (E / kTmin) * np.exp(-E / kTmax)
        IE = item1 * (E < kTmin) + item2 * ((E >= kTmin) & (E <= kTmax)) + item3 * (E > kTmax)
        NE = A * (m + 1) / ((kTmax / kTmin) ** (m + 1) - 1) * kTmin ** (-2) * IE
        return NE
