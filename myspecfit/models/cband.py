import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
# 10.1088/0004-637X/751/2/90


class cband(model):

    def __init__(self):
        self.expr = 'cband'
        self.redshift = 0
        self.comment = 'band function with cut-off'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha_1$'] = Parameter(1, -2, 2)
        self.pdicts['$\\alpha_2$'] = Parameter(-1, -2, 2)
        self.pdicts['log$E_{b}$'] = Parameter(1, 0, 3)
        self.pdicts['log$E_{p}$'] = Parameter(3, 1, 4)
        self.pdicts['log$A$'] = Parameter(0, -6, 6)


    def func(self, E, T=None):
        alpha1 = self.pdicts['$\\alpha_1$'].value
        alpha2 = self.pdicts['$\\alpha_2$'].value
        logEb = self.pdicts['log$E_{b}$'].value
        logEp = self.pdicts['log$E_{p}$'].value
        logA = self.pdicts['log$A$'].value

        Eb = 10 ** logEb
        Ep = 10 ** logEp
        A = 10 ** logA

        if alpha1 <= alpha2:
            return np.ones_like(E) * np.nan

        zi = 1 + self.redshift
        E = E * zi

        E2 = Ep / (2 + alpha2)
        E1 = 1 / (1 / E2 + (alpha1 - alpha2) / Eb)
        NE = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = E > Eb
        NE[i1] = A * E[i1] ** alpha1 * np.exp(- E[i1] / E1)
        NE[i2] = A * Eb ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) * E[i2] ** alpha2 * np.exp(- E[i2] / E2)
        return NE