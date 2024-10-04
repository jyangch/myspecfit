import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
# 10.1088/0004-637X/751/2/90


class dband(model):

    def __init__(self):
        self.expr = 'dband'
        self.redshift = 0
        self.comment = 'double band functions'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha_{1}$'] = Parameter(1, -2, 2)
        self.pdicts['$\\alpha_{2}$'] = Parameter(-1, -2, 2)
        self.pdicts['$\\beta$'] = Parameter(-3, -5, -2)
        self.pdicts['log$E_{b}$'] = Parameter(1, 0, 3)
        self.pdicts['log$E_{p}$'] = Parameter(3, 1, 4)
        self.pdicts['log$A$'] = Parameter(0, -6, 6)


    def func(self, E, T=None):
        alpha1 = self.pdicts['$\\alpha_{1}$'].value
        alpha2 = self.pdicts['$\\alpha_{2}$'].value
        beta = self.pdicts['$\\beta$'].value
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

        E2 = Ep / (alpha2 - beta)
        E1 = 1 / (1 / E2 + (alpha1 - alpha2) / Eb)
        NE = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = (E > Eb) & (E <= Ep); i3 = E > Ep
        NE[i1] = A * E[i1] ** alpha1 * np.exp(- E[i1] / E1)
        NE[i2] = A * Eb ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) * E[i2] ** alpha2 * np.exp(- E[i2] / E2)
        NE[i3] = A * Eb ** (alpha1 - alpha2) * np.exp(alpha2 - alpha1) * ((alpha2 - beta) * E2) ** (alpha2 - beta) * np.exp(beta - alpha2) * E[i3] ** beta
        return NE