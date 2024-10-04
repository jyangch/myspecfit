import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


class band(model):

    def __init__(self):
        self.expr = 'band'
        self.redshift = 0
        self.comment = 'grb band function'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -2, 2)
        self.pdicts['$\\beta$'] = Parameter(-4, -6, -2)
        self.pdicts['log$E_{p}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A$'] = Parameter(-1, -6, 5)


    def func(self, E, T=None):
        alpha = self.pdicts['$\\alpha$'].value
        beta = self.pdicts['$\\beta$'].value
        logEp = self.pdicts['log$E_{p}$'].value
        logA = self.pdicts['log$A$'].value

        Ep = 10 ** logEp
        A = 10 ** logA
        
        zi = 1 + self.redshift
        E = E * zi

        Ec = Ep / (2 + alpha)
        Eb = (alpha - beta) * Ec
        NE = np.zeros_like(E, dtype=float)

        i1 = E <= Eb; i2 = E > Eb
        NE[i1] = A * (E[i1] / 100) ** alpha * np.exp(-E[i1] / Ec)
        NE[i2] = A * (Eb / 100) ** (alpha - beta) * (E[i2] / 100) ** beta * np.exp(beta - alpha)
        return NE