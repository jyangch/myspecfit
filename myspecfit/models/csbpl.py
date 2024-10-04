import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


class csbpl(model):

    def __init__(self):
        self.expr = 'csbpl'
        self.redshift = 0
        self.comment = 'smooth broken power law with cutoff'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha_{1}$'] = Parameter(1, -2, 2)
        self.pdicts['$\\alpha_{2}$'] = Parameter(-1, -2, 2)
        self.pdicts['log$E_{b}$'] = Parameter(1, 0, 3)
        self.pdicts['log$E_{p}$'] = Parameter(3, 1, 4)
        self.pdicts['log$A$'] = Parameter(0, -6, 6)


    def func(self, E, T=None):
        alpha1 = self.pdicts['$\\alpha_{1}$'].value
        alpha2 = self.pdicts['$\\alpha_{2}$'].value
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

        n = 5.38
        Ec = Ep / (2 + alpha2)
        NE = A * Eb ** alpha1 * ((E / Eb) ** (-alpha1 * n) + (E / Eb) ** (-alpha2 * n)) ** (-1 / n) * np.exp(-E / Ec)
        return NE