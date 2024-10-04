import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
#Ravasio_A&A_2018#


class ssbpl(model):

    def __init__(self):
        self.expr = 'ssbpl'
        self.redshift = 0
        self.comment = 'single smoothly broken power law'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -4, 3)
        self.pdicts['$\\beta$'] = Parameter(-3, -5, -2)
        self.pdicts['log$E_{p}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A$'] = Parameter(0, -6, 6)


    def func(self, E, T=None):
        alpha = self.pdicts['$\\alpha$'].value
        beta = self.pdicts['$\\beta$'].value
        logEp = self.pdicts['log$E_{p}$'].value
        logA = self.pdicts['log$A$'].value

        Ep = 10 ** logEp
        A = 10 ** logA

        if alpha <= beta:
            return np.ones_like(E) * np.nan

        zi = 1 + self.redshift
        E = E * zi

        n = 2.69
        Ej = Ep * (-(alpha + 2) / (beta + 2)) ** (1 / ((beta - alpha) * n))
        NE = A * Ej ** alpha * ((E / Ej) ** (-alpha * n) + (E / Ej) ** (-beta * n)) ** (-1 / n)
        return NE
