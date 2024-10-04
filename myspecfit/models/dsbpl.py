import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
#Ravasio_A&A_2018#


class dsbpl(model):

    def __init__(self):
        self.expr = 'dsbpl'
        self.redshift = 0
        self.comment = 'double smooth broken power laws'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha_{1}$'] = Parameter(1, -2, 2)
        self.pdicts['$\\alpha_{2}$'] = Parameter(-1, -2, 2)
        self.pdicts['$\\beta$'] = Parameter(-4, -6, -2)
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

        n1 = 5.38
        n2 = 2.69
        Ej = Ep * (-(alpha2 + 2) / (beta + 2)) ** (1 / ((beta - alpha2) * n2))
        sbpl1 = ((E / Eb) ** (-alpha1 * n1) + (E / Eb) ** (-alpha2 * n1)) ** (n2 / n1)
        sbpl2 = ((Ej / Eb) ** (-alpha1 * n1) + (Ej / Eb) ** (-alpha2 * n1)) ** (n2 / n1)
        NE = A * Eb ** alpha1 * (sbpl1 + (E / Ej) ** (-beta * n2) * sbpl2) ** (-1 / n2)
        return NE
