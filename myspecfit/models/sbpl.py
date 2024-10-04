import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
#Kaneko_ApJS_2006#


class sbpl(model):

    def __init__(self):
        self.expr = 'sbpl'
        self.redshift = 0
        self.comment = 'smoothly broken power law'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -4, 3)
        self.pdicts['$\\beta$'] = Parameter(-2, -5, 2)
        self.pdicts['log$E_{b}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A$'] = Parameter(-1, -6, 5)


    def func(self, E, T=None):
        alpha = self.pdicts['$\\alpha$'].value
        beta = self.pdicts['$\\beta$'].value
        logEb = self.pdicts['log$E_{b}$'].value
        logA = self.pdicts['log$A$'].value

        Eb = 10 ** logEb
        A = 10 ** logA

        zi = 1 + self.redshift
        E = E * zi

        Delta = 0.3
        m = (beta - alpha) / 2
        b = (alpha + beta) / 2
        q = np.log10(E / Eb) / Delta
        qpiv = np.log10(100 / Eb) / Delta
        a = m * Delta * np.log((np.e ** q + np.e ** (-q)) / 2)
        apiv = m * Delta * np.log((np.e ** qpiv + np.e ** (-qpiv)) / 2)
        NE = A * (E / 100) ** b * 10 ** (a - apiv)
        return NE