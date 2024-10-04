import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


class ppl(model):

    def __init__(self):
        self.expr = 'ppl'
        self.redshift = 0
        self.comment = 'cutoff power law model but replacing Ec with Ep'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -2, 2)
        self.pdicts['log$E_{p}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A$'] = Parameter(-1, -6, 5)


    def func(self, E, T=None):
        alpha = self.pdicts['$\\alpha$'].value
        logEp = self.pdicts['log$E_{p}$'].value
        logA = self.pdicts['log$A$'].value

        Ep = 10 ** logEp
        A = 10 ** logA

        zi = 1 + self.redshift
        E = E * zi

        Ec = Ep / (2 + alpha)
        NE = A * (E / 100) ** alpha * np.exp(-1.0 * E / Ec)
        return NE
    