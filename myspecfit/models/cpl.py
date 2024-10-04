import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


class cpl(model):

    def __init__(self):
        self.expr = 'cpl'
        self.redshift = 0
        self.comment = 'cutoff power law model'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -8, 4)
        self.pdicts['log$E_{c}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A$'] = Parameter(-1, -6, 5)


    def func(self, E, T=None):
        alpha = self.pdicts['$\\alpha$'].value
        logEc = self.pdicts['log$E_{c}$'].value
        logA = self.pdicts['log$A$'].value

        Ec = 10 ** logEc
        A = 10 ** logA

        zi = 1 + self.redshift
        E = E * zi

        NE = A * (E / 100) ** alpha * np.exp(-1.0 * E / Ec)
        return NE
