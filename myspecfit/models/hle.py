import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict
# 10.1088/0004-637X/690/1/L10


class hleband(model):
    
    def __init__(self):
        self.expr = 'hleband'
        self.redshift = 0
        self.comment = 'curvature effect model for band function'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -2, 2)
        self.pdicts['$\\beta$'] = Parameter(-4, -6, -2)
        self.pdicts['log$E_{p,c}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A_{c}$'] = Parameter(0, -6, 6)
        self.pdicts['$t_{0}$'] = Parameter(0, -20, 20)
        self.pdicts['$t_{c}$'] = Parameter(10, 0, 50)


    def func(self, E, T):
        alpha = self.pdicts['$\\alpha$'].value
        beta = self.pdicts['$\\beta$'].value
        logEpc = self.pdicts['log$E_{p,c}$'].value
        logAc = self.pdicts['log$A_{c}$'].value
        t0 = self.pdicts['$t_{0}$'].value
        tc = self.pdicts['$t_{c}$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc

        if tc <= t0 or tc > np.min(T):
            return np.ones_like(E) * np.nan

        zi = 1 + self.redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)
        Ebt = (alpha - beta) / (alpha + 2) * Ept
        NEt = np.zeros_like(E, dtype=float)

        i1 = E < Ebt; i2 = E >= Ebt
        NEt[i1] = At[i1] * (E[i1] / 100) ** alpha * np.exp(-(2 + alpha) * E[i1] / Ept[i1])
        NEt[i2] = At[i2] * (Ebt[i2] / 100) ** (alpha - beta) * (E[i2] / 100) ** beta * np.exp(beta - alpha)
        return NEt



class hlecpl(model):
    
    def __init__(self):
        self.expr = 'hlecpl'
        self.redshift = 0
        self.comment = 'curvature effect model for cpl function'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -2, 2)
        self.pdicts['log$E_{p,c}$'] = Parameter(2, 0, 4)
        self.pdicts['log$A_{c}$'] = Parameter(0, -6, 6)
        self.pdicts['$t_{0}$'] = Parameter(0, -20, 20)
        self.pdicts['$t_{c}$'] = Parameter(10, 0, 50)


    def func(self, E, T):
        alpha = self.pdicts['$\\alpha$'].value
        logEpc = self.pdicts['log$E_{p,c}$'].value
        logAc = self.pdicts['log$A_{c}$'].value
        t0 = self.pdicts['$t_{0}$'].value
        tc = self.pdicts['$t_{c}$'].value

        Epc = 10 ** logEpc
        Ac = 10 ** logAc

        if tc <= t0 or tc > np.min(T):
            return np.ones_like(E) * np.nan

        zi = 1 + self.redshift
        E = E * zi

        Ept = Epc * ((T - t0) / (tc - t0)) ** (- 1)
        At = Ac * ((T - t0) / (tc - t0)) ** (alpha - 1)

        NEt = At * (E / 100) ** alpha * np.exp(-(2 + alpha) * E / Ept)
        return NEt
