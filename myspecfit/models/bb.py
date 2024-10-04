import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


class bb(model):

    def __init__(self):
        self.expr = 'bb'
        self.redshift = 0
        self.comment = 'black-body model'

        self.pdicts = OrderedDict()
        self.pdicts['log$kT$'] = Parameter(2, 0, 3)
        self.pdicts['log$A$'] = Parameter(-1, -6, 5)


    def func(self, E, T=None):
        logKT = self.pdicts['log$kT$'].value
        logA = self.pdicts['log$A$'].value

        kT = 10 ** logKT
        A = 10 ** logA

        zi = 1 + self.redshift
        E = E * zi

        NE = A*8.0525*E**2/(kT**4*(np.exp(E/kT)-1))
        return NE
