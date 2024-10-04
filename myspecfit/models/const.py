import numpy as np
from .model import model
from Tools import Parameter
from collections import OrderedDict


class const(model):

    def __init__(self):
        self.expr = 'const'
        self.redshift = 0
        self.comment = 'const model'

        self.pdicts = OrderedDict()
        self.pdicts['$C$'] = Parameter(1, -10, 10)


    def func(self, E, T=None):
        C = self.pdicts['$C$'].value
        NE = np.ones_like(E, dtype=float) * C
        return NE