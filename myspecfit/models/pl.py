from .model import model
from Tools import Parameter
from collections import OrderedDict


class pl(model):

    def __init__(self):
        self.expr = 'pl'
        self.redshift = 0
        self.comment = 'power law model'

        self.pdicts = OrderedDict()
        self.pdicts['$\\alpha$'] = Parameter(-1, -8, 5)
        self.pdicts['log$A$'] = Parameter(-1, -10, 8)


    def func(self, E, T=None):
        alpha = self.pdicts['$\\alpha$'].value
        logA = self.pdicts['log$A$'].value
        
        A = 10 ** logA

        zi = 1 + self.redshift
        E = E * zi

        NE = A * (E / 100) ** alpha
        return NE
