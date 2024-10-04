from Tools import Parameter
from collections import OrderedDict


class model(object):

    def __init__(self):
        self.expr = 'model'
        self.redshift = 0
        self.comment = 'model is the base class of other models'

        self.pdicts = OrderedDict()
        self.pdicts['p'] = Parameter(1, 0, 2)


    def func(self, E, T):
        p = self.pdicts['p'].value
        f = p * E + T
        return f


    @property
    def info(self):
        print('> name: %s' % self.expr)
        print('> redshift: %s' % self.redshift)
        print('> comment: %s' % self.comment)
        print('> parameters:')
        for pl, pc in self.pdicts.items():
            print('  > %s' % pl)
            for key, value in pc.todict().items():
                print('    > %s: %s' % (key, value))


    def __add__(self, other):
        return composite_model(self, '+', other)
    

    def __sub__(self, other):
        return composite_model(self, '-', other)


    def __mul__(self, other):
        return composite_model(self, '*', other)


    def __truediv__(self, other):
        return composite_model(self, '/', other)



class composite_model(model):

    def __init__(self, m1, do, m2):
        self.expr = m1.expr + do + m2.expr
        self.comment = '\n'.joint(m1.comment + m2.comment)
        self.pdicts = OrderedDict()
        self.pdicts.update(m1.pdicts)
        self.pdicts.update(m2.pdicts)
