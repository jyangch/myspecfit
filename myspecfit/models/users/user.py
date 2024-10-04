from ..model import model
from Tools import Parameter
from collections import OrderedDict
#+++++++editable area+++++++
# import other package
#+++++++editable area+++++++


class user(model):

    def __init__(self):
        #+++++++editable area+++++++
        # name of your model
        self.expr = 'user'
        # redshift of your model
        self.redshift = 0
        # comment of your model
        self.comment = 'user-defined model'
        #+++++++editable area+++++++

        self.pdicts = OrderedDict()

        #+++++++editable area+++++++
        # set your model parameters
        self.pdicts['$p_{1}$'] = Parameter(-1, -2, 0)
        # above sentence define a parameter $p_{1}$ 
        # with value of -1 and range of [-2, 0]
        #+++++++editable area+++++++


    def func(self, E, T):
        """
        Parameters
        ----------
        E: energy array in keV
        T: time array in second
        Returns
        -------
        NET: photon spectrum N(E, T) in photons/cm2/s/keV
        """

        #+++++++editable area+++++++
        # get the value of model parameter
        p1 = self.pdicts['$p_{1}$'].value
        #+++++++editable area+++++++

        zi = 1 + self.redshift
        E = E * zi

        #+++++++editable area+++++++
        # code your model to calculate N(E, T)
        NET = E ** p1
        #+++++++editable area+++++++

        return NET