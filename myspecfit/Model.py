import re
import os
import json
import numpy as np
from models import *
from collections import OrderedDict
from Tools import JsonEncoder, Parameter, oper_model


class Model(object):

    def __init__(self):
        self.clear()


    def clear(self):
        self.pdicts = None
        self.fdicts = None
        self.spectrum = None

        self.expr = None
        self.expr_sp = None
        self.mode = None


    def set(self, expr=None, obje=None):
        self.pdicts, self.fdicts = OrderedDict(), OrderedDict()
        if obje is not None:
            if type(obje) is not list:
                self.expr = obje.expr
                self.expr_sp = [self.expr]
                for key, value in obje.pdicts.items(): self.pdicts['.'.join([self.expr, key])] = value
                self.fdicts[self.expr] = obje.func
            else:
                assert expr is not None, 'you should provide expr argument to describe ' \
                                         'how to perform calculations on these models!'
                self.expr = re.sub('\s*', '', expr)
                self.expr_sp = re.split(r"[(+\-*/)]", self.expr)
                self.expr_sp = [ex for ex in self.expr_sp if ex != '']
                assert set(self.expr_sp) == set([oi.expr for oi in obje]), expr + ' is not consistent with your obje!'
                for oi in obje:
                    for key, value in oi.pdicts.items():
                        assert '.'.join([oi.expr, key]) not in self.pdicts, 'there are same parameter labels!'
                        self.pdicts['.'.join([oi.expr, key])] = value
                    self.fdicts[oi.expr] = oi.func
        else:
            assert expr is not None, 'you should provide expr argument to describe the model operation!'
            self.expr = re.sub('\s*', '', expr)
            self.expr_sp = re.split(r"[(+\-*/)]", self.expr)
            self.expr_sp = [ex for ex in self.expr_sp if ex != '']
            if len(self.expr_sp) == 1:
                obje = self.set_mo(self.expr)
                for key, value in obje.pdicts.items(): self.pdicts['.'.join([self.expr, key])] = value
                self.fdicts[self.expr] = obje.func
            else:
                obje = [self.set_mo(ex) for ex in self.expr_sp]
                for oi in obje:
                    for key, value in oi.pdicts.items(): 
                        assert '.'.join([oi.expr, key]) not in self.pdicts, 'there are same parameter labels!'
                        self.pdicts['.'.join([oi.expr, key])] = value
                    self.fdicts[oi.expr] = oi.func
        self.info


    @property
    def info(self):
        if self.expr is None:
            print('+-----------------------------------------------+')
            print(' model has not been determined!')
            print('+-----------------------------------------------+\n')
        else:
            print('+-----------------------------------------------+')
            print(' model name: ' + self.expr)
            print('+-----------------------------------------------+')
            print(' %-25s%-25s' % ('plabel', 'prange'))
            print('+-----------------------------------------------+')
            _ = list(map(print, [' %-25s%-25s' % (pl_, pr_.range) for pl_, pr_ in self.pdicts.items()]))
            print('+-----------------------------------------------+\n')


    @property
    def mo_dict(self):
        return {'band': band, 
                'bb': bb, 
                'cpl': cpl,
                'ppl': ppl, 
                'pl': pl, 
                'mbb': mbb, 
                'cband': cband, 
                'dband': dband, 
                'sbpl': sbpl, 
                'ssbpl': ssbpl, 
                'csbpl': csbpl, 
                'dsbpl': dsbpl, 
                'phabs': phabs, 
                'tbabs': tbabs, 
                'hlecpl': hlecpl, 
                'hleband': hleband, 
                'const': const}


    def set_mo(self, expr):
        if expr not in self.mo_dict:
            raise ValueError('invalid value for expr!')
        else:
            return self.mo_dict[expr]()


    def frozen(self, plabel, pvalue):
        assert self.expr is not None, 'please set a model!'
        assert plabel in self.pdicts, plabel + ' is not one of model parameter labels!'
        self.pdicts[plabel].frozen_at(pvalue)
        self.info


    def limit(self, plabel, pmin, pmax):
        assert self.expr is not None, 'please set a model!'
        assert plabel in self.pdicts, plabel + ' is not one of model parameter labels!'
        self.pdicts[plabel].limit_in(pmin, pmax)
        self.info


    def replace(self, rkey, key=None, value=None):
        assert self.expr is not None, 'please set a model!'
        assert rkey in self.pdicts, rkey + ' is not one of model parameter labels!'
        if key is not None:
            pls = [key if pl_ == rkey else pl_ for pl_ in self.pdicts.keys()]
            prs = list(self.pdicts.values())
            self.pdicts = OrderedDict(list(zip(pls, prs)))
            if value is not None:
                assert isinstance(value, Parameter), 'value should be Parameter type!'
                self.pdicts[key] = value
        else:
            if value is not None:
                assert isinstance(value, Parameter), 'value should be Parameter type!'
                self.pdicts[rkey] = value
        self.info


    def fit_to(self, spectrum):
        from Spectrum import Spectrum
        assert isinstance(spectrum, Spectrum), 'spectrum parameter should be Spectrum type!'
        self.spectrum = spectrum
        spectrum.model = self


    def eval_model(self):
        self.mdicts = OrderedDict()
        for ex, fu in self.fdicts.items():
            if 'phabs' in ex or 'tbabs' in ex:
                eval_mo_ = fu(self.spectrum.E_coord, self.spectrum.T_coord)
                eval_mo = [np.mean(eval_mo_[i:j]) for (i, j) in zip(self.spectrum.ET_start, self.spectrum.ET_stop)]
                self.mdicts[ex] = np.array(eval_mo)
            else:
                eval_mo_ = fu(self.spectrum.E_coord, self.spectrum.T_coord)
                eval_mo = [np.mean(eval_mo_[i:j]) for (i, j) in zip(self.spectrum.ET_start, self.spectrum.ET_stop)]
                self.mdicts[ex] = np.array(eval_mo) * np.array(self.spectrum.ET_width)
        self.mode = oper_model(self.expr, self.mdicts)


    def conv_drm(self):
        self.eval_model()
        self.mode_by_spec = [self.mode[i:j] for (i, j) in zip(self.spectrum.ET_Start, self.spectrum.ET_Stop)]
        self.morate = [np.dot(mi, rsp.drm * rf.value) for (mi, rsp, rf) in zip(self.mode_by_spec, self.spectrum.rsp_info, self.spectrum.rfs)]
        return self.morate


    def func(self, Er, T, expr=None):
        assert self.expr is not None, 'please set a model!'
        E = np.array([np.sqrt(Eri[0] * Eri[1]) for Eri in Er])

        ne, ene, e2ne = {}, {}, {}
        for ex, fu in self.fdicts.items():
            if 'phabs' in ex or 'tbabs' in ex:
                # ne[ex] = np.array(fu(Er, T))
                ne[ex] = np.array(fu(E, T))
                ene[ex] = ne[ex]; e2ne[ex] = ne[ex]
            else:
                ne[ex] = np.array(fu(E, T))
                ene[ex] = 1.60218e-9 * E * ne[ex]
                e2ne[ex] = 1.60218e-9 * E * E * ne[ex]

        if expr is not None:
            expr = re.sub('\s*', '', expr)
            expr_sp = re.split(r"[(+\-*/)]", expr)
            expr_sp = [ex for ex in expr_sp if ex != '']
            assert set(expr_sp) <= set(self.expr_sp), expr + ' is not present!'

            ne[expr] = oper_model(expr, ne)
            ene[expr] = oper_model(expr, ene)
            e2ne[expr] = oper_model(expr, e2ne)

        return ne, ene, e2ne
    

    def save(self, savepath):
        assert self.expr is not None, 'please set a model!'

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json.dump(self.expr, open(savepath + 'expr.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.expr_sp, open(savepath + 'expr_sp.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.pdicts, open(savepath + 'pdicts.json', 'w'), indent=4, cls=JsonEncoder)
