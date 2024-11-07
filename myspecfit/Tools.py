import re
import json
import ctypes
import numpy as np
import streamlit as st
from io import BytesIO
import subprocess as sp
from Significance import ppsig, pgsig
from os.path import isfile, dirname, abspath


def transpose(data):
    if len(data[0]) == 0:
        return data
    trans_data = [[row[col] for row in data] for col in range(len(data[0]))]
    return trans_data


def fill2D(data, fill_value='--'):
    if type(data[0]) is not list and (type(data[0]) is not np.ndarray):
        print('ERROR: data should be 2-D list or array!')
        return False
    nrow = len(data)
    ncol = max([len(data[i]) for i in range(nrow)])
    data = [[row[i] if i < len(row) else fill_value for i in range(ncol)] for row in data]
    return data


def pop2D(data, pop_value='--'):
    if type(data[0]) is not list and (type(data[0]) is not np.ndarray):
        print('ERROR: data should be 2-D list or array!')
        return False
    nrow = len(data)
    ncol = len(data[0])
    data = [[data[r][c] for c in range(ncol) if data[r][c] != pop_value] for r in range(nrow)]
    return data


def savetxt(file, data, fmt=None, trans=False, header=None):
    if data is None:
        data = [[]]
    elif len(data) == 0:
        data = [[]]
    elif (type(data[0]) is not list) and (type(data[0]) is not np.ndarray):
        print('Warning: data is 1-D array or list!')
        data = [data]

    data = fill2D(data)
    if trans: data = transpose(data)
    row = len(data)
    col = len(data[0])

    if col == 0:
        f = open(file, 'w+')
        _ = [f.write(''.join(data[r]) + '\n') for r in range(row)]
        f.close()
        return True

    if fmt is None:
        fmt = [['s'] * col] * row
    elif type(fmt) is str:
        fmt = [[fmt] * col] * row
    elif (type(fmt) is list) and (type(fmt[0]) is not list):
        if len(fmt) != col:
            print('ERROR: the fmt lenth should equal to col(after trans)!')
            return False
        else:
            fmt = [fmt] * row
    elif (type(fmt) is list) and (type(fmt[0]) is list):
        if trans: fmt = transpose(fmt)
        if len(fmt) != row or len(fmt[0]) != col:
            print('ERROR: the fmt shape should be same with data!')
            return False
    else:
        print('ERROR: wrong fmt!')

    if header is None:
        header = []
    else:
        if len(header) != col:
            print('ERROR: the header lenth should equal to col(after trans)!')

    header = ['%s'%h for h in header]
    data = [['--' if data[r][c] == '--' else ('%'+fmt[r][c])%data[r][c] for c in range(col)] for r in range(row)]

    length = max([len(data[r][c]) for r in range(row) for c in range(col)] + [len(h) for h in header])
    header = [h + ' ' * (length + 4 - len(h)) for h in header]
    data = [[data[r][c] + ' ' * (length + 4 - len(data[r][c])) for c in  range(col)] for r in range(row)]

    f = open(file, 'w+')
    f.write('' if len(header)==0 else ''.join(header) + '\n')
    _ = [f.write(''.join(data[r]) + '\n') for r in range(row)]
    f.close()
    return True


def loadtxt(file, fmt=None, trans=False, underline=True):
    data = []
    f = open(file, 'r')
    for line in f:
        data.append([i for i in re.split(r'[\t\n\s]', line) if i != ''])
    f.close()

    row = len(data)
    col = len(data[0])

    if col == 0:
        return data

    for r in range(row):
        for c in range(col):
            try:
                data[r][c] = int(data[r][c])
            except ValueError:
                try:
                    data[r][c] = float(data[r][c])
                except ValueError:
                    data[r][c] = data[r][c]

    if fmt is not None:
        if type(fmt) is str:
            fmt = [[fmt] * col] * row
        elif (type(fmt) is list) and (type(fmt[0]) is not list):
            if len(fmt) != col:
                print('ERROR: the fmt lenth should equal to col(after trans)!')
                return False
            else:
                fmt = [fmt] * row
        elif (type(fmt) is list) and (type(fmt[0]) is list):
            if len(fmt) != row or len(fmt[0]) != col:
                print('ERROR: the fmt shape should be same with data!')
                return False
        else:
            print('ERROR: wrong fmt!')

        for r in range(row):
            for c in range(col):
                try:
                    data[r][c] = ('%' + fmt[r][c]) % data[r][c]
                except TypeError:
                    print('format %s is not suitable for data %s'%(fmt[r][c], data[r][c]))
    if trans:
        data = transpose(data)
        row = len(data)
        col = len(data[0])
    if not underline:
        data = [[data[r][c] for c in range(col) if data[r][c] != '--'] for r in range(row)]
    return data


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.
    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range [0, 1].
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These
    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at 'q'.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


def listidx(list0, idx):
    assert type(list0) is list0, '"list" should be a list!'
    assert type(idx) is list0, '"idx" should be a list!'
    list1 = [list0[i] for i in idx]
    return list1


def copy(f1, f2):
    if isfile(f1):
        sp.call('cp -rf ' + f1 + ' ' + f2, shell=True)
    else:
        print('\n-------------------------------------------------')
        print('FILE NOT FOUND: ' + f1)
        print('-------------------------------------------------')
        f2_name = f2.split('/')[-1]
        sp.call('touch ' + dirname(f2) + '/%s_Not_Found.txt'%f2_name, shell=True)


def intersection(A, B):
    #A = [[0,2], [5,10], [13,23], [24,25]]
    #B = [[1,5], [8,12], [15,24], [25,26]]
    #--------------
    #sort A and B
    #--------------
    A1 = np.array([i[-1] for i in A])
    B1 = np.array([i[-1] for i in B])
    A = np.array(A)[np.argsort(A1)]
    B = np.array(B)[np.argsort(B1)]

    i, j = 0, 0
    res = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i][0], A[i][1]
        b1, b2 = B[j][0], B[j][1]
        if b2 > a1 and a2 > b1:
            res.append([max(a1, b1), min(a2, b2)])
        if b2 < a2: j += 1
        else: i += 1
    return res


def union(bins):
    if len(bins) == 0:
        return []
    #--------------
    #sort bins
    #--------------
    bins1 = np.array([bin_[0] for bin_ in bins])
    bins = np.array(bins)[np.argsort(bins1)]
    bins = bins.tolist()

    res = [bins[0]]
    for i in range(1, len(bins)):
        a1, a2 = res[-1][0], res[-1][1]
        b1, b2 = bins[i][0], bins[i][1]
        if b2 >= a1 and a2 >= b1:
            res[-1] = [min(a1, b1), max(a2, b2)]
        else: res.append(bins[i])

    return res


def oper_model(expr, mdicts):
    expr_sp = re.split(r"([+\-*/])", expr)
    if len(expr_sp) == 1:
        mode = mdicts[expr]
    else:
        mode = eval(expr, {}, mdicts)
    return mode


def flag_grouping(
    s, 
    b, 
    berr, 
    ts, 
    tb, 
    ss, 
    sb, 
    min_sigma=None, 
    min_evt=None, 
    max_bin=None, 
    stat=None, 
    ini_flag=None
    ):
    
    # grouping flag:
    # grpg = 0 if the channel is not allowed to group, including the not qualified noticed channels
    # grpg = +1 if the channel is the start of a new bin
    # grpg = -1 if the channel is part of a continuing bin
    
    if ini_flag is None:
        ini_flag = [1] * len(s)
        
    if min_sigma is None:
        min_sigma = -np.inf
    
    if min_evt is None:
        min_evt = 0
        
    if max_bin is None:
        max_bin = np.inf
        
    alpha = ts * ss / (tb * sb)

    flag, gs = [], []
    nowbin = False
    cs, cb, cberr, cp = 0, 0, 0, 0
    for i in range(len(s)):
        if ini_flag[i] != 1:
            flag.append(0)
            if nowbin:
                if len(gs) < 2:
                    pass
                else:
                    flag[gs[-1]] = -1
            nowbin = False
            cs, cb, cberr, cp = 0, 0, 0, 0
        else:
            if not nowbin:
                flag.append(1)
                gs.append(i)
                cp = 1
            else:
                flag.append(-1)
                cp += 1

            si = s[i]
            bi = b[i]
            bierr = berr[i]
            cs += si
            cb += bi
            cberr = np.sqrt(cberr ** 2 + bierr ** 2)
            
            if stat is None: stat = 'pgstat'
            if stat == 'cstat':
                if (cb < 0 or cs < 0) and (cb != cs):
                    sigma = 0
                else:
                    sigma = ppsig(cs, cb, alpha)
            elif stat == 'pgstat':
                if cs <= 0 or cberr == 0:
                    sigma = 0
                else:
                    sigma = pgsig(cs, cb * alpha, cberr * alpha)
            else:
                raise AttributeError(f'unsupported stat: {stat}')
            
            evt = cs - cb * alpha
            
            if ((sigma >= min_sigma) and (evt >= min_evt)) or cp == max_bin:
                nowbin = False
                cs, cb, cberr, cp = 0, 0, 0, 0
            else:
                nowbin = True

            if nowbin and i == (len(s) - 1):
                if len(gs) < 2:
                    pass
                else:
                    flag[gs[-1]] = -1

    return np.array(flag)


def foo(id):
    return ctypes.cast(id, ctypes.py_object).value


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Parameter):
            return obj.todict()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, BytesIO):
            return obj.name
        else:
            return super(JsonEncoder, self).default(obj)


class Parameter(object):

    def __init__(self, val=None, min=None, max=None, comment=None, scale='linear', unit=1, frozen=False):
        self._val = val
        self.min = min
        self.max = max
        self.comment = comment
        self.scale = scale
        self.unit = unit
        self.frozen = frozen
        self.mates = set()

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, new_val):
        self._val = new_val
        for mate in self.mates:
            if mate.val != self.val:
                mate.val = self.val

    def link(self, other):
        assert isinstance(other, Parameter)
        self.mates.add(other)
        other.mates.add(self)
        self.val = self._val

    def frozen_at(self, new_val):
        self.frozen = True
        self.val = new_val

    def limit_in(self, min, max):
        self.min = min
        self.max = max
        
    @property
    def value(self):
        if self.scale == 'linear':
            return self._val * self.unit
        elif self.scale == 'log':
            return 10 ** self._val * self.unit
        else:
            raise ValueError('invalid parameter scale')
    
    @property
    def range(self):
        if self.frozen:
            return [self._val] * 2
        else:
            return [self.min, self.max]

    def todict(self):
        return {'val': self._val, 
                'min': self.min, 
                'max': self.max, 
                'comment': self.comment, 
                'scale': self.scale, 
                'unit': self.unit, 
                'frozen': self.frozen}

    @property
    def info(self):
        param_dict = self.todict()
        for key, value in param_dict.items():
            print('> %s: %s' % (key, value))


def init_session_state():
    if 'spec' not in st.session_state:
        st.session_state.spec = {}
    if 'spec_state' not in st.session_state:
        st.session_state.spec_state = {}
    if 'mo' not in st.session_state:
        st.session_state.mo = {}
    if 'mo_component' not in st.session_state:
        st.session_state.mo_component = {}
    if 'mo_state' not in st.session_state:
        st.session_state.mo_state = {}
    if 'fit' not in st.session_state:
        st.session_state.fit = None
    if 'fit_state' not in st.session_state:
        st.session_state.fit_state = {}
        st.session_state.fit_state['run_state'] = False
    if 'ana' not in st.session_state:
        st.session_state.ana = None
    if 'ana_state' not in st.session_state:
        st.session_state.ana_state = {}
        st.session_state.ana_state['run_state'] = False
    if 'plot' not in st.session_state:
        st.session_state.plot = None
    if 'plot_state' not in st.session_state:
        st.session_state.plot_state = {}
    if 'calc' not in st.session_state:
        st.session_state.calc = None
    if 'calc_state' not in st.session_state:
        st.session_state.calc_state = {}
