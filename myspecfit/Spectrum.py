import os
import json
import warnings
import numpy as np
from Stat import Stat
from io import BytesIO
from Source import Source
from Response import Response
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Background import Background
from collections import OrderedDict
from Tools import JsonEncoder, Parameter, flag_grouping


class Spectrum(object):

    def __init__(self):
        self.clear()


    def clear(self):
        self.groups = OrderedDict()
        self.check_status = False
        self.model = None


    def set(self, expr, spec=None, src=None, bkg=None, rsp=None, rmf=None, arf=None, specT=None, 
            ql=None, gr=None, nt=None, rii=None, sii=None, bii=None, rf=None, sf=None, bf=None, bvf=None, 
            wt=1, stat='pgstat'):
        assert type(expr) is str, '<expr> parameter should be str'
        if expr not in self.groups:
            self.groups[expr] = {}
        else:
            warnings.warn('note that %s has already in spectrum group dict!' % expr)

        if spec is not None:
            assert type(spec) is dict, 'if set spec, <spec> parameter should be dict!'
            self.groups[expr].update(spec)

        if src is not None:
            assert type(src) is str or isinstance(src, BytesIO), 'if set src, <src> parameter should be str or BytesIO!'
            self.groups[expr]['src'] = src

        if bkg is not None:
            assert type(bkg) is str or isinstance(bkg, BytesIO), 'if set bkg, <bkg> parameter should be str or BytesIO!'
            self.groups[expr]['bkg'] = bkg

        if rsp is not None:
            assert type(rsp) is str or isinstance(rsp, BytesIO), 'if set rsp, <rsp> parameter should be str or BytesIO!'
            self.groups[expr]['rsp'] = rsp

        if rmf is not None:
            assert type(rmf) is str or isinstance(rmf, BytesIO), 'if set rmf, <rmf> parameter should be str or BytesIO!'
            self.groups[expr]['rmf'] = rmf

        if arf is not None:
            assert type(arf) is str or isinstance(arf, BytesIO), 'if set arf, <arf> parameter should be str or BytesIO!'
            self.groups[expr]['arf'] = arf

        if specT is not None:
            assert type(specT) is int or type(specT) is float, \
                'if set specT, <specT> parameter should be int or float!'
        self.groups[expr]['specT'] = specT

        if ql is not None:
            assert type(ql) is list or type(ql) is np.ndarray, \
                'if set quality, <ql> parameter should be list or array!'
        self.groups[expr]['quality'] = ql

        if gr is not None:
            assert type(gr) is dict, \
                'if set group significance, <gr> parameter should be dict!'
        self.groups[expr]['group'] = gr

        if nt is not None:
            assert type(nt) is list or type(nt) is np.ndarray, \
                'if set notice, <nt> parameter should be list or array!'
        self.groups[expr]['notice'] = nt

        if rii is not None:
            assert type(rii) is int, 'if set rsp-ii idx, <rii> parameter should be int!'
        self.groups[expr]['rsp_ii'] = rii

        if sii is not None:
            assert type(sii) is int, 'if set src-ii idx, <sii> parameter should be int!'
        self.groups[expr]['src_ii'] = sii

        if bii is not None:
            assert type(bii) is int, 'if set bkg-ii idx, <bii> parameter should be int!'
        self.groups[expr]['bkg_ii'] = bii
    
        if rf is None: rf = Parameter(val=1, frozen=True)
        else: assert isinstance(rf, Parameter), \
            'if set rsp factor, <rf> parameter should be Parameter!'
        self.groups[expr]['rsp_factor'] = rf

        if sf is None: sf = Parameter(val=1, frozen=True)
        else: assert isinstance(sf, Parameter), \
                'if set src exp factor, <sf> parameter should be Parameter!'
        self.groups[expr]['sexp_factor'] = sf

        if bf is None: bf = Parameter(val=1, frozen=True)
        else: assert isinstance(bf, Parameter), \
            'if set bkg exp factor, <bf> parameter should be Parameter!'
        self.groups[expr]['bexp_factor'] = bf

        if bvf is None: bvf = Parameter(val=0, frozen=True)
        else: assert isinstance(bvf, Parameter), \
            'if set bkg var factor, <bvf> parameter should be Parameter!'
        self.groups[expr]['bvar_factor'] = bvf

        assert type(wt) is int or type(wt) is float, \
            'if set weight, <wt> parameter should be int or float!'
        self.groups[expr]['weight'] = wt

        assert type(stat) is str, 'if set stat, <stat> parameter should be str!'
        self.groups[expr]['stat'] = stat

        self.info(expr)


    def pop(self, expr, which):
        assert expr in self.groups, '<expr> parameter is not a key of groups'

        if which == 'spec':
            _ = self.groups.pop(expr)

        if which == 'src':
            assert 'src' in self.groups[expr], 'src is not a key of <groups[expr]>'
            _ = self.groups[expr].pop('src')

        if which == 'bkg':
            assert 'bkg' in self.groups[expr], 'bkg is not a key of <groups[expr]>'
            _ = self.groups[expr].pop('bkg')

        if which == 'rsp':
            assert 'rsp' in self.groups[expr], 'rsp is not a key of <groups[expr]>'
            _ = self.groups[expr].pop('rsp')

        if which == 'rmf':
            assert 'rmf' in self.groups[expr], 'rmf is not a key of <groups[expr]>'
            _ = self.groups[expr].pop('rmf')

        if which == 'arf':
            assert 'arf' in self.groups[expr], 'arf is not a key of <groups[expr]>'
            _ = self.groups[expr].pop('arf')

        if which == 'specT':
            assert 'specT' in self.groups[expr], 'specT is not a key of <groups[expr]>'
            self.groups[expr]['specT'] = None

        if which == 'ql':
            assert 'quality' in self.groups[expr], 'quality is not a key of <groups[expr]>'
            self.groups[expr]['quality'] = None

        if which == 'gr':
            assert 'group' in self.groups[expr], 'group is not a key of <groups[expr]>'
            self.groups[expr]['group'] = None

        if which == 'nt':
            assert 'notice' in self.groups[expr], 'notice is not a key of <groups[expr]>'
            self.groups[expr]['notice'] = None

        if which == 'rii':
            assert 'rsp_ii' in self.groups[expr], 'rsp_ii is not a key of <groups[expr]>'
            self.groups[expr]['rsp_ii'] = None

        if which == 'sii':
            assert 'src_ii' in self.groups[expr], 'src_ii is not a key of <groups[expr]>'
            self.groups[expr]['src_ii'] = None

        if which == 'bii':
            assert 'bkg_ii' in self.groups[expr], 'bkg_ii is not a key of <groups[expr]>'
            self.groups[expr]['bkg_ii'] = None

        if which == 'rf':
            assert 'rsp_factor' in self.groups[expr], 'rsp_factor is not a key of <groups[expr]>'
            self.groups[expr]['rsp_factor'] = None

        if which == 'sf':
            assert 'sexp_factor' in self.groups[expr], 'sexp_factor is not a key of <groups[expr]>'
            self.groups[expr]['sexp_factor'] = None

        if which == 'bf':
            assert 'bexp_factor' in self.groups[expr], 'bexp_factor is not a key of <groups[expr]>'
            self.groups[expr]['bexp_factor'] = None

        if which == 'bvf':
            assert 'bvar_factor' in self.groups[expr], 'bvar_factor is not a key of <groups[expr]>'
            self.groups[expr]['bvar_factor'] = None

        if which == 'wt':
            assert 'weight' in self.groups[expr], 'weight is not a key of <groups[expr]>'
            self.groups[expr]['weight'] = 1

        if which == 'stat':
            assert 'stat' in self.groups[expr], 'stat is not a key of <groups[expr]>'
            self.groups[expr]['stat'] = 'pgstat'

        self.info(expr)


    def info(self, expr):
        if expr in self.groups:
            spec = self.groups[expr]
            print('+-----------------------------------------------+')
            print(' spec alias: %s'%expr)
            for key, value in spec.items():
                if key in ['rsp_factor', 'sexp_factor', 'bexp_factor', 'bvar_factor']:
                    print(' ' + key + ': ' + str(value.range))
                elif key in ['src', 'bkg', 'rsp', 'rmf', 'arf'] and isinstance(value, BytesIO):
                    print(' ' + key + ': ' + str(value.name))
                else:
                    print(' ' + key + ': ' + str(value))
            print('+-----------------------------------------------+\n')
        else:
            print('<expr> parameter is not a key of groups')


    def check(self, expr=None):
        if expr is None:
            self.check_groups = self.groups
        else:
            if expr in self.groups:
                self.check_groups = {expr: self.groups[expr]}
            else:
                print('<expr> parameter is not a key of groups')

        self.g1 = {'src', 'bkg', 'rsp'}
        self.g2 = {'src', 'bkg', 'rmf', 'arf'}

        print('+-----------------------------------------------+')
        if False in list(map(lambda x: self.g1 < set(x.keys()) or self.g2 < set(x.keys()), self.check_groups.values())):
            print(' some groups are NOT complete!')
            print(' please check to ensure them complete!')
            self.check_status = False
        else:
            print(' All groups are complete!')
            self.check_status = True
        print('+-----------------------------------------------+\n')

        if self.check_status:
            self.making()
            self.qualifying()
            self.grouping()
            self.create_ET_coord()
            self.create_ET_patch()

        
    def fit_with(self, model):
        from Model import Model
        assert isinstance(model, Model), 'model parameter should be Model type!'
        self.model = model
        model.spectrum = self


    def display(self, expr):
        assert self.check_status, 'please run check method first!'
        assert expr in self.check_groups, '<expr> parameter is not a key of check groups'
        i = self.exprs.index(expr)

        rcParams['font.family'] = 'sans-serif'
        # rcParams['font.sans-serif'] = 'Arial'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        self.fig = plt.figure(figsize=(6, 5))
        gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
        ax = self.fig.add_subplot(gs[0, 0])

        ax.errorbar(self.rsp_info[i].ChanCenter, self.src_info[i].SrcCounts, yerr=self.src_info[i].SrcErr, 
                    xerr=[np.array(self.rsp_info[i].ChanCenter) - np.array(self.rsp_info[i].ChanMin), 
                            np.array(self.rsp_info[i].ChanMax) - np.array(self.rsp_info[i].ChanCenter)], 
                    fmt='-m', lw=0.8, ds='steps-mid', elinewidth=0.8, capsize=0, label=expr + ' src')
        ax.errorbar(self.rsp_info[i].ChanCenter, self.bkg_info[i].BkgCounts, yerr=self.bkg_info[i].BkgErr, 
                    xerr=[np.array(self.rsp_info[i].ChanCenter) - np.array(self.rsp_info[i].ChanMin), 
                            np.array(self.rsp_info[i].ChanMax) - np.array(self.rsp_info[i].ChanCenter)], 
                    fmt='-b', lw=0.8, ds='steps-mid', elinewidth=0.8, capsize=0, label=expr + ' bkg')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.autoscale(axis='x', tight=True)
        ax.autoscale(axis='y', tight=True)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(which='major', width=0.5, length=3)
        ax.tick_params(which='minor', width=0.5, length=2)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(frameon=False)


    def making(self):
        self.exprs = []
        self.srcs, self.bkgs, self.rsps, self.specTs = [], [], [], []
        self.qls, self.grs, self.nts, self.wts = [], [], [], []
        self.riis, self.siis, self.biis, self.stats = [], [], [], []
        self.rfs, self.sfs, self.bfs, self.bvfs = [], [], [], []
        for key, value in self.check_groups.items():
            self.exprs.append(key)
            self.srcs.append(value['src'])
            self.bkgs.append(value['bkg'])

            try:
                self.rsps.append([value['rsp']])
            except KeyError:
                self.rsps.append([value['rmf'], value['arf']])

            self.specTs.append(value['specT'])
            self.qls.append(value['quality'])
            self.grs.append(value['group'])
            self.nts.append(value['notice'])
            self.riis.append(value['rsp_ii'])
            self.siis.append(value['src_ii'])
            self.biis.append(value['bkg_ii'])
            self.rfs.append(value['rsp_factor'])
            self.sfs.append(value['sexp_factor'])
            self.bfs.append(value['bexp_factor'])
            self.bvfs.append(value['bvar_factor'])
            self.wts.append(value['weight'])
            self.stats.append(value['stat'])

        self.nspec = len(self.check_groups)
        self.srcs = list(map(lambda src: src if isinstance(src, BytesIO) else os.path.abspath(src), self.srcs))
        self.bkgs = list(map(lambda bkg: bkg if isinstance(bkg, BytesIO) else os.path.abspath(bkg), self.bkgs))
        self.rsps = list(map(lambda rsps: [rsp if isinstance(rsp, BytesIO) else os.path.abspath(rsp) for rsp in rsps], self.rsps))

        self.src_info = [Source(src_file, sii) for src_file, sii in zip(self.srcs, self.siis)]
        self.src_quals = [src.SrcQual for src in self.src_info]
        self.src_grpgs = [src.SrcGrpg for src in self.src_info]
        self.src_backsc = [src.SrcBackSc for src in self.src_info]

        self.bkg_info = [Background(bkg_file, bii) for bkg_file, bii in zip(self.bkgs, self.biis)]
        self.bkg_backsc = [bkg.BkgBackSc for bkg in self.bkg_info]

        self.scaling()

        self.rsp_info = [Response(rsp_file, specT, rii) for rsp_file, specT, rii in zip(self.rsps, self.specTs, self.riis)]

        self.group_info = list(zip(self.src_info, self.bkg_info, self.rsp_info))

        self.params = [OrderedDict({'rf@%s'%expr: rf, 'sf@%s'%expr: sf, 'bf@%s'%expr: bf, 'bvf@%s'%expr: bvf})
                       for expr, rf, sf, bf, bvf in zip(self.exprs, self.rfs, self.sfs, self.bfs, self.bvfs)]

        stat_obj = Stat()
        self.stat_funcs = [stat_obj.set_stat(stat) for stat in self.stats]


    def scaling(self):
        for (bkg, src_sc, bkg_sc) in zip(self.bkg_info, self.src_backsc, self.bkg_backsc):
            bkg.scaling(src_sc, bkg_sc)
            
        self.src_eff = [src.SrcEff for src in self.src_info]
        self.bkg_eff = [bkg.BkgEff for bkg in self.bkg_info]
        
        self.alpha = [se / be for se, be in zip(self.src_eff, self.bkg_eff)]

        print('+-----------------------------------------------+')
        print(' %-17s%-17s%-17s' % ('Spec', 'SrcScal', 'BkgScal'))
        print('+-----------------------------------------------+')
        for i in range(self.nspec):
            print(' %-17s%-17s%-17s' % (self.exprs[i], self.src_backsc[i], self.bkg_backsc[i]))
        print('+-----------------------------------------------+\n')


    def qualifying(self, idx=':'):
        for i, (rsp, ql, nt) in enumerate(zip(self.rsp_info, self.qls, self.nts)):
            if ql is None:
                rsp.qualifying(self.src_quals[i], nt)
            else:
                rsp.qualifying(ql, nt)

        print('+-----------------------------------------------+')
        print(' %-20s%-30s' % ('Spec', 'Qual-Notc'))
        print('+-----------------------------------------------+')
        for i, rsp in enumerate(self.rsp_info):
            if idx == ':' or idx == i:
                print(' %-20s%-30s' % (self.exprs[i], rsp.Qualified_Notice))
        print('+-----------------------------------------------+\n')


    def grouping(self):
        for i, (src, bkg, rsp, gr) in enumerate(zip(self.src_info, self.bkg_info, self.rsp_info, self.grs)):
            if gr is None:
                src.grouping(self.src_grpgs[i])
                bkg.grouping(self.src_grpgs[i])
                rsp.grouping(self.src_grpgs[i])
            else:
                gr_ = {'min_evt': None, 'min_sigma': None, 'max_bin': None}
                gr_.update(gr)
                ini_flag = np.array(rsp.Qualified_Notice_ID).astype(int)
                grpg = flag_grouping(src.SrcCounts, bkg.BkgCounts, bkg.BkgErr, src.SrcExpo, bkg.BkgExpo, 
                                     src.SrcBackSc, bkg.BkgBackSc, 
                                     min_evt=gr_['min_evt'], min_sigma=gr_['min_sigma'],
                                     max_bin=gr_['max_bin'], stat=self.stats[i], ini_flag=ini_flag)
                src.grouping(grpg)
                bkg.grouping(grpg)
                rsp.grouping(grpg)


    def create_ET_coord(self):
        self.E_coord = []
        self.T_coord = []
        self.ET_level = [0]
        
        for rsp in self.rsp_info:
            num = len(rsp.Eval_Energy)
            self.E_coord.extend(rsp.Eval_Energy)
            self.T_coord.extend([rsp.specT] * num)
            self.ET_level.extend(rsp.Eval_Level[1:])
        self.E_coord = np.array(self.E_coord, dtype=np.float64)
        self.T_coord = np.array(self.T_coord, dtype=np.float64)
        self.ET_level = np.array(self.ET_level, dtype=int)
        self.ET_start = np.cumsum(self.ET_level)[:-1]
        self.ET_stop = np.cumsum(self.ET_level)[1:]


    def create_ET_patch(self):
        self.E_patch = []
        self.T_patch = []
        self.ET_width = []
        self.ET_number = [0]

        for rsp in self.rsp_info:
            num = len(rsp.EnerBins)
            self.E_patch.extend(rsp.EnerBins)
            self.T_patch.extend([rsp.specT] * num)
            self.ET_width.extend(rsp.EnerWidth)
            self.ET_number.append(num)
        self.E_patch = np.array(self.E_patch, dtype=np.float64)
        self.T_patch = np.array(self.T_patch, dtype=np.float64)
        self.ET_width = np.array(self.ET_width, dtype=np.float64)
        self.ET_Start = np.cumsum(self.ET_number)[:-1]
        self.ET_Stop = np.cumsum(self.ET_number)[1:]


    def save(self, savepath):
        assert self.check_status, 'please run check method first!'

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        np.savetxt(savepath + 'E_coord.txt', self.E_coord)
        np.savetxt(savepath + 'T_coord.txt', self.T_coord)
        np.savetxt(savepath + 'E_patch.txt', self.E_patch)
        np.savetxt(savepath + 'T_patch.txt', self.T_patch)
        np.savetxt(savepath + 'ET_width.txt', self.ET_width)
        np.savetxt(savepath + 'ET_level.txt', self.ET_level, fmt='%d')
        np.savetxt(savepath + 'ET_number.txt', self.ET_number, fmt='%d')

        json.dump(self.check_groups, open(savepath + 'groups.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump(self.params, open(savepath + 'params.json', 'w'), indent=4, cls=JsonEncoder)

        src_info_ = [Source(src_file, sii) for src_file, sii in zip(self.srcs, self.siis)]
        bkg_info_ = [Background(bkg_file, bii) for bkg_file, bii in zip(self.bkgs, self.biis)]
        rsp_info_ = [Response(rsp_file, specT, rii) for rsp_file, specT, rii in zip(self.rsps, self.specTs, self.riis)]

        json.dump({ex: src.ChanIndex for ex, src in zip(self.exprs, src_info_)}, 
                  open(savepath + 'src_ChanIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcCounts for ex, src in zip(self.exprs, src_info_)}, 
                  open(savepath + 'src_SrcCounts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcErr for ex, src in zip(self.exprs, src_info_)}, 
                  open(savepath + 'src_SrcErr.json', 'w'), indent=4, cls=JsonEncoder)
        
        json.dump({ex: bkg.ChanIndex for ex, bkg in zip(self.exprs, bkg_info_)}, 
                  open(savepath + 'bkg_ChanIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: bkg.BkgCounts for ex, bkg in zip(self.exprs, bkg_info_)}, 
                  open(savepath + 'bkg_BkgCounts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: bkg.BkgErr for ex, bkg in zip(self.exprs, bkg_info_)}, 
                  open(savepath + 'bkg_BkgErr.json', 'w'), indent=4, cls=JsonEncoder)
        
        json.dump({ex: rsp.ChanIndex for ex, rsp in zip(self.exprs, rsp_info_)}, 
                  open(savepath + 'rsp_ChanIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.ChanBins for ex, rsp in zip(self.exprs, rsp_info_)}, 
                  open(savepath + 'rsp_ChanBins.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.drm for ex, rsp in zip(self.exprs, rsp_info_)}, 
                  open(savepath + 'rsp_drm.json', 'w'), cls=JsonEncoder)

        json.dump({ex: src.ChanIndex for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-ChanIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcCounts for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-SrcCounts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcErr for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-SrcErr.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcExpo for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-SrcExpo.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcQual for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-SrcQual.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcGrpg for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-SrcGrpg.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: src.SrcBackSc for ex, src in zip(self.exprs, self.src_info)}, 
                  open(savepath + 'src-SrcBackSc.json', 'w'), indent=4, cls=JsonEncoder)
        
        json.dump({ex: bkg.ChanIndex for ex, bkg in zip(self.exprs, self.bkg_info)}, 
                  open(savepath + 'bkg-ChanIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: bkg.BkgCounts for ex, bkg in zip(self.exprs, self.bkg_info)}, 
                  open(savepath + 'bkg-BkgCounts.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: bkg.BkgErr for ex, bkg in zip(self.exprs, self.bkg_info)}, 
                  open(savepath + 'bkg-BkgErr.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: bkg.BkgExpo for ex, bkg in zip(self.exprs, self.bkg_info)}, 
                  open(savepath + 'bkg-BkgExpo.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: bkg.BkgBackSc for ex, bkg in zip(self.exprs, self.bkg_info)}, 
                  open(savepath + 'bkg-BkgBackSc.json', 'w'), indent=4, cls=JsonEncoder)
        
        json.dump({ex: rsp.ChanIndex for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-ChanIndex.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.ChanBins for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-ChanBins.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.EnerBins for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-EnerBins.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.specresp for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-specresp.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.matrix for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-matrix.json', 'w'), cls=JsonEncoder)
        json.dump({ex: rsp.drm for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-drm.json', 'w'), cls=JsonEncoder)
        json.dump({ex: rsp.Eval_Energy for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-Eval_Energy.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.Eval_Level for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-Eval_Level.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.qualified for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-qualified.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.notice for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-notice.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.qualified_id for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-qualified_id.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.notice_id for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-notice_id.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.Qualified_Notice for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-Qualified_Notice.json', 'w'), indent=4, cls=JsonEncoder)
        json.dump({ex: rsp.Qualified_Notice_ID for ex, rsp in zip(self.exprs, self.rsp_info)}, 
                  open(savepath + 'rsp-Qualified_Notice_ID.json', 'w'), indent=4, cls=JsonEncoder)
