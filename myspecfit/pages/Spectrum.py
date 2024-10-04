import sys
import numpy as np
import pandas as pd
from io import BytesIO
import streamlit as st
from copy import deepcopy
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Spectrum import Spectrum
from Tools import init_session_state


st.set_page_config(
    page_title="Spectrum",
    page_icon="🔭")

css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

def set_ini(key, ini=None):
    if key not in st.session_state.spec_state:
        st.session_state.spec_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.spec_state[key] = st.session_state[key]
    return st.session_state.spec_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.spec_state[key] = st.session_state[key]
    value = st.session_state.spec_state[key]
    idx = value if value is None else options.index(value)
    return idx

def get_file(key, accept_multiple_files=False):
    if accept_multiple_files:
        if st.session_state[key] != []:
            st.session_state.spec_state[key] = st.session_state[key]
        else:
            if st.session_state.spec_state[key] != []:
                for file in st.session_state.spec_state[key]:
                    st.write('📄', file.name)
    else:
        if st.session_state[key] is not None:
            st.session_state.spec_state[key] = st.session_state[key]
        else:
            if st.session_state.spec_state[key] is not None:
                st.write('📄', st.session_state.spec_state[key].name)
    return st.session_state.spec_state[key]

def reset_spec():
    st.session_state.spec = {}

key = 'nspec'; ini = 'min'; set_ini(key, ini)
nspec = st.sidebar.number_input('**Input the number of spectrum object**', min_value=1, value=get_val(key), key=key, on_change=reset_spec)
for i in range(nspec): st.session_state.spec['Spec%d' % (i+1)] = Spectrum()

for si, spec_key in enumerate(st.session_state.spec.keys()):
    st.session_state.spec[spec_key].clear()
    with st.expander("***Configure the spectrum object %s***" % spec_key, expanded=False):
        ngroup_col, _, fit_col = st.columns([4.9, 0.2, 4.9])
        with ngroup_col:
            key = '%s_ngroup' % spec_key; ini = 'min'; set_ini(key, ini)
            ngroup = st.number_input('Input the number of groups of spectra', min_value=1, value=get_val(key), key=key)
        with fit_col:
            key = '%s_model' % spec_key; ini = None; options = list(st.session_state.mo.keys()); set_ini(key, ini)
            mo_key = st.selectbox('Choose a model object fitting to this spectrum object', options, 
                                  index=get_idx(key, options), key=key)
            st.session_state.mo_state['%s_spectrum' % mo_key] = spec_key

        group_keys = ['group%d-%d' % (si+1, i+1) for i in range(ngroup)]
        group_tabs = st.tabs(group_keys)
        for gi, (group_key, group_tab) in enumerate(zip(group_keys, group_tabs)):
            with group_tab:
                set_col, _, info_col = st.columns([4.9, 0.2, 4.9])
                with set_col:
                    ##### expr #####
                    key = '%s_%s_expr' % (spec_key, group_key); ini = group_key; set_ini(key, ini)
                    expr = st.text_input('Input group name', value=get_val(key), placeholder=group_key, key=key)
                    if expr is None or expr == '': expr = group_key
                    
                    if expr in st.session_state.spec[spec_key].groups:
                        st.warning('Sorry for prohibiting the use of the same group name', icon="⚠️")

                    ##### spec files #####
                    key = '%s_%s_spec' % (spec_key, group_key); ini = []; set_ini(key, ini)
                    spec_files = st.file_uploader('Choose spectral files: src, bkg, rsp (or rmf & arf)', accept_multiple_files=True, key=key)
                    spec_files = get_file(key, True)
                    if spec_files is None: spec = {}
                    else:
                        spec = {}
                        for speci in spec_files:
                            if 'src' in speci.name or 'pha' in speci.name: spec['src'] = speci
                            if 'bkg' in speci.name or 'bak' in speci.name: spec['bkg'] = speci
                            if 'rsp' in speci.name or 'resp' in speci.name: spec['rsp'] = speci
                            if 'rmf' in speci.name: spec['rmf'] = speci
                            if 'arf' in speci.name: spec['arf'] = speci
                    
                    key = '%s_%s_src' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    src = st.file_uploader('Choose source spectrum: src', key=key)
                    src = get_file(key)

                    key = '%s_%s_bkg' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    bkg = st.file_uploader('Choose background spectrum: bkg', key=key)
                    bkg = get_file(key)

                    key = '%s_%s_rsp' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    rsp = st.file_uploader('Choose response matrix: rsp', key=key)
                    rsp = get_file(key)

                    key = '%s_%s_rmf' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    rmf = st.file_uploader('Choose redistribution matrix: rmf', key=key)
                    rmf = get_file(key)

                    key = '%s_%s_arf' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    arf = st.file_uploader('Choose auxiliary response matrix: arf', key=key)
                    arf = get_file(key)

                    ##### stat #####
                    key = '%s_%s_stat' % (spec_key, group_key); ini = 'pgstat'; set_ini(key, ini)
                    options = ['pgstat', 'cstat', 'chi^2']
                    stat = st.selectbox('Choose fitting statistic metric: stat', options, index=get_idx(key, options), key=key)

                    ##### nt #####
                    key = '%s_%s_nt' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    nt = st.text_input('Set notice energy: nt', value=get_val(key), placeholder='8-30;40-1000 (defaults to None)', key=key)
                    if nt == '': nt = None
                    if nt is not None:
                        nt_list = nt.split(';')
                        nt = []
                        for nt_str in nt_list:
                            nt_range = nt_str.split('-')
                            if len(nt_range) == 2: 
                                try: nt1 = float(nt_range[0].strip())
                                except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                                try: nt2 = float(nt_range[1].strip())
                                except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                                nt.append([nt1, nt2])
                            else: st.error('The input value is in the wrong format!', icon="🚨")

                    ##### specT #####
                    key = '%s_%s_specT' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    specT = st.text_input('Set spectral time: specT', value=get_val(key), placeholder='1.0 (defaults to None)', key=key)
                    if specT == '': specT = None
                    if specT is not None:
                        try: specT = float(specT)
                        except: st.error('The input value needs to be able to be converted to float!', icon="🚨")

                    ##### gr #####
                    key = '%s_%s_gr_evt' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    gr_min_evt = st.text_input('Set grouping minimum events: gr_min_evt', value=get_val(key), placeholder='5 (defaults to None)', key=key)
                    if gr_min_evt == '': gr_min_evt = None
                    if gr_min_evt is not None:
                        try: gr_min_evt = int(gr_min_evt)
                        except: st.error('The input value needs to be able to be converted to int!', icon="🚨")

                    key = '%s_%s_gr_sig' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    gr_min_sigma = st.text_input('Set grouping minimum sigma: gr_min_sigma', value=get_val(key), placeholder='3 (defaults to None)', key=key)
                    if gr_min_sigma == '': gr_min_sigma = None
                    if gr_min_sigma is not None:
                        try: gr_min_sigma = float(gr_min_sigma)
                        except: st.error('The input value needs to be able to be converted to float!', icon="🚨")

                    key = '%s_%s_gr_bin' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    gr_max_bin = st.text_input('Set grouping maximum bins: gr_max_bin', value=get_val(key), placeholder='20 (defaults to None)', key=key)
                    if gr_max_bin == '': gr_max_bin = None
                    if gr_max_bin is not None:
                        try: gr_max_bin = int(gr_max_bin)
                        except: st.error('The input value needs to be able to be converted to int!', icon="🚨")

                    if gr_min_evt is None and gr_min_sigma is None and gr_max_bin is None: gr = None
                    else: gr = {'min_evt': gr_min_evt, 'min_sigma': gr_min_sigma, 'max_bin': gr_max_bin}

                    ##### rii #####
                    key = '%s_%s_rii' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    rii = st.text_input('Set rsp2 index: rii', value=get_val(key), placeholder='0 (defaults to None)', key=key)
                    if rii == '': rii = None
                    if rii is not None:
                        try: rii = int(rii) 
                        except: st.error('The input value needs to be able to be converted to int!', icon="🚨")

                    ##### sii #####
                    key = '%s_%s_sii' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    sii = st.text_input('Set src2 index: sii', value=get_val(key), placeholder='0 (defaults to None)', key=key)
                    if sii == '': sii = None
                    if sii is not None:
                        try: sii = int(sii) 
                        except: st.error('The input value needs to be able to be converted to int!', icon="🚨")

                    ##### bii #####
                    key = '%s_%s_bii' % (spec_key, group_key); ini = None; set_ini(key, ini)
                    bii = st.text_input('Set bkg2 index: bii', value=get_val(key), placeholder='0 (defaults to None)', key=key)
                    if bii == '': bii = None
                    if bii is not None:
                        try: bii = int(bii) 
                        except: st.error('The input value needs to be able to be converted to int!', icon="🚨")

                    ##### spec.set #####
                    src = deepcopy(src); bkg = deepcopy(bkg); rsp = deepcopy(rsp)
                    st.session_state.spec[spec_key].set(expr=expr, spec=spec, src=src, bkg=bkg, rsp=rsp, rmf=rmf, arf=arf, 
                                                        specT=specT, ql=None, gr=gr, nt=nt, rii=rii, sii=sii, bii=bii, rf=None, 
                                                        sf=None, bf=None, bvf=None, wt=1, stat=stat)

                with info_col: 
                    st.write(''); st.write('')
                    spec_info = deepcopy(st.session_state.spec[spec_key].groups[expr])

                    key = '%s_%s_info' % (spec_key, group_key); ini = False; set_ini(key, ini)
                    if st.checkbox('Show spectral infomation', value=get_val(key), key=key):
                        if 'src' in spec_info and isinstance(spec_info['src'], BytesIO): spec_info['src'] = spec_info['src'].name
                        if 'bkg' in spec_info and isinstance(spec_info['bkg'], BytesIO): spec_info['bkg'] = spec_info['bkg'].name
                        if 'rsp' in spec_info and isinstance(spec_info['rsp'], BytesIO): spec_info['rsp'] = spec_info['rsp'].name
                        if 'rmf' in spec_info and isinstance(spec_info['rmf'], BytesIO): spec_info['rmf'] = spec_info['rmf'].name
                        if 'arf' in spec_info and isinstance(spec_info['arf'], BytesIO): spec_info['arf'] = spec_info['arf'].name
                        spec_info['rsp_factor'] = spec_info['rsp_factor'].range
                        spec_info['sexp_factor'] = spec_info['sexp_factor'].range
                        spec_info['bexp_factor'] = spec_info['bexp_factor'].range
                        spec_info['bvar_factor'] = spec_info['bvar_factor'].range

                        spec_info_df = pd.DataFrame({'group properties': spec_info.keys(), 
                                                     'values': ['%s' % value for value in spec_info.values()]})
                        st.dataframe(spec_info_df, use_container_width=True, hide_index=True)

                    key = '%s_%s_display' % (spec_key, group_key); ini = False; set_ini(key, ini)
                    if st.checkbox('Display spectral shape', value=get_val(key), key=key):
                        st.session_state.spec[spec_key].check(expr)
                        if not st.session_state.spec[spec_key].check_status:
                            st.warning('spectral group is **NOT** complete!', icon="⚠️")
                        else:
                            ch_energy = st.session_state.spec[spec_key].rsp_info[0].ChanCenter
                            src_cts = st.session_state.spec[spec_key].src_info[0].SrcCounts
                            bkg_cts = st.session_state.spec[spec_key].bkg_info[0].BkgCounts
                            
                            obs_df = pd.DataFrame({'logE': np.log10(ch_energy).astype(float), 
                                                   'SrcCts': src_cts.astype(float), 
                                                   'BkgCts': bkg_cts.astype(float)})
                            st.line_chart(obs_df, x='logE', y=['SrcCts', 'BkgCts'], use_container_width=True)
