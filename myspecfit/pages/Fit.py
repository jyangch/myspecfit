import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
from itertools import chain
from os.path import abspath, dirname
from threading import current_thread
from contextlib import contextmanager
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
sys.path.append(dirname(dirname(abspath(__file__))))
from Fit import Fit
from Tools import init_session_state


st.set_page_config(
    page_title="Fit",
    page_icon="⚖️")

css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    "this will show the prints"
    with st_redirect(sys.stdout, dst):
        yield

@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield

def set_ini(key, ini=None):
    if key not in st.session_state.fit_state:
        st.session_state.fit_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.fit_state[key] = st.session_state[key]
    return st.session_state.fit_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.fit_state[key] = st.session_state[key]
    value = st.session_state.fit_state[key]
    idx = value if value is None else options.index(value)
    return idx

st.session_state.fit = Fit()
st.session_state.fit_state['sm_pair'] = False
st.session_state.fit_state['run_state'] = False

sm_pairs = {}
for spec_key in st.session_state.spec.keys():
    mo_key = st.session_state.spec_state['%s_model' % spec_key]
    if mo_key is not None:
        if st.session_state.mo_state['%s_spectrum' % mo_key] == spec_key:
            sm_pairs['%s 🔗 %s' % (spec_key, mo_key)] = [spec_key, mo_key]

with st.expander('***Configure the fit object***', expanded=True):
    check_col, _, set_col = st.columns([4.9, 0.2, 4.9])

    with check_col:

        key = 'fit_pairs'; ini = list(sm_pairs.keys()); set_ini(key, ini)
        options = list(sm_pairs.keys())
        pairs = st.multiselect('Select fitting pairs', options=options, default=get_val(key), key=key)

        if len(pairs) > 0:
            st.session_state.fit_state['sm_pair'] = True
            for pair in pairs:
                spec_key, mo_key = sm_pairs[pair]
                st.session_state.fit.set(st.session_state.spec[spec_key], st.session_state.mo[mo_key])

        key = 'fit_check'; ini = True; set_ini(key, ini)
        if st.checkbox('Check fitting information', value=get_val(key), key=key):
            if not st.session_state.fit_state['sm_pair']:
                st.warning('**NO** spec-mo fitting pair!', icon="⚠️")
            else:
                st.session_state.fit.check()

                sms = {'spectrum': [], 'statistic': [], 'model': []}
                for i, j in enumerate(chain.from_iterable([i] * n for i, n in enumerate(st.session_state.fit.nspecs))):
                    sms['spectrum'].append(st.session_state.fit.spec_exprs[i])
                    sms['statistic'].append(st.session_state.fit.stat_exprs[i])
                    sms['model'].append(st.session_state.fit.mo_exprs[j])
                sms_df = pd.DataFrame(sms)
                st.dataframe(sms_df, use_container_width=True, hide_index=True)

                key = 'fit_nlink'; ini = 'min'; set_ini(key, ini)
                nlink = st.number_input('Input the number of groups of linking', min_value=0, value=get_val(key), key=key)

                for idx in range(nlink):
                    key = 'fit_link_%d' % idx; ini = None; set_ini(key, ini)
                    options = list(st.session_state.fit.pdicts.keys())
                    pids = st.multiselect('Select the parameters to link', options=options, default=get_val(key), key=key)
                    if len(pids) > 1:
                        st.session_state.fit.link(pids)

                key = 'fit_pdict'; ini = False; set_ini(key, ini)
                if st.checkbox('Show all fitting parameters', value=get_val(key), key=key):
                    pdict = []
                    for i, (pid, pv) in enumerate(st.session_state.fit.pdicts.items()):
                        expr, label = pv['expr'], pv['label']
                        range_, frozen = '%s' % pv['param'].range, '%s' % pv['frozen']
                        mates = ','.join(pv['mates']) if len(pv['mates']) != 0 else 'None'
                        pdict.append({'pid': pid, 'expr': expr, 'label': label, 'range': range_, 'frozen': frozen, 'link': mates})
                    pdict_df = pd.DataFrame(pdict)
                    st.dataframe(pdict_df, use_container_width=True, hide_index=True)

                key = 'fit_param'; ini = False; set_ini(key, ini)
                if st.checkbox('Show free fitting parameters', value=get_val(key), key=key):
                    param = []
                    for pid, expr, label, range_ in zip(st.session_state.fit.pids, 
                                                        st.session_state.fit.pexprs, 
                                                        st.session_state.fit.plabels, 
                                                        st.session_state.fit.pranges):
                        range_ = '%s' % range_
                        param.append({'pid': pid, 'expr': expr, 'label': label, 'range': range_})
                    param_df = pd.DataFrame(param)
                    st.dataframe(param_df, use_container_width=True, hide_index=True)

    with set_col:

        key = 'fit_engine'; ini = 'multinest'; options = ['multinest', 'emcee']; set_ini(key, ini)
        engine = st.selectbox('Choose fitting engine', options, index=get_idx(key, options), key=key)

        key = 'fit_path'; ini = None; set_ini(key, ini)
        path = st.text_input('Input folder name of results', value=get_val(key), placeholder='msf', key=key)
        if path == '' or path is None: path = 'msf_%d' % (np.random.uniform() * 1e10)
        path = dirname(dirname(abspath(__file__))) + '/results/' + path
        if os.path.exists(path):
            st.info('Note: the folder of results has already existed!')

        key = 'fit_resume'; ini = 'Yes'; options = ['Yes', 'No']; set_ini(key, ini)
        resume = st.selectbox('Choose to resume or not', options, index=get_idx(key, options), key=key)
        if resume == 'Yes': resume = True
        if resume == 'No': resume = False

        if engine == 'multinest':
            key = 'fit_nlive'; ini = 300; set_ini(key, ini)
            nlive = st.slider('Select the number of live point', 20, 1000, get_val(key), step=10, key=key)
            nstep = 2000; discard = 100
        if engine == 'emcee':
            key = 'fit_nstep'; ini = 2000; set_ini(key, ini)
            nstep = st.slider('Select the number of steps', 0, 10000, get_val(key), step=1000, key=key)
            key = 'fit_discard'; ini = 100; set_ini(key, ini)
            discard = st.slider('Select the discard steps', 0, 2000, get_val(key), step=100, key=key)
            nlive = 300

    key = 'fit_run'; ini = False; set_ini(key, ini)
    run = st.sidebar.toggle(':rainbow[**RUN RUN RUN**]', value=ini, key=key)
    if run:
        if not st.session_state.fit_state['sm_pair']:
            st.warning('**NO** spec-mo fitting pair!', icon="⚠️")
        else:
            with st.sidebar:
                with st.status('running spectral fitting...', expanded=True) as status:
                    st.write('Start: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    with st_stdout("info"):
                        st.session_state.fit.run(engine, path, nlive, nstep, discard, resume)
                    st.write('Stop: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    st.session_state.fit_state['run_state'] = True
                    status.update(label="run complete!", state="complete", expanded=False)
