import re
import sys
import streamlit as st
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Plot import Plot
from Tools import init_session_state


st.set_page_config(
    page_title="Plot",
    page_icon="🎨")

css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

def set_ini(key, ini=None):
    if key not in st.session_state.plot_state:
        st.session_state.plot_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.plot_state[key] = st.session_state[key]
    return st.session_state.plot_state[key]

if not st.session_state.ana_state['run_state']:
    st.sidebar.warning('Please run complete the analyse process', icon="⚠️")
else:
    st.session_state.plot = Plot(st.session_state.ana)

    with st.expander('***Configure the plot object***', expanded=True):
        spec_col, _, mo_col = st.columns([4.9, 0.2, 4.9])

        with spec_col:
            spec_exprs = st.session_state.plot.fobj.spec_exprs
            stat_exprs = st.session_state.plot.fobj.stat_exprs

            key = 'plot_spec_sex'; ini = 'all'; set_ini(key, ini)
            options = ['all'] + spec_exprs
            sexs = st.multiselect('Select spectrum groups to plot', options=options, default=get_val(key), key=key)

            if len(sexs) > 0:
                if 'all' in sexs: sexs = 'all'

                with st.popover("Rebin settings", use_container_width=True):
                    key = 'plot_rebin_save'; ini = False; set_ini(key, ini)
                    save = st.toggle('Save rebin results', value=get_val(key), key=key)

                    group_tabs = st.tabs(spec_exprs)
                    rebin_dict = {}
                    for group_key, group_tab, stat_key in zip(spec_exprs, group_tabs, stat_exprs):
                        with group_tab:
                            ini = 3 if stat_key == 'pgstat' else 0
                            key = 'plot_%s_min_sigma' % group_key; set_ini(key, ini)
                            min_sigma = st.slider('Select mimimum sigma of rebin', 0, 10, get_val(key), key=key)

                            ini = 10 if stat_key == 'pgstat' else 1
                            key = 'plot_%s_max_bin' % group_key; set_ini(key, ini)
                            max_bin = st.slider('Select maximum bins of rebin', 1, 100, get_val(key), key=key)

                            rebin_dict[group_key] = {'min_sigma': min_sigma, 'max_bin': max_bin}
                    st.session_state.plot.rebin(rebin_dict, save=save)

                key = 'plot_cspec'; ini = False; set_ini(key, ini)
                if st.checkbox('Plot observed and model-predicted counts spectra', value=ini, key=key):
                    st.session_state.plot.cspec(spec_exprs=sexs, ploter='plotly')
                    st.plotly_chart(st.session_state.plot.cspec_fig, theme="streamlit", use_container_width=True)

                key = 'plot_pspec'; ini = False; set_ini(key, ini)
                if st.checkbox('Plot model-predicted photon spectra', value=ini, key=key):
                    st.session_state.plot.pspec(spec_exprs=sexs, ploter='plotly')
                    st.plotly_chart(st.session_state.plot.pspec_fig, theme="streamlit", use_container_width=True)

        with mo_col:
            mo_exprs_sp = st.session_state.plot.fobj.mo_exprs_sp
            mo_funcs_sp = st.session_state.plot.fobj.mo_funcs_sp

            key = 'plot_mspec_add_exprs'; ini = None; set_ini(key, ini)
            exprs = st.text_input('Input any model expressions', value=get_val(key), key=key)
            if exprs is None: add_exprs = []
            elif exprs == '': add_exprs = []
            else:
                expr_list = exprs.split(';')
                add_exprs = []
                for expr in expr_list:
                    expr = re.sub('\s*', '', expr)
                    if expr in mo_exprs_sp: continue
                    expr_sp = re.split(r"[(+\-*/)]", expr)
                    expr_sp = [ex for ex in expr_sp if ex != '']
                    if not (set(expr_sp) <= set(mo_exprs_sp)):
                        st.warning('The model expression %s include invalid component name!' % expr, icon="⚠️")
                    else:
                        add_exprs = add_exprs + [expr]

            key = 'plot_mspec_exprs'; ini = None; set_ini(key, ini); options = list(set(mo_exprs_sp + add_exprs + st.session_state.fit.mo_exprs))
            mo_exprs = st.multiselect('Select model expressions', options=options, default=get_val(key), key=key)

            if len(mo_exprs) > 0:
                with st.popover("Plot settings", use_container_width=True):
                    # key = 'plot_mspec_type'; ini = 'N(E)'; set_ini(key, ini)
                    # type = st.select_slider('Select spectral type to display', options=['fv', 'N(E)', 'vfv'], value=get_val(key), key=key)

                    mo_spec = []
                    mo_tabs = st.tabs(['\*'.join(expr.split('*')) for expr in mo_exprs])
                    for mo_key, mo_tab in zip(mo_exprs, mo_tabs):
                        with mo_tab:
                            key = 'plot_mspec_%s_erange'%mo_key; ini = (0, 4); set_ini(key, ini)
                            erange = st.slider('Select energy range in logspace', -1, 5, get_val(key), key=key)
                            Emin = 10 ** erange[0]; Emax = 10 ** erange[1]

                            key = 'plot_mspec_%s_epoch'%mo_key; ini = None; set_ini(key, ini); placeholder = '1.0;2.0'
                            epoch_str = st.text_input('Input time points', value=get_val(key), placeholder=placeholder, key=key)
                            if epoch_str == '' or epoch_str is None: epochs = [None]
                            else:
                                epoch_list = epoch_str.split(';')
                                if len(epoch_list) == 1: 
                                    try: epochs = [float(epoch_list[0])]
                                    except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                                else:
                                    epochs = []
                                    for epoch in epoch_list:
                                        try: epoch = float(epoch)
                                        except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                                        else: epochs.append(epoch)
                            
                        for epoch in epochs:
                            mo_spec.append({'expr': mo_key, 'Emin': Emin, 'Emax': Emax, 'T': epoch})

                key = 'plot_mspec'; ini = False; set_ini(key, ini)
                if st.checkbox('Plot model expressions and components', value=ini, key=key):
                    st.session_state.plot.mspec(mo_spec, ploter='plotly')
                    st.plotly_chart(st.session_state.plot.mspec_fig, theme="streamlit", use_container_width=True)
