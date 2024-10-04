import sys
import pandas as pd
import streamlit as st
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Analyse import Analyse
from Tools import init_session_state


st.set_page_config(
    page_title="Analyse",
    page_icon="📝")

css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

def set_ini(key, ini=None):
    if key not in st.session_state.ana_state:
        st.session_state.ana_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.ana_state[key] = st.session_state[key]
    return st.session_state.ana_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.ana_state[key] = st.session_state[key]
    value = st.session_state.ana_state[key]
    idx = value if value is None else options.index(value)
    return idx

st.session_state.ana_state['run_state'] = False

if not st.session_state.fit_state['run_state']:
    st.sidebar.warning('Please run complete the fitting process', icon="⚠️")
else:
    st.session_state.ana = Analyse(st.session_state.fit)

    with st.expander('***Configure the analyse object***', expanded=True):
        set_col, _, check_col = st.columns([4.9, 0.2, 4.9])

        with set_col:
            key = 'ana_post_style'; ini = 'maxl'; options = ['maxl', 'midv']; set_ini(key, ini)
            post_style = st.selectbox('Choose posterior analyse style', options, index=get_idx(key, options), key=key)

            key = 'ana_post_level'; ini = '1sigma'; options = ['nsigma', '1sigma', '2sigma', '3sigma']; set_ini(key, ini)
            post_level = st.selectbox('Choose confidence level of maxl style', options, index=get_idx(key, options), key=key)

            key = 'ana_err_level'; ini = '1sigma'; options = ['1sigma', '2sigma', '3sigma']; set_ini(key, ini)
            err_level = st.selectbox('Choose confidence level of uncertainty', options, index=get_idx(key, options), key=key)

            st.session_state.ana.set(err_level=err_level, post_style=post_style, post_level=post_level)

        with check_col:
            key = 'ana_post'; ini = False; set_ini(key, ini)
            on = st.toggle('Analyse posterior sample', value=get_val(key), key=key)
            if on: 
                st.session_state.ana.post(pdf=False)
                st.session_state.ana_state['run_state'] = True

            key = 'ana_post_param'; ini = False; set_ini(key, ini)
            if st.checkbox('Show posterior parameters', value=get_val(key), key=key):
                if not st.session_state.ana_state['run_state']:
                    st.warning('Please analyse posterior sample!', icon="⚠️")
                else:
                    post_fit = {'pid': st.session_state.ana.fobj.pids, 
                                'expr': st.session_state.ana.fobj.pexprs, 
                                'label': st.session_state.ana.fobj.plabels, 
                                'value': ['%.2f_{-%.2f}^{+%.2f}'%(pv[2], pv[3], pv[4]) for pv in st.session_state.ana.Post_Fit]}
                    post_fit_df = pd.DataFrame(post_fit)
                    st.dataframe(post_fit_df, use_container_width=True, hide_index=True)

            key = 'ana_post_goodness'; ini = False; set_ini(key, ini)
            if st.checkbox('Show posterior goodness', value=get_val(key), key=key):
                if not st.session_state.ana_state['run_state']:
                    st.warning('Please analyse posterior sample!', icon="⚠️")
                else:
                    post_goodness = {'lnL': ['%.2f'%st.session_state.ana.post_lnL], 
                                     'STAT': ['%.2f'%st.session_state.ana.post_stat], 
                                     'dof': ['%d'%st.session_state.ana.post_dof], 
                                     'BIC': ['%.2f'%st.session_state.ana.post_bic], 
                                     'AIC': ['%.2f'%st.session_state.ana.post_aic], 
                                     'AICc': ['%.2f'%st.session_state.ana.post_aicc], 
                                     'lnZ': ['%.2f'%st.session_state.ana.post_lnZ]}
                    post_goodness_df = pd.DataFrame(post_goodness)
                    st.dataframe(post_goodness_df, use_container_width=True, hide_index=True)

            key = 'ana_corner'; ini = False; set_ini(key, ini)
            if st.checkbox('Plot posterior parameters', value=ini, key=key):
                if not st.session_state.ana_state['run_state']:
                    st.warning('Please analyse posterior sample!', icon="⚠️")
                else:
                    with st.popover("Plot settings", use_container_width=True):
                        key = 'plot_corner_smooth'; ini = 2; set_ini(key, ini)
                        smooth = st.slider('Select the smooth parameter', 0, 5, get_val(key), key=key)

                        key = 'plot_corner_level'; ini = [1, 1.5, 2]; set_ini(key, ini); options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                        level = st.multiselect('Select the confidence levels', options=options, default=get_val(key), key=key)
                        if len(level) == 0:
                            level = [1, 1.5, 2]
                            st.info('No confidence levels, set to defaults [1, 1.5, 2]!')

                    st.session_state.ana.corner(smooth=smooth, level=level)
                    st.pyplot(st.session_state.ana.corner_fig, use_container_width=True)
