import re
import sys
import pandas as pd
import streamlit as st
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Calculate import Calculate
from Tools import init_session_state


st.set_page_config(
    page_title="Calculate",
    page_icon="🔢")

css='''
<style>
    section.main > div {max-width:40rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

def set_ini(key, ini=None):
    if key not in st.session_state.calc_state:
        st.session_state.calc_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.calc_state[key] = st.session_state[key]
    return st.session_state.calc_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.calc_state[key] = st.session_state[key]
    value = st.session_state.calc_state[key]
    idx = value if value is None else options.index(value)
    return idx

if not st.session_state.ana_state['run_state']:
    st.sidebar.warning('Please run complete the analyse process', icon="⚠️")
else:
    st.session_state.calc = Calculate(st.session_state.ana)

    with st.expander('***Configure the calculation object***', expanded=True):
        mo_exprs_sp = st.session_state.calc.fobj.mo_exprs_sp
        mo_funcs_sp = st.session_state.calc.fobj.mo_funcs_sp

        key = 'calc_flux_add_exprs'; ini = None; set_ini(key, ini)
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

        key = 'calc_flux_exprs'; ini = None; set_ini(key, ini); options = list(set(mo_exprs_sp + add_exprs + st.session_state.fit.mo_exprs))
        mo_expr = st.selectbox('Select model expression', options, index=get_idx(key, options), key=key)

        if mo_expr is not None: 
            with st.popover("Calculation settings", use_container_width=True):
                key = 'calc_flux_erange'; ini = None; set_ini(key, ini)
                erange = st.text_input('Set energy range', value=get_val(key), placeholder='defaults to 10-1000', key=key)
                if erange == '': erange = None
                if erange is None: e1 = 10; e2 = 1000
                if erange is not None:
                    erange = erange.split('-')
                    if len(erange) == 2: 
                        try: e1 = float(erange[0].strip())
                        except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                        try: e2 = float(erange[1].strip())
                        except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                    else: st.error('The input value is in the wrong format!', icon="🚨")

                key = 'calc_flux_epoch'; ini = None; set_ini(key, ini); placeholder = '1.0'
                epoch = st.text_input('Input time points', value=get_val(key), placeholder=placeholder, key=key)
                if epoch == '': epoch = None
                if epoch is not None:
                    try: epoch = float(epoch)
                    except: st.error('The input value needs to be able to be converted to float!', icon="🚨")

            key = 'calc_flux'; ini = False; set_ini(key, ini)
            if st.checkbox('Calculate photon and energy flux', value=ini, key=key):
                st.session_state.calc.flux(mo_expr, T=epoch, e1=e1, e2=e2)
                flux = {' ': ['phtflux', 'ergflux'], 
                        'value': ['%.2e' % st.session_state.calc.phtflux_mp['value'], '%.2e' % st.session_state.calc.ergflux_mp['value']], 
                        'value-': ['%.2e' % st.session_state.calc.phtflux_mp['1sigma_err'][0], '%.2e' % st.session_state.calc.ergflux_mp['1sigma_err'][0]], 
                        'value+': ['%.2e' % st.session_state.calc.phtflux_mp['1sigma_err'][1], '%.2e' % st.session_state.calc.ergflux_mp['1sigma_err'][1]]}
                flux_df = pd.DataFrame(flux)
                st.dataframe(flux_df, use_container_width=True, hide_index=True)
