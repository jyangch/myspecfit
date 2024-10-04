import re
import sys
import json
import importlib
import numpy as np
import pandas as pd
import streamlit as st
from code_editor import code_editor
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Model import Model
from Tools import init_session_state


st.set_page_config(
    page_title="Model",
    page_icon="🌈")

css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

init_session_state()

def set_ini(key, ini=None):
    if key not in st.session_state.mo_state:
        st.session_state.mo_state[key] = ini

def get_val(key):
    if key in st.session_state:
        st.session_state.mo_state[key] = st.session_state[key]
    return st.session_state.mo_state[key]

def get_data(key):
    if key in st.session_state:
        for row, edited in st.session_state[key]['edited_rows'].items():
            for col, value in edited.items():
                st.session_state.mo_state[key].loc[int(row), col] = value
    return st.session_state.mo_state[key]

def get_resp(key):
    if key in st.session_state:
        if st.session_state[key] is not None:
            if st.session_state[key]['type'] in ['submit', 'saved']:
                st.session_state.mo_state[key] = st.session_state[key]['text']
    return st.session_state.mo_state[key]

def get_idx(key, options):
    if key in st.session_state:
        st.session_state.mo_state[key] = st.session_state[key]
    value = st.session_state.mo_state[key]
    idx = value if value is None else options.index(value)
    return idx

def reset_mo():
    st.session_state.mo = {}
    st.session_state.mo_component = {}

def reset_mo():
    st.session_state.mo = {}

def reset_key(keys):
    for key in keys:
        if key in st.session_state:
            _ = st.session_state.pop(key)
        if key in st.session_state.mo_state:
            _ = st.session_state.mo_state.pop(key)

key = 'nmo'; ini = 'min'; set_ini(key, ini)
nmo = st.sidebar.number_input('**Input the number of model object**', min_value=1, value=get_val(key), key=key, on_change=reset_mo)
for i in range(nmo): st.session_state.mo['Mo%d' % (i+1)] = Model()
for i in range(nmo): st.session_state.mo_component['Mo%d' % (i+1)] = {}

for mi, mo_key in enumerate(st.session_state.mo.keys()):
    st.session_state.mo[mo_key].clear()
    with st.expander('***Configure the model object %s***' % mo_key, expanded=False):
        ncomponent_col, _, fit_col = st.columns([4.9, 0.2, 4.9])
        with ncomponent_col:
            key = '%s_ncomponent' % mo_key; ini = 'min'; set_ini(key, ini)
            ncomponent = st.number_input('Input the number of components of model', min_value=1, value=get_val(key), key=key)
        with fit_col:
            key = '%s_spectrum' % mo_key; ini = None; options = list(st.session_state.spec.keys()); set_ini(key, ini)
            spec_key = st.selectbox('Choose a spectrum object fitting with this model object', options, 
                                    index=get_idx(key, options), key=key)
            st.session_state.spec_state['%s_model' % spec_key] = mo_key

        component_keys = ['comp%d-%d' % (mi+1, i+1) for i in range(ncomponent)]
        expression_key = 'model expression'
        all_tabs = st.tabs(component_keys + [expression_key])
        component_tabs = all_tabs[:-1]
        expression_tab = all_tabs[-1]
        for component_key, component_tab in zip(component_keys, component_tabs):
            with component_tab:
                set_col, _, info_col = st.columns([4.9, 0.2, 4.9])
                with set_col:
                    key = '%s_%s_name' % (mo_key, component_key); ini = None; set_ini(key, ini)
                    options = list(st.session_state.mo[mo_key].mo_dict.keys()) + ['user']
                    component_name = st.selectbox('Choose a model component', options, index=get_idx(key, options), key=key, 
                                                  on_change=reset_key, args=(['%s_%s_param' % (mo_key, component_key), 
                                                                              '%s_%s_expr' % (mo_key, component_key)],))

                    if component_name is None: 
                        expr = component_key
                        component = None
                    else:
                        if component_name == 'user':
                            info = """**Note: Please make sure to back up yourself defined model, as this APP will not save it. 
                            If you want to use it as a build-in model for this APP, please contact the APP author.**"""
                            st.info(info)

                            editor_dir = dirname(dirname(abspath(__file__))) + '/docs/CodeEditor'
                            with open(editor_dir + '/example_custom_buttons_bar_alt.json') as json_button_file_alt:
                                custom_buttons_alt = json.load(json_button_file_alt)
                            with open(editor_dir + '/example_info_bar.json') as json_info_file:
                                info_bar = json.load(json_info_file)
                            with open(editor_dir + '/example_code_editor_css.scss') as css_file:
                                css_text = css_file.read()

                            comp_props = {"css": css_text, "globalCSS": ":root {\n  --streamlit-dark-font-family: monospace;\n}"}
                            ace_props = {"style": {"borderRadius": "0px 0px 8px 8px"}}

                            user_dir = dirname(dirname(abspath(__file__))) + '/models/users'
                            with open(user_dir + '/user.py') as file_obj:
                                mo_format = file_obj.read()

                            key = '%s_%s_user_model' % (mo_key, component_key); ini = mo_format; set_ini(key, ini)
                            response_dict = code_editor(get_resp(key), height=[30], lang='python', theme='default', 
                                                        shortcuts='vscode', focus=False, buttons=custom_buttons_alt, 
                                                        info=info_bar, component_props=comp_props, props=ace_props, 
                                                        options={"wrap": True}, key=key)

                            if response_dict['type'] == "submit" and len(response_dict['id']) != 0:
                                st.info('Note: you have submitted you model!')

                                key = '%s_%s_user_fname'%(mo_key, component_key); ini = 'user_%s_%s'%(mo_key, component_key); set_ini(key, ini)
                                user_fname = get_val(key)
                                with open(user_dir + '/%s.py' % user_fname, 'w') as file_obj:
                                    file_obj.write(response_dict['text'])

                                component = importlib.import_module('models.users.%s' % user_fname).user()
                                expr = component.expr
                            else:
                                expr = component_key
                                component = None

                        else:
                            component = st.session_state.mo[mo_key].set_mo(component_name)
                            st.info('Comment: %s' % component.comment)

                            ##### expr #####
                            key = '%s_%s_expr' % (mo_key, component_key); ini = component.expr; set_ini(key, ini)
                            expr = st.text_input('Input component name', value=get_val(key), placeholder=component.expr, key=key)
                            if expr is None or expr == '': expr = component.expr
                            if expr in st.session_state.mo_component[mo_key]:
                                st.warning('Sorry for prohibiting the use of the same component name', icon="⚠️")
                            component.expr = expr

                            ##### redshift #####
                            key = '%s_%s_redshift' % (mo_key, component_key); ini = component.redshift; set_ini(key, ini)
                            redshift = st.text_input('Input component redshift', value=get_val(key), placeholder=component.redshift, key=key)
                            if redshift == '' or redshift is None: 
                                redshift = component.redshift
                            else:
                                try: redshift = float(redshift)
                                except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                            component.redshift = redshift

                            ##### parameters #####
                            param_df = {'parameter': [], 'value': [], 'minimum': [], 'maximum': [], 'frozen': []}
                            for pl, pv in component.pdicts.items():
                                param_df['parameter'].append(pl)
                                param_df['value'].append(float(pv.val))
                                param_df['minimum'].append(float(pv.min))
                                param_df['maximum'].append(float(pv.max))
                                param_df['frozen'].append(pv.frozen)
                            param_df = pd.DataFrame(param_df)
                            key = '%s_%s_param' % (mo_key, component_key); ini = param_df; set_ini(key, ini)
                            param_df = st.data_editor(get_data(key), column_config={'frozen': st.column_config.CheckboxColumn()}, 
                                                    use_container_width=True, num_rows='fixed', disabled=['parameter'], 
                                                    hide_index=True, key=key)
                            
                            for _, row in param_df.to_dict('index').items():
                                component.pdicts[row['parameter']].val = row['value']
                                component.pdicts[row['parameter']].min = row['minimum']
                                component.pdicts[row['parameter']].max = row['maximum']
                                component.pdicts[row['parameter']].frozen = row['frozen']

                    st.session_state.mo_component[mo_key][expr] = component

                with info_col:
                    st.write(''); st.write('')

                    key = '%s_%s_display' % (mo_key, component_key); ini = False; set_ini(key, ini)
                    if st.checkbox('Display model component shape', value=get_val(key), key=key):
                        if component_name is None:
                            st.warning('The model component is **NOT** set!', icon="⚠️")
                        elif component_name == 'user' and component is None:
                            st.warning('The user-defined model component has **NOT** been submitted!', icon="⚠️")
                        else:
                            with st.popover("Display settings", use_container_width=True):
                                key = '%s_%s_type' % (mo_key, component_key); ini = 'N(E)'; set_ini(key, ini)
                                type = st.select_slider('Select spectral type to display', options=['fv', 'N(E)', 'vfv'], value=get_val(key), key=key)

                                key = '%s_%s_erange' % (mo_key, component_key); ini = (0, 4); set_ini(key, ini)
                                erange = st.slider('Select energy range in logspace', -1, 5, get_val(key), key=key)
                                ebin = np.vstack((np.logspace(erange[0], erange[1], 300)[:-1],
                                                np.logspace(erange[0], erange[1], 300)[1:])).T
                                earr = np.array([np.sqrt(bin[0] * bin[1]) for bin in ebin])

                                key = '%s_%s_epoch' % (mo_key, component_key); ini = None; set_ini(key, ini)
                                epoch = st.text_input('Input spectral time point', value=get_val(key), placeholder='1.0', key=key)
                                if epoch == '' or epoch is None: tarr = None
                                if epoch is not None:
                                    try: epoch = float(epoch)
                                    except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                                    else: tarr = epoch * np.ones_like(earr)

                            if component_name == 'phabs' or component_name == 'tbabs':
                                ne = component.func(ebin, tarr).astype(float)
                                fv = ne; vfv = ne
                            else:
                                ne = component.func(earr, tarr).astype(float)
                                fv = 1.60218e-9 * earr * ne
                                vfv = 1.60218e-9 * earr * earr * ne
                            
                            sed = {'N(E)': ne, 'fv': fv, 'vfv': vfv}
                            sed_df = pd.DataFrame({'logE': np.log10(earr), 'log%s' % type: np.log10(sed[type])})
                            st.line_chart(sed_df, x='logE', y='log%s' % type, use_container_width=True)

        with expression_tab:
            set_col, _, info_col = st.columns([4.9, 0.2, 4.9])
            with set_col:
                info = """**Note: The model expression defines a combined model involved with multiple components, 
                which is also the model used by this model object.**"""
                st.info(info)

                key = '%s_expr' % mo_key; ini = None; set_ini(key, ini)
                val = get_val(key)
                if len(st.session_state.mo_component[mo_key].values()) == 1:
                    if list(st.session_state.mo_component[mo_key].values())[0] is not None:
                         val = list(st.session_state.mo_component[mo_key].keys())[0]
                placeholder = '+'.join(st.session_state.mo_component[mo_key].keys())
                expr = st.text_input('Input model expression', value=val, placeholder=placeholder, key=key)
                if expr == '': expr = None

                if expr is not None:
                    expr = re.sub('\s*', '', expr)
                    expr_sp = re.split(r"[(+\-*/)]", expr)
                    expr_sp = [ex for ex in expr_sp if ex != '']
                    if len(set(expr_sp)) < len(expr_sp):
                        st.warning('Sorry for prohibiting the use of the same component name!', icon="⚠️")
                    elif not (set(expr_sp) <= set(st.session_state.mo_component[mo_key].keys())):
                        st.warning('The model expression include invalid component name!', icon="⚠️")
                    elif None in [st.session_state.mo_component[mo_key][ex] for ex in expr_sp]:
                        st.warning('Some model components are **NOT** set!', icon="⚠️")
                    else:
                        st.session_state.mo[mo_key].set(expr, [st.session_state.mo_component[mo_key][ex] for ex in expr_sp])

                        ##### parameters #####
                        param_df = {'parameter': [], 'value': [], 'minimum': [], 'maximum': [], 'frozen': []}
                        for pl, pv in st.session_state.mo[mo_key].pdicts.items():
                            param_df['parameter'].append(pl)
                            param_df['value'].append(pv.val)
                            param_df['minimum'].append(pv.min)
                            param_df['maximum'].append(pv.max)
                            param_df['frozen'].append(pv.frozen)
                        param_df = pd.DataFrame(param_df)
                        key = '%s_param' % mo_key
                        param_df = st.data_editor(param_df, column_config={'frozen': st.column_config.CheckboxColumn()}, 
                                                  use_container_width=True, disabled=True, hide_index=True, key=key)
                        
                        for _, row in param_df.to_dict('index').items():
                            st.session_state.mo[mo_key].pdicts[row['parameter']].val = row['value']
                            st.session_state.mo[mo_key].pdicts[row['parameter']].min = row['minimum']
                            st.session_state.mo[mo_key].pdicts[row['parameter']].max = row['maximum']
                            st.session_state.mo[mo_key].pdicts[row['parameter']].frozen = row['frozen']

            with info_col:
                st.write(''); st.write('')

                key = '%s_display' % mo_key; ini = False; set_ini(key, ini)
                if st.checkbox('Display combined model shape', value=get_val(key), key=key):
                    if st.session_state.mo[mo_key].expr is None:
                        st.warning('The model expression is **NOT** set!', icon="⚠️")
                    else:
                        with st.popover("Display settings", use_container_width=True):
                            key = '%s_type' % mo_key; ini = 'N(E)'; set_ini(key, ini)
                            type = st.select_slider('Select spectral type to display', options=['fv', 'N(E)', 'vfv'], value=get_val(key), key=key)

                            key = '%s_erange' % mo_key; ini = (0, 4); set_ini(key, ini)
                            erange = st.slider('Select energy range in logspace', -1, 5, get_val(key), key=key)
                            ebin = np.vstack((np.logspace(erange[0], erange[1], 300)[:-1],
                                                np.logspace(erange[0], erange[1], 300)[1:])).T
                            earr = np.array([np.sqrt(bin[0] * bin[1]) for bin in ebin])

                            key = '%s_epoch' % mo_key; ini = None; set_ini(key, ini)
                            epoch = st.text_input('Input spectral time point', value=get_val(key), placeholder='1.0', key=key)
                            if epoch == '' or epoch is None: tarr = None
                            if epoch is not None:
                                try: epoch = float(epoch)
                                except: st.error('The input value needs to be able to be converted to float!', icon="🚨")
                                else: tarr = epoch * np.ones_like(earr)

                        ne, fv, vfv = st.session_state.mo[mo_key].func(ebin, tarr, st.session_state.mo[mo_key].expr)

                        key = '%s_comps' % mo_key; ini = None; set_ini(key, ini)
                        options = list(ne.keys())
                        comps = st.multiselect('Select the components to display', options=options, default=get_val(key), key=key)

                        if len(comps) > 0:
                            sed = {'logE': np.log10(earr)}
                            if type == 'N(E)':
                                for mi, si in ne.items():
                                    sed[mi] = np.log10(si)
                            if type == 'fv':
                                for mi, si in fv.items():
                                    sed[mi] = np.log10(si)
                            if type == 'vfv':
                                for mi, si in vfv.items():
                                    sed[mi] = np.log10(si)

                            sed_df = pd.DataFrame(sed)
                            st.line_chart(sed_df, x='logE', y=comps, use_container_width=True)
