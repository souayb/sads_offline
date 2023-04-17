import pathlib
from contextlib import suppress
import json
from itertools import count
from collections import Counter
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os 
import pickle 

import zipfile
from io import BytesIO
from datetime import datetime

# sk-learn model import 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import utils 
import pandas as pd
from matplotlib.colors import ListedColormap
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
from os.path import exists as file_exists

# """
# pip install streamlit-aggrid
# """

save_path = 'sads_data'
with suppress(FileExistsError):
        os.mkdir(save_path)
import base64
# caching.clear_cache()
st.set_page_config(layout="wide") # setting the display in the 

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#         button[data-baseweb="tab"] {font-size: 26px;}
#         </style>
#         """


hide_menu_style = """
        <style>
        footer {visibility: hidden;}
      
        button[data-baseweb="tab"] {font-size: 26px;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

SMALL_SIZE = 3
MEDIUM_SIZE =3
BIGGER_SIZE = 3
# plt.rcParams['figure.figsize'] = (5, 10)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE, dpi=600)  # fontsize of the figure title
plt.style.context('bmh')
new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> SADS: Shop-floor Anomaly Detection Service: Offl`ine mode </p> </h2></center>'

import base64
import shutil


def create_download_zip(zip_directory, zip_path, filename='foo.zip'):
    """ 
        zip_directory (str): path to directory  you want to zip 
        zip_path (str): where you want to save zip file
        filename (str): download filename for user who download this
    """
    shutil.make_archive(zip_path, 'zip', zip_directory)
    with open(zip_path, 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{filename}\'>\
            download file \
        </a>'
        st.markdown(href, unsafe_allow_html=True)

st.markdown(new_title, unsafe_allow_html=True)

st.cache(suppress_st_warning=True)
# @st.experimental_memo(suppress_st_warning=True)
def data_reader(dataPath:str) -> pd.DataFrame :
    df = pd.read_csv(dataPath, decimal=',')
    prepro = utils.Preprocessing()
    data = prepro.preprocess(df)
    # data = data[['BarCode', 'Face', 'Cell', 'Point', 'Group' , 'Output Joules' , 'Charge (v)', 'Residue (v)', 'Force L N','Force L N_1', 'Y/M/D hh:mm:ss']]
    data.rename(columns={'BarCode':'Barcode', 'Output Joules': 'Joules', 'Charge (v)':'Charge', 'Residue (v)':'Residue','Force L N':'Force_N', 'Force L N_1':'Force_N_1', 'Y/M/D hh:mm:ss': 'Datetime'}, inplace=True)
    data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']] = data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].apply(np.float32)
    data[['Face', 'Cell', 'Point']] = data[['Face', 'Cell', 'Point']].values.astype( int )
    JOULES = data['Joules'].values 
    return data[['Barcode', 'anomaly', 'Face', 'Cell', 'Point','Face_Cell_Point','Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'ts']]           

st.cache()
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Define function to zip folder
def zip_folder(folder_path):
    # Create in-memory zip file
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zip_file:
    # with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))
    # Seek to beginning of buffer
    zip_buffer.seek(0)
    return zip_buffer



# ########################## PREDICTION FORM #######################################
# SADA_settings = st.sidebar.form("SADS")
# SADA_settings.title("SADS settings")

SADS_CONFIG_FILE = 'sads_config.json'

SADS_CONFIG = {}
JOULES = []
SHIFT_DETECTED = False
SHIFT_RESULT = []
RESULT_CHANGED = False

RESULTING_DATAFRAME = pd.DataFrame()


st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_logger(save:bool=True):
    """
    Generic utility function to get logger object with fixed configurations
    :return:
    logger object
    """
    SADS_CONFIG['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    SADS_CONFIG['drift_detected'] = SHIFT_DETECTED
    SADS_CONFIG['Joules'] = JOULES
    SADS_CONFIG['drift_result'] = SHIFT_RESULT
    SADS_CONFIG['result_change'] =  RESULT_CHANGED

    if not os.path.exists(SADS_CONFIG_FILE):
        # Create the file
        with open(SADS_CONFIG_FILE, 'w') as outfile:
            json.dump(SADS_CONFIG, outfile)
    if save:
        with open(SADS_CONFIG_FILE, 'w') as outfile:
            json.dump(SADS_CONFIG, outfile)
    else:
        with open(SADS_CONFIG_FILE) as infile:
            return json.load(infile)

with st.sidebar.container():
    st.title("SADS settings input") 
    training_type =     st.radio(
            "Apply on: 👇",
            ["Pack", "Whole"],
            disabled=False,
            # horizontal= True,
        )

    with st.form('Saving setting'):
        
        with st.expander('Model saving input'):
            st.subheader("Save following SADS results")
            check_left, check_right = st.columns(2)
            pack_download = check_left.checkbox('pack images', value=True )
            table_download = check_right.checkbox('The table', value=True)
            chart_download = check_left.checkbox('The chart', value=True)
        save_submit = st.form_submit_button('Download')
        # with st.container():
        #     with st.expander('Explanation'):
        #         st.write("Writing explanation text")

        # save_submit = st.form_submit_button('Download')
    with st.form('Input setting'):
        with st.expander('Model control'):
            # with st.form("Models"):
            st.subheader("SADS models")
            check_left, check_right = st.columns(2)
            model_ifor = check_left.checkbox('Isolation forest', value=True )
            model_lof = check_left.checkbox('Local Outlier Factor', value=False)
            model_repeat = check_left.checkbox('Repeat', value=False)
            model_gmm = check_right.checkbox('Gaussian Mixture', value=False)
            model_bgmm = check_right.checkbox('Bayesian gaussian Mixture', value=False)
            model_svm = check_right.checkbox('One Class SVM', value=False)
            # train_ = st.form_submit_button("Apply")

        # st.subheader("Table control input")
        with st.expander("Table control"):
        # with st.form('my_form'): 
            st.subheader("Table setting")
            sample_size = st.number_input("rows", min_value=10, value=30)
            grid_height = st.number_input("Grid height", min_value=200, max_value=800, value=300)

            return_mode = st.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
            return_mode_value = DataReturnMode.__members__[return_mode]

            # update_mode = st.selectbox("Update Mode", list(GridUpdateMode.__members__), index=len(GridUpdateMode.__members__)-1)
            # update_mode_value = GridUpdateMode.__members__[update_mode]

            #enterprise modules
            enable_enterprise_modules = st.checkbox("Enable Enterprise Modules")
            if enable_enterprise_modules:
                enable_sidebar =st.checkbox("Enable grid sidebar", value=False)
            else:
                enable_sidebar = False
            #features
            fit_columns_on_grid_load = st.checkbox("Fit Grid Columns on Load")

            enable_selection=st.checkbox("Enable row selection", value=True)

            if enable_selection:
                
                # st.sidebar.subheader("Selection options")
                selection_mode = st.radio("Selection Mode", ['single','multiple'], index=1)

                use_checkbox = st.checkbox("Use check box for selection", value=True)
                if use_checkbox:
                    groupSelectsChildren = st.checkbox("Group checkbox select children", value=True)
                    groupSelectsFiltered = st.checkbox("Group checkbox includes filtered", value=True)

                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    rowMultiSelectWithClick = st.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
                    if not rowMultiSelectWithClick:
                        suppressRowDeselection = st.checkbox("Suppress deselection (while holding CTRL)", value=False)
                    else:
                        suppressRowDeselection=False
                st.text("___")

            enable_pagination = st.checkbox("Enable pagination", value=False)
            if enable_pagination:
                st.subheader("Pagination options")
                paginationAutoSize = st.checkbox("Auto pagination size", value=True)
                if not paginationAutoSize:
                    paginationPageSize = st.number_input("Page size", value=5, min_value=0, max_value=sample_size)
                st.text("___")

        with st.expander('Plot control'):
            st.subheader("Plot setting")
            chart_left, chart_right = st.columns(2)
            show_joules = chart_left.checkbox('Joules', value=True)
            show_force_n = chart_left.checkbox('Force right', value=False)
            show_pairplot = chart_left.checkbox('Pairplot', value=False)
            show_force_n_1 = chart_right.checkbox('Force left', value=False)
            show_residue = chart_right.checkbox('Residue', value=False)
            show_charge = chart_right.checkbox('Charge', value=False)


        submitted = st.form_submit_button('Apply')


uploaded_files = st.file_uploader("Choose a CSV file" )
if uploaded_files is not None:

    if pathlib.Path ( uploaded_files.name ).suffix not in ['.csv', '.txt']:
        st.error ( "the file need to be in one the follwing format ['.csv', '.txt'] not {}".format (
            pathlib.Path ( uploaded_files.name ).suffix ) )
        raise Exception ( 'please upload the right file ' )

    with st.spinner('Wait for preprocess and model training'):
        st.info('Preporcessing started ')
        data = data_reader(uploaded_files)
        new_joule = data['Joules'].values 
        st.success('Preprocessing complete !')
        if not os.path.exists(SADS_CONFIG_FILE):
            JOULES = new_joule.tolist()
            get_logger(save=True)
            IF = pickle.load(open('/app/model.pkl', 'rb'))
        else : 
            SADS_CONFIG = get_logger(save=False)
            # SHIFT_RESULT = SADS_CONFIG['drift_result']
            # set_trace()
            # testing drift
            # set_trace()
            to_test = np.hstack([np.array(SADS_CONFIG['Joules'][:500]), new_joule[:500]])
            test_resutl = utils.pettitt_test(to_test, alpha=0.8)
            if test_resutl.cp >= 500 and test_resutl.cp <= 502: 
                st.write("DRIFT FOUND NEED THE RETRAIN THE MODEL")
                JOULES = new_joule.tolist()
                SHIFT_DETECTED = True
                get_logger(save=True)
                if training_type=='Whole':
                    with st.spinner('Training...: This may take some time'):
                        IF = utils.train_model(data=data)
                        pickle.dump(IF, open('model.pkl', 'wb'))
                        st.success('Training completed !')
                    
            else : 
                # JOULES = new_joule.tolist()
                # get_logger(save=True)
                st.write(" NO DRIFT FOUND")
                IF = pickle.load(open('model.pkl', 'rb'))

    init_options = data['Barcode'].unique().tolist()
    if 'options' not in st.session_state:
        st.session_state.options = init_options
    if 'default' not in st.session_state:
        st.session_state.default = []
    # print('initial option', st.session_state.options)

    ms = st.multiselect(
        label='Pick a Barcodef',
        options=st.session_state.options,
        default=st.session_state.default
    )
    DDDF = st.empty()
    Main = st.empty()
    # day_left, time_right = Main.columns(2)
    pack_view, table_view, chart_view = st.tabs(["Pack", "🗃Table", "📈 Charts"])
        # Example controlers
    
    if ms:
        # print('we are in ms', ms)
        pack_path = os.path.join(save_path, ms[-1])
        with suppress(FileExistsError) or suppress(FileNotFoundError):
            os.makedirs(pack_path)
        if ms in st.session_state.options:
            st.session_state.options.remove(ms[-1])
            st.session_state.default = ms[-1]
            st.experimental_rerun()
        pack_data = data[data['Barcode']== ms[-1]]

        ## TRAINING THE MODEL
        if SHIFT_DETECTED:
            if training_type == 'Pack':
                if model_ifor:
                    ifor = utils.train_model(pack_data, model_type='ifor')
                    ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

                if model_gmm :
                    gmm = utils.train_model(pack_data, model_type='gmm')
                    gmm_cluster = gmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['gmm_anomaly']  =  gmm_cluster 
                    pack_data['gmm_anomaly']  =  pack_data['gmm_anomaly'].astype(bool)

                if model_bgmm :
                    bgmm = utils.train_model(pack_data, model_type='bgmm')
                    bgmm_cluster = bgmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['bgmm_anomaly']  =  bgmm_cluster 
                    pack_data['bgmm_anomaly']  =  pack_data['bgmm_anomaly'].astype(bool)

                if model_lof:
                    lof = utils.train_model(pack_data, model_type='lof')
                    lof_cluster = lof.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['lof_anomaly']  =  lof_cluster 
                    pack_data['lof_anomaly']  =  pack_data['lof_anomaly'].astype(bool)

                if model_svm:
                    svm = utils.train_model(pack_data, model_type='svm')
                    svm_cluster = svm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['svm_anomaly']  =  svm_cluster 
                    pack_data['svm_anomaly']  =  pack_data['svm_anomaly'].astype(bool)

            else :
                if model_ifor:
                    ifor = utils.train_model(data, model_type='ifor')
                    ifor_cluster = ifor.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    data['ifor_anomaly']  =data['ifor_anomaly'].astype(bool)

                if model_gmm :
                    gmm = utils.train_model(data, model_type='gmm')
                    gmm_cluster = gmm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['gmm_anomaly']  =  gmm_cluster 
                    data['gmm_anomaly']  =  data['gmm_anomaly'].astype(bool)

                if model_bgmm :
                    bgmm = utils.train_model(data, model_type='bgmm')
                    bgmm_cluster = bgmm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['bgmm_anomaly']  =  bgmm_cluster 
                    data['bgmm_anomaly']  =  data['bgmm_anomaly'].astype(bool)

                if model_lof:
                    lof = utils.train_model(data, model_type='lof')
                    lof_cluster = lof.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['lof_anomaly']  =  lof_cluster 
                    data['lof_anomaly']  =  data['lof_anomaly'].astype(bool)

                if model_svm:
                    svm = utils.train_model(data, model_type='svm')
                    svm_cluster = svm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['svm_anomaly']  =  lof_cluster 
                    data['svm_anomaly']  =  data['svm_anomaly'].astype(bool)
        else:

            if training_type == 'Pack':
                if model_ifor:
                    ifor = utils.train_model(pack_data, model_type='ifor')
                    ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

                if model_gmm :
                    gmm = utils.train_model(pack_data, model_type='gmm')
                    gmm_cluster = gmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['gmm_anomaly']  =  gmm_cluster 
                    pack_data['gmm_anomaly']  =  pack_data['gmm_anomaly'].astype(bool)

                if model_bgmm :
                    bgmm = utils.train_model(pack_data, model_type='bgmm')
                    bgmm_cluster = bgmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['bgmm_anomaly']  =  bgmm_cluster 
                    pack_data['bgmm_anomaly']  =  pack_data['bgmm_anomaly'].astype(bool)

                if model_lof:
                    lof = utils.train_model(pack_data, model_type='lof')
                    lof_cluster = lof.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['lof_anomaly']  =  lof_cluster 
                    pack_data['lof_anomaly']  =  pack_data['lof_anomaly'].astype(bool)

                if model_svm:
                    svm = utils.train_model(pack_data, model_type='svm')
                    svm_cluster = svm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['svm_anomaly']  =  svm_cluster 
                    pack_data['svm_anomaly']  =  pack_data['svm_anomaly'].astype(bool)

                # RESULTING_DATAFRAME = pd.concat([RESULTING_DATAFRAME,pack_data])
                


        with table_view:        
            gb = GridOptionsBuilder.from_dataframe(pack_data)

            cellsytle_jscode = JsCode("""
            function(params) {
                if (params.value == 0) {
                    
                    return {
                        'color': 'white',
                        'backgroundColor': 'darkred'
                    }
                } else {
                    return {
                        'color': 'black',
                        'backgroundColor': 'white'
                    }
                }
            };
            """)
            gb.configure_column("ifor_anomaly", cellStyle=cellsytle_jscode)

            if enable_sidebar:
                gb.configure_side_bar()

            if enable_selection:
                gb.configure_selection(selection_mode)
                if use_checkbox:
                    gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

            if enable_pagination:
                if paginationAutoSize:
                    gb.configure_pagination(paginationAutoPageSize=True)
                else:
                    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

            gb.configure_grid_options(domLayout='normal')
            gridOptions = gb.build()

            #Display the grid``
            # print(f" mss {ms[-1]} -- {type(ms[-1])}")
            st.header(f"Table view : -- {ms[-1]}") 
            st.markdown("""
                This is the table view of the battery pack filtered using the Barcode
            """)

            grid_response = AgGrid(
                pack_data, 
                gridOptions=gridOptions,
                height=grid_height, 
                width='100%',
                data_return_mode=return_mode_value, 
                # update_mode=update_mode_value,
                update_mode=GridUpdateMode.MANUAL,
                fit_columns_on_grid_load=fit_columns_on_grid_load,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                enable_enterprise_modules=enable_enterprise_modules
                )
            # grid_table = AgGrid(face_1_df_1 , height=500,  gridOptions=gridoptions,
            #                                 update_mode=GridUpdateMode.SELECTION_CHANGED, allow_unsafe_jscode=True)
            #             selected_row = grid_table["selected_rows"]
            #             st.table(selected_row)
            # df = grid_response['data']
            # selected = grid_response['selected_rows']
            if table_download:
                table_save = os.path.join(pack_path, 'table_vew.csv')
                pack_data.to_csv(table_save)
        with chart_view :
            st.header(f"Chart view : -- {ms[-1]}") 
            if model_ifor:
                with st.expander("ISOLATION FOREST"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['ifor_anomaly'] = pack_data['ifor_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='ifor_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='ifor_anomaly' , names='ifor_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='ifor_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='ifor_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='ifor_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='ifor_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )

                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            pack_data['ifor_anomaly'] = pack_data['ifor_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'ifor_anomaly']], diag='box',index='ifor_anomaly', 
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )
            if model_gmm:
                with st.expander("GAUSSIAN MIXTURE"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['gmm_anomaly'] = pack_data['gmm_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='gmm_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='gmm_anomaly' , names='gmm_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='gmm_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='gmm_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='gmm_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='gmm_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )

                    
                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            pack_data['gmm_anomaly'] = pack_data['gmm_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'gmm_anomaly']], diag='box',index='gmm_anomaly', 
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )


            if model_bgmm:
                with st.expander("BAYESIAN GAUSSIAN MIXTURE"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['bgmm_anomaly'] = pack_data['bgmm_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='bgmm_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='bgmm_anomaly' , names='bgmm_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='bgmm_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='bgmm_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='bgmm_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='bgmm_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )


                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            pack_data['bgmm_anomaly'] = pack_data['bgmm_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'bgmm_anomaly']], diag='box',index='bgmm_anomaly', 
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )

            if model_lof:
                with st.expander("LOCAL OUTLIER FACTOR"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['lof_anomaly'] = pack_data['lof_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='lof_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='lof_anomaly' , names='lof_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='lof_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='lof_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='lof_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='lof_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )
                    
                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            pack_data['lof_anomaly'] = pack_data['lof_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'lof_anomaly']], diag='box',index='lof_anomaly', 
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )

            if model_svm:
                with st.expander("One-Class SVM"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['svm_anomaly'] = pack_data['svm_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='svm_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='svm_anomaly' , names='svm_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='svm_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='svm_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='svm_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='svm_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )
                    
                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            pack_data['svm_anomaly'] = pack_data['svm_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'svm_anomaly']], diag='box',index='svm_anomaly', 
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )
            

            if model_repeat:
                with st.expander("REPEAT FROM MACHINE"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['anomaly'] = pack_data['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='anomaly' , names='anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )
                    

                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            pack_data['anomaly'] = pack_data['anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'anomaly']], diag='box',index='anomaly', 
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )
            

            
        with pack_view :
                 
            st.header(f"Pack view : -- {ms[-1]}")

            pack_data_non_dup = pack_data[~pack_data.duplicated(subset=['Barcode', 'Face', 'Cell', 'Point'], keep= 'last')]
            pack_data_dup = pack_data[pack_data.duplicated(subset=['Barcode',  'Face', 'Cell', 'Point'], keep= 'last')]

            face_1 = np.ones ( shape=(14, 16) ) * 0.
            face_2 = np.ones ( shape=(14, 16) ) * 0.

            face_1_maske = np.ones ( shape=(14, 16) )
            face_2_maske = np.ones ( shape=(14, 16) )

            face_1_repeat = np.zeros ( shape=(14, 16) )
            face_2_repeat = np.zeros ( shape=(14, 16) )

            face_1_repeat_mask = np.ones ( shape=(14, 16) )
            face_2_repeat_mask = np.ones ( shape=(14, 16) )

            colorscale = [[0.0, 'rgb(169,169,169)'],
                        [0.5, 'rgb(0, 255, 0)'],
                        [1.0, 'rgb(255, 0, 0)']]
            time_plot_1_1 = 0
            time_plot_1_2 = 0
            time_plot_2_1 = 0
            time_plot_2_2 = 0
            plot_count_1 = 0
            plot_count_1 = 0
            plot_count_2 = 0
            # pack = data[data['Barcode']== ms[0]]

            if model_ifor:
                with st.expander("ISOLATION FOREST"):
                    pack_face1, pack_face2 = st.columns(2)

                    pack_data_non_dup['ifor_anomaly'] = pack_data_non_dup['ifor_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df = pack_data_non_dup[pack_data_non_dup['Face']==2]

                    face_1_df_1_val = face_1_df_1['ifor_anomaly'].values
                    face_1_df_1_val = face_1_df_1_val.reshape(-1, 16)

                    face_1_df_2_val = face_1_df_2['ifor_anomaly'].values
                    face_1_df_2_val = face_1_df_2_val.reshape(-1, 16)

                    face_2_df_1_val = face_2_df_1['ifor_anomaly'].values
                    face_2_df_1_val = face_2_df_1_val.reshape(-1, 16)

                    face_2_df_2_val = face_2_df_2['ifor_anomaly'].values
                    face_2_df_2_val = face_2_df_2_val.reshape(-1, 16)


                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    face_1[0::2,:] = face_1_df_1_val
                    face_1_maske[0::2,:] = False
                    face_1[1::2,:] = face_1_df_2_val
                    face_1_maske[1::2,:] = False

                    face_2[0::2,:] = face_2_df_1_val
                    face_2_maske[0::2,:] = False
                    face_2[1::2,:] = face_2_df_2_val
                    face_2_maske[1::2,:] = False


                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[0].set_title ( "Face 1" )
                        # cbar_kws={
                        #     'pad': .001,
                        #     'ticks': [0, 1],
                        #     "shrink": 0.01
                        # },
                                
                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[0].set_title ( "Face 2" )
                    sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[1].set_title ( "Reapeted face 1" )
                    sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    pack_face1.pyplot ( fig_pack_1)#, use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2)#, use_container_width=True )
                    if pack_download:
                        ifor_face1 = os.path.join(pack_path, 'ifor_face1')
                        ifor_face2 = os.path.join(pack_path, 'ifor_face2')
                        fig_pack_1.savefig(ifor_face1)
                        fig_pack_2.savefig(ifor_face2)


            if model_gmm:
                with st.expander("GAUSSIAN MIXTURE"):
                    pack_face1, pack_face2 = st.columns(2)
                    pack_data_non_dup['gmm_anomaly'] = pack_data_non_dup['gmm_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df = pack_data_non_dup[pack_data_non_dup['Face']==2]

                    face_1_df_1_val = face_1_df_1['gmm_anomaly'].values
                    face_1_df_1_val = face_1_df_1_val.reshape(-1, 16)

                    face_1_df_2_val = face_1_df_2['gmm_anomaly'].values
                    face_1_df_2_val = face_1_df_2_val.reshape(-1, 16)

                    face_2_df_1_val = face_2_df_1['gmm_anomaly'].values
                    face_2_df_1_val = face_2_df_1_val.reshape(-1, 16)

                    face_2_df_2_val = face_2_df_2['gmm_anomaly'].values
                    face_2_df_2_val = face_2_df_2_val.reshape(-1, 16)


                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    face_1[0::2,:] = face_1_df_1_val
                    face_1_maske[0::2,:] = False
                    face_1[1::2,:] = face_1_df_2_val
                    face_1_maske[1::2,:] = False

                    face_2[0::2,:] = face_2_df_1_val
                    face_2_maske[0::2,:] = False
                    face_2[1::2,:] = face_2_df_2_val
                    face_2_maske[1::2,:] = False


                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[0].set_title ( "Face 1" )
                        # cbar_kws={
                        #     'pad': .001,
                        #     'ticks': [0, 1],
                        #     "shrink": 0.01
                        # },
                                
                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[0].set_title ( "Face 2" )
                    sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[1].set_title ( "Reapeted face 1" )
                    sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    pack_face1.pyplot ( fig_pack_1,)# use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2,)# use_container_width=True )

                    if pack_download:
                        gmm_face1 = os.path.join(pack_path, 'gmm_face1')
                        gmm_face2 = os.path.join(pack_path, 'gmm_face2')
                        fig_pack_1.savefig(gmm_face1)
                        fig_pack_2.savefig(gmm_face2)

            if model_repeat:
                with st.expander("GAUSSIAN MIXTURE"):
                    pack_face1, pack_face2 = st.columns(2)
                    pack_data_non_dup['anomaly'] = pack_data_non_dup['anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df = pack_data_non_dup[pack_data_non_dup['Face']==2]

                    face_1_df_1_val = face_1_df_1['anomaly'].values
                    face_1_df_1_val = face_1_df_1_val.reshape(-1, 16)

                    face_1_df_2_val = face_1_df_2['anomaly'].values
                    face_1_df_2_val = face_1_df_2_val.reshape(-1, 16)

                    face_2_df_1_val = face_2_df_1['anomaly'].values
                    face_2_df_1_val = face_2_df_1_val.reshape(-1, 16)

                    face_2_df_2_val = face_2_df_2['anomaly'].values
                    face_2_df_2_val = face_2_df_2_val.reshape(-1, 16)


                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    face_1[0::2,:] = face_1_df_1_val
                    face_1_maske[0::2,:] = False
                    face_1[1::2,:] = face_1_df_2_val
                    face_1_maske[1::2,:] = False

                    face_2[0::2,:] = face_2_df_1_val
                    face_2_maske[0::2,:] = False
                    face_2[1::2,:] = face_2_df_2_val
                    face_2_maske[1::2,:] = False


                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[0].set_title ( "Face 1" )
                        # cbar_kws={
                        #     'pad': .001,
                        #     'ticks': [0, 1],
                        #     "shrink": 0.01
                        # },
                                
                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[0].set_title ( "Face 2" )
                    sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[1].set_title ( "Reapeted face 1" )
                    sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    pack_face1.pyplot ( fig_pack_1,)# use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2,)# use_container_width=True )
                    if pack_download:
                        repeat_face1 = os.path.join(pack_path, 'repeat_face1')
                        repeat_face2 = os.path.join(pack_path, 'repeat_face2')
                        fig_pack_1.savefig(repeat_face1)
                        fig_pack_2.savefig(repeat_face2)

#### SAVING THE DATE INTO THE LOCAL MACHINE 

    if save_submit:
        # Define folder to zip
     
        # Zip the folder
        zip_file = zip_folder(save_path)

        # Download the zipped folder using Streamlit
        st.sidebar.download_button(
            label="Download zipped folder",
            data=zip_file.getvalue(),
            file_name="my_zipped_folder.zip",
            mime="application/zip"
        )

 