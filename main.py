import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import math
import glob
import ipywidgets as widgets
import zipfile
import os
import pvlib
import rdtools
import statsmodels.api as sm
import scipy as sc
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore')
from IPython.display import clear_output
from FDD_preprocessing import *
from FDD_simulation import *
from PIL import Image

logo = Image.open('3s.png')
st.set_page_config(
     page_title="Fault Detection and Diagnosis Algorithm",
     page_icon=logo,
     layout="wide" )

l1, l2 = st.beta_columns([10,3])
l2.image('Logos_3S_EPFL_PVlab.png')



h1, h2 = st.beta_columns([1, 9])
h1.image('Logo_red.png')

with h2:
    st.title('Fault Detection and Diagnosis Algorithm')
    
f_cs = None
year = None
info = pd.read_csv('fault_info.csv', error_bad_lines=False, sep = ';').replace(np.nan, '-')


file_upload = st.beta_container()
fault_info = st.beta_container()
figures = st.beta_container()

@st.cache
def select_year(data_file, metadata_file):
    datetime_format = '%d/%m/%Y %H:%M'
    data = pd.read_csv(data_file)
    data.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
    data['Time'] = pd.to_datetime(data['Time'], format=datetime_format)
    years=  data.Time.dt.year.unique()
    return years
    

with file_upload:
    st.markdown("""---""")
    
    st.header('Upload files')
    st.markdown('Upload a csv file with the DC PV system data (Timestamps, Voltage, Current, Power).')
    data_file = st.file_uploader('PV data', '.csv', )
    st.markdown('Upload a csv file with the PV system metadata.')
    metadata_file = st.file_uploader('PV metadata', '.csv', )
    
    if ((data_file is not None) and (metadata_file is not None)):
        years =  select_year(data_file, metadata_file)
        st.write("##")
        para1, para2 = st.beta_columns(2)
        para1.markdown('Please select the year of analysis.')
        year = para1.selectbox(label = 'Year of analysis', options = list(years))
        
        para2.markdown('Please select the sample frequency of the analysis.')
        freq = para2.radio(label = 'Fequency', options = ['5min', '10min', '15min', '30min', '1h'])
        
        ready = st.button('Analysis')

    
    st.markdown("""---""")
    
    st.header('Fault Analysis Results')
    if (f_cs is None):
        st.markdown('This can take up to a few minutes.')
 
    
    
    
import plotly.graph_objects as go





with figures:
    
    
        
    if ((data_file is not None) and (metadata_file is not None) and (year is not None) and (ready)):
         f_cs, f_fault, f_current, f_power,voltage_compare_fig, f_dist, F0, F1, F2, F3, F4, F5, SS, S = FFDA_run_all(data_file, metadata_file, year, freq) 
            

            

    if (f_cs is not None):
        st.subheader('Clear-sky power simulation')
        st.markdown('Clear-sky DC power output simulation based on single-diode model compared to the actual DC power output.')
        st.plotly_chart(f_cs, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', # one of png, svg, jpeg, webp'filename': 'custom_image',
        'height': 500,'width': 900,'scale': 1}})
        
        st.subheader('FDDA threshold visualisation')
        st.markdown('Full visualisation of FDDA outputs - Simulated and actual DC outputs, fault occurence colormap and relative current loss heatmap.')
        st.plotly_chart(voltage_compare_fig,use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', # one of png, svg, jpeg, webp'filename': 'custom_image',
        'height': 500,'width': 900,'scale': 1}})
        
        
        
        p1, p2 = st.beta_columns(2)
        p1.subheader('Shading fault heatmap')
        p1.markdown('Shading faults colormap showing the temporal distribution of faults. ')
        p1.plotly_chart(f_fault,use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'saved_image','width': 700,'scale': 1}})
        p2.subheader('Current loss heat map')
        p2.markdown('Relative current loss heatmap, computed by comparing actual and clear-sky outputs.')
        p2.plotly_chart(f_current,use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'saved_image','width': 700,'scale': 1}})
        
        
        k1, k2 = st.beta_columns(2)
        k1.subheader('Shading fault occurence histogram')
        k1.markdown('Histogram of fault occurence as a function of the relative time of day. The values of the x-axis represent normalised hours in a day, given the differences in day length during the year.')
        k1.plotly_chart(f_dist,use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'saved_image','width': 700,'scale': 1}})
        
        
        labels = ['F2 - Low V, low I','F3 - Low V', 'F4 - High V, low I', 'F5 - High V']
        values = [F1, F2, F3, F4]
        fig_prop = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, sort = False)])
        colors = ['steelblue', 'rgb(57,86,143)', 'rgb(31,150,139)', 'teal']
        fig_prop.update_traces(marker=dict(colors=colors))
        
        k2.subheader('Fault type distribution')
        k2.markdown('Relative occurence of fault types in the system. The fault classification and probable causes is shown in the table below.')
        k2.plotly_chart(fig_prop,use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'saved_image','width': 700,'scale': 1}})
        
      
        #st.table(info)
        
        st.image('Fault_table.png') 

