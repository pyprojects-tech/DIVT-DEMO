import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm 
import matplotlib
from sklearn.metrics.pairwise import haversine_distances
import seaborn as sns
import random
import ast 

from datetime import datetime

st.set_page_config(layout="centered")
#plt.style.use('dark_background')
#plt.style.use('fivethirtyeight') 

#plt.style.use('dark_background')

#Start of App Including Type and Raw Data
st.title('DIVT Demo Capabilities Dashboard')
st.set_option('deprecation.showPyplotGlobalUse', False)

file =  pd.read_csv('cars.csv',low_memory=False)

with st.expander('Show Data Table'):
    st.dataframe(file)

stylelist = ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 
             'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 
             'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 
             'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 
             'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

styleselect = st.selectbox('Select Graph Style Option',options=stylelist,index=4)
plt.style.use(styleselect) 

params_num_col = file.select_dtypes(np.number)
params_cat_col = file.select_dtypes(object)

cat_select =['Year','Driveline','Fuel Type']
#Keep these for selection options
#num_params = st.multiselect('Numerical Data Selection', params_num_col.columns)
#cat_params = st.multiselect('Categorical Data Selection', params_cat_col.columns)

def histogram(keys):
    
    with st.expander("Graph Selection Parameters"):
        num_params = st.selectbox('Numerical Data Selection', params_num_col.columns,key=keys[0])
        hue_param = st.selectbox('Select Categorical Variables', cat_select,index=1,key=keys[1])
    plot = sns.histplot(data=file,x=num_params,kde=True,hue=hue_param)
    plot.set(title='Demo Histogram')
    fig = plot.get_figure()
    
    return st.pyplot(fig,clear_figure=True,key=keys[3])

def scatter(keys):
   
    with st.expander("Graph Selection Parameters"):
        x_param = st.selectbox('Select X-Axis Variable', params_num_col.columns,index=8,key=keys[0])
        y_param = st.selectbox('Select Y-Axis Variables', params_num_col.columns,index=4,key=keys[1])
        hue_param = st.selectbox('Select Categorical Variables', cat_select,index=1,key=keys[2])
    plot = sns.scatterplot(data=file,x=x_param, y=y_param,hue=hue_param)
    plot.set(title='Demo Scatter Plot')
    fig = plot.get_figure()
    
    return st.pyplot(fig,clear_figure=True,key=keys[3])

def heatmap(keys):
    
    with st.expander("Graph Selection Parameters"):
        #x_param = st.selectbox('Select X-Axis Variable', params_num_col.columns,index=8,key=keys[0])
        #y_param = st.selectbox('Select Y-Axis Variables', params_num_col.columns,index=7,key=keys[1])
        z_param = st.selectbox('Select Z-Axis Variables', params_num_col.columns,index=4,key=keys[2])
    file0 = file.pivot(index='Driveline',columns='Fuel Type',values=z_param)
    plot = sns.heatmap(file0,linewidths=.5)
    plot.set(title='Demo Scatter Plot')
    fig = plot.get_figure()
    
    return st.pyplot(fig,clear_figure=True,key=keys[3])

def correlation(keys):
   
    with st.expander("Graph Selection Parameters"):
        params = st.multiselect('Select Correlation Variables', params_num_col.columns,key=keys[0],default=['City mpg','Highway mpg'])
        hue_param = st.selectbox('Select Categorical Variables', cat_select,index=1,key=keys[1])
    file0 = file[np.append(params,hue_param)]
    plot = sns.pairplot(file0)
 
    return st.pyplot(clear_figure=True,key=keys[3])

def barplot(keys):
   
    with st.expander("Graph Selection Parameters"):
        #x_param = st.selectbox('Select X-Axis Variable', params_num_col.columns,index=8,key=keys[0])
        y_param = st.selectbox('Select Y-Axis Variables', params_num_col.columns,index=4,key=keys[1])
        #hue_param = st.selectbox('Select Categorical Variables', cat_select,index=1,key=keys[2])
    plot = sns.barplot(x=file['Year'], y=file[y_param],hue=file['Fuel Type'])
    sns.despine(offset=10, trim=True)
    
    return st.pyplot(clear_figure=True,key=keys[3])

def boxplot(keys):
   
    with st.expander("Graph Selection Parameters"):
        #x_param = st.selectbox('Select X-Axis Variable', params_num_col.columns,index=8,key=keys[0])
        y_param = st.selectbox('Select Y-Axis Variables', params_num_col.columns,index=4,key=keys[1])
        hue_param = st.selectbox('Select Categorical Variables', cat_select,index=1,key=keys[2])
    plot = sns.boxplot(x=file['Year'], y=file[y_param],hue=file['Fuel Type'])
    sns.despine(offset=10, trim=True)
    
    return st.pyplot(clear_figure=True,key=keys[3])

def jointplot(keys):
   
    with st.expander("Graph Selection Parameters"):
        x_param = st.selectbox('Select X-Axis Variable', params_num_col.columns,index=5,key=keys[0])
        y_param = st.selectbox('Select Y-Axis Variables', params_num_col.columns,index=4,key=keys[1])
        hue_param = st.selectbox('Select Categorical Variables', cat_select,index=1,key=keys[2])
    plot = sns.jointplot(x=file[x_param], y=file[y_param])
    
    return st.pyplot(clear_figure=True,key=keys[3])
col1,col2=st.columns(2)

with col1:
    keys=['1','2','3','4','5']
    #graph_type= st.selectbox('Select Graph Type', ['Scatter','Bar Plot','Histogram','Heatmap','Correlation'],key='graph1')
    st.header('Scatter Plot')
    scatter(keys)

with col2: 
    keys=['6','7','8','9','10']
    #graph_type2= st.selectbox('Select Graph Type', ['Scatter','Bar Plot','Histogram','Heatmap','Correlation'],key='graph2')
    st.header('Histogram')
    histogram(keys)

col11,col22=st.columns(2)
with col11:
    keys=['16','17','18','19','20']
    #graph_type= st.selectbox('Select Graph Type', ['Scatter','Bar Plot','Histogram','Heatmap','Correlation'],key='graph1')
    st.header('Box Plot')
    boxplot(keys)

with col22:
    keys=['11','12','13','14','15']
    #graph_type3= st.selectbox('Select Graph Type', ['Scatter','Bar Plot','Histogram','Heatmap','Correlation'],key='graph3')
    st.header('Bar Plot')
    barplot(keys)
    
col111,col222=st.columns(2)    
with col111:
    keys=['21','22','23','24','25']
    #graph_type= st.selectbox('Select Graph Type', ['Scatter','Bar Plot','Histogram','Heatmap','Correlation'],key='graph1')
    st.header('Joint Plot')
    jointplot(keys)
    
with col222:
    keys=['26','27','28','29','30']
    #graph_type= st.selectbox('Select Graph Type', ['Scatter','Bar Plot','Histogram','Heatmap','Correlation'],key='graph1')
    st.header('Correlation Plot')
    correlation(keys)