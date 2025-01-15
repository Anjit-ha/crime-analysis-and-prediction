# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:02:48 2023
 
@author: USER
"""

from sklearn import preprocessing 
import streamlit as st
import pandas as pd
#import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
loaded_model = pickle.load(open('C:\\Users\\USER\\Documents\\MP\\project\\model.pkl','rb'))
df = pd.read_csv("C:\\Users\\USER\\Documents\\MP\\project\\Clustered_Train.csv")
cr=pd.read_csv("C:\\Users\\USER\\Documents\\MP\\project\\01_District_wise_crimes_committed_IPC_2001_2012.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Crime Analysis and Prediction")
nav = st.sidebar.radio("Navigation",["Home","Prediction","Data"])
if nav == "Home":
    image=Image.open("C:\\Users\\USER\\Documents\\MP\\project\\Blue Purple Futuristic Virus Hacks Youtube Thumbnail (1).png")
    st.image(image)
    st.markdown('Crime is an alarming aspect of our society, and its prevention is a vital task. Crime analysis is a well-organised way of detecting and examining patterns and trends in crime. It is of most importance to study reason, consider different factors and determine the relationship among various crimes occurring and discover the best suitable methods to control crime.')
    
if nav == "Prediction":
    st.write ("Prediction")    
    with st.form("my_form"):
        state=st.text_input(label='STATE/UT')
        district=st.text_input(label='DISTRICT')
        year=st.text_input(label='YEAR')

    
        data=[[state,district,year]]
        submitted = st.form_submit_button("Predict Safe/Unsafe")

    if submitted:
        clust = loaded_model.predict(data)[0]
        cluster_g = df[df['SAFE/UNSAFE'] == clust]
        st.write("The district is",clust)


      
if nav == "Data":
    if st.checkbox("Show Original Dataset"):
        file_ = pd.read_csv("C:\\Users\\USER\\Desktop\\MP\\01_District_wise_crimes_committed_IPC_2001_2012.csv")
        st.table(file_)
    
    if st.checkbox("Show Clustered Dataset"):
        file = pd.read_csv("C:\\Users\\USER\\Desktop\\MP\\Clustered_Train.csv")
        st.table(file)