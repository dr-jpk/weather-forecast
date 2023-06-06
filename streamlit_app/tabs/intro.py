import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import base64
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.basemap import Basemap
#import seaborn as sns
#from pandas.plotting import scatter_matrix
#import bokeh
#from geopy import distance 
#from geopy.geocoders import Nominatim
#import geopy

title = "Weather Forecast in Australia"
sidebar_name = "Presentation of our Project"


def run():
    st.image("../data/plots/nearby_locations_test_.png")
    st.title(title)
    st.markdown("---")
    
    st.markdown(
    """
    Here, we present our weather forecasting project to you. The project is developed in the course of a data science bootcamp of [DataScientest](https://datascientest.com/).

    We created a presentation in pdf format to present our project. You can choose to display the presentation in an embedded form within streamlit or open it with a pdf viewer of your choice.
    """
    )
    fname="Presentation_KK_JP_Kevin_7_0.pdf"
    with open("assets/"+fname,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Embed presentation below'):
            pdf_display = F'<embed src="data:application/pdf;base64,\
            {base64_pdf}" width="700" height="1000" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
    with col2:
        if st.button('Open presentation in pdf viewer'):
            pdf_display = F'<iframe src="data:application/pdf;base64,\
            {base64_pdf}" width="700" height="1000" type="application/pdf">\
            </iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    st.markdown("---")