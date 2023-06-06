import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.basemap import Basemap
#import seaborn as sns
#from pandas.plotting import scatter_matrix
#import bokeh
#from geopy import distance 
#from geopy.geocoders import Nominatim
#import geopy
import sys
sys.path.append('../')
from notebooks.app_JP import predict_weather

title = "Conclusions and Outlook"
sidebar_name = "Conclusions and Outlook"


def run():
    st.title(title)
    st.markdown("---")
    
    st.markdown("""
    - We achieve the :blue[**best possible performance**] given the :blue[**limitations of the available data**] (low spatial / temporal resolution) by:
        - Exploring :blue[**extensive feature engineering**] routes (e.g. features based on weighted-average over nearby weather stations).
        - Implementing :blue[**Machine- and simple Deep-Learning models**].
    - With a :blue[**Random Forest Classifier**], we are able to :blue[**predict**] if it will :blue[**rain on the next day**] with an :blue[**accuracy of 0.86**], a macro-averaged F1-score of 0.77, and a weighted-averaged F1-score of 0.85 :blue[**on the test data set**].
    - Based on the results, the project can be improved and extended in the :blue[**future**]:
        - Interactive streamlit app:
            - Use :blue[**webscrapping**] to be able to run this model for dates outside of the given date interval.
            - Use a weighted average over nearby stations for the prediction.
            - Increase :blue[**user-friendliness**], e.g. by including an interactive map / plot using Bokeh.
        - Data and model:
            - Use meteorological :blue[**data of higher resolution in time and space**], e.g. gridded satellite data with an hourly and 100km resolution, respectively.
            - Implement a :blue[**sophisticated recurrent LSTM-based neural network or a physics-informed neural network**] (including meteorological and climatological equations).
            - Predict other meteorological variables (e.g. temperature) and do :blue[**longterm predictions**].
    """)#<span style="color:#1f77b4">some *blue* text</span>
    st.markdown("---")
