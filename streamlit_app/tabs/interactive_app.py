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

title = "Will it Rain Tomorrow at my Location in Australia?"
sidebar_name = "Interactive App"


def run():
    st.title(title)
    st.markdown("---")
    st.markdown(
    """
    Here you can ask for a weather prediction for any date and location in 
    Australia that come to your mind.
    """
    )
    
    form = st.form(key='app-input-form')
    input_date = form.date_input("Please enter the :blue[**date**], for which\
 you want to predict, if it will rain on the next day.\nThe date must be in\
 the interval [2015-05-12, 2017-06-25]", dt.date(2015, 5, 27))
    input_location = form.text_input("Please enter the :blue[**location**],\
 for which you want to predict, if it will rain on the next day.\n The format\
 is not restricted. You can name for instance a town, monument, or a full\
 address.",
                                     "Great Barrier Reef")
    submit = form.form_submit_button('Submit')
    
    if submit:
        input_date = input_date.strftime('%Y-%m-%d')
        pred_output,pred_plot,avail_data = predict_weather(input_date,
                                                           input_location)
        st.markdown("""
        **The output of the app for your inputs above is given here:**
        """)
        st.write(pred_output)
        st.pyplot(pred_plot)
        plt.clf()
        st.markdown("""
        **The weather prediction is based on the data shown in the table below:**
        """)
        st.write(avail_data)
        st.markdown("""
        **Feel free to play with the input data fields at the top of this tab.**
        """)
    st.markdown("---")
