### import public python packages
import sys
import os
import numpy as np
import pandas as pd
import itertools
# import tabulate as tab

# time
import datetime as dt
from time import time
from tqdm import tqdm

# machine learning
import sklearn
import pickle
import joblib
#import interactions
#import scikitplot
import xgboost
import imblearn

# math / stats
import scipy.special
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import statsmodels.api

# plotting
from IPython.display import display, Markdown, Latex
import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from pandas.plotting import scatter_matrix
import bokeh
from geopy import distance 
from geopy.geocoders import Nominatim
import geopy


def data_preprocessing(df,
                       nearby_locations = 0., # int/float: radius[km]/corr thr
                       nan_treatment_features = "mean_mode_all",
                        # mean_mode_all, mean_mode_local, drop
                       scaling = "std", # std, minmax, None
                       sampling = "None", #None, over, under
                       select_location = "None",  # data of selected locations
                       save_output="None"): # save data to fname in data subdir
    """
    input parameters:
    -df = dataframe to preprocess (should be the weather_AUS.csv dataset from
     kaggle/BOM)
     
    -nearby_locations [int or float]
    = parameter used to calculate the (ToDo: weighted!?-)
     mean of all features within: correlation higher than a threshold given as
      float (0. to 1.) or a given distance (in units of km) as int
     -default value is 0. for now, but should be optimized
     -possible values:
      -int value giving the distance/radius in km, within which the locations
       should be considered for calculating the mean.
      -float value giving a threshold of minimum correlation between locations
       (with respect to Rainfall feature), above which locations are used for
       calculating the average
      -The mean of the features is added to the dataframe
       (ToDo: to be implemented. think about: replace features with the mean
       instead of adding them as new features?)
       
    -nan_treatment_features ["mean_mode_all", "mean_mode_local", "drop"]
    = select the strategy for replacing Nan values (does not affect dropping
      of target variable and related features (RainTomorrow, RainToday,
      Rainfall))
    -default value is "mean_mode_all"
    -possible values:
     -"mean_mode_all": replace NaN of each feature by the mean (mode in case of
      discrete variables) over all rows, thus over all locations and all times
      in training dataset (not recommended)
     -"mean_mode_local": replace NaN of each feature by the average of nearby
      locations (nearby_locations int/float must be given in this case) for the
      same date (if an average feature has a NaN value for this date, the row
      will be dropped)
     -"drop": drop all rows with NaNs
      a
      
    -scaling ["std", "minmax", "None"]
    = define which scaling should be used on all features:
     -default value is "std"
     -possible values:
      -"std": StandardScaler
      -"minmax": MinMaxScaler
      -"None": do not use a Scaler
      
    -sampling ["None", "over", "under"]
    = define a way to counteract the imbalance in the target variable
     -default value is "None", i.e. no sampling is applied
     -possible values:
      -"over": use oversampling
      -"under": use undersampling
      -"None": no sampling
    
    -select_location ["None"] or (list) of locations to be selected
    = (list of) location(s) to select from the overall data, all other location
     data is dropped: use e.g. MelbourneAirport (location with least NaN rows)
     to model only this specific location
     -default value is "None": all locations are used
     -possible values:
      -"None"
      -name, name= name of locations to be selected (from "Location" column)
    
    -save_output ["None"] or str giving filename to save output to
    = filename of file in data subdirectory, that is created to save the output
     -default value is "None": output just returned, not saved to file
     -possible values:
      -"None"
      -"filename"
    
    description:
    -Convert variables in df to more useful variables for the modeling later on
    -Add new varibles by feature engineering
    -drop irrelevant variables
    -define features and target variable
    -split data into train, test set: take care to sort data by date first
    -treat NaNs: replace/drop
    -scale the variable ranges
    -Changes are made directly on the df given to the function
    
    note:
    -Nan replacement and scaling should be done only on the training data,
     as taking a mean/mode for NaN replacement on test data is an information
     leak. Also scaling should only be done on training data.
    
    returns:
    -df: after preprocessing (concatenation of X_train/test and y_train/test),
     but the scaling is reversed so the values of df are more human readable.
     Scaled data is found in X_train, ...:
    -X_train, X_test, y_train, y_test:
     RainTomorrow is y, other vars are features (change if modeling goal 
     changes (e.g. prediction of amount of rain, or more days in the future))
    """
    
    """
    first part of preprocessing:
    -replacing categorical / string variables by numerical variables
    -feature engineering to obtain more useful features
    """
    display("initial/original data shape:", df.shape)
    # change type of Date to datetime format
    df.Date = pd.to_datetime(df.Date, format = "%Y-%m-%d", exact = True)
    # create new date variables
    df["Day"] = df.Date.dt.day
    df["Month"] = df.Date.dt.month
    df["Year"] = df.Date.dt.year
    
    # add coordinates to locations including altitude and number/identifier of
    # nearest bureau station
    df_temp = pd.DataFrame({"Location":df.Location.unique(),
 "NearestBureauStationNr" : ["072160", "067108", "048027", "059151", "053115",
"061055", "061366", "200288", "067113", "067105", "066214", "066037", "072150",
"061078", "068228", "070351", "070339", "070349", "089002", "081123", "085072",
"086282", "086338", "076031", "078015", "090171", "086068", "090194", "040913",
"031011", "040764", "032040", "023034", "026021", "023373", "016001", "009500",
"109521", "009053", "009021", "009225", "012071", "009998", "094029", "091237",
"015590", "014015", "014932", "015635"],
 "Location_Latitude" : [36.07, 33.90, 31.48, 30.32, 29.49, 32.92, 33.28, 29.04,
33.72, 33.60, 33.86, 33.95, 35.16, 32.79, 34.37, 35.31, 35.42, 35.53, 37.51,
36.74, 38.12, 37.67, 37.83, 34.24, 36.31, 38.31, 37.74, 37.92, 27.48, 16.87,
27.94, 19.25, 34.95, 37.75, 34.48, 31.16, 35.03, 34.03, 31.67, 31.93, 31.92,
32.99, 34.95, 42.89, 41.42, 23.80, 12.42, 14.52, 25.19],
 "Location_Longitude" : [146.95, 150.73, 145.83, 153.12, 149.85, 151.8, 151.58,
167.94, 150.68, 150.78, 151.20, 151.17, 147.46, 151.84, 150.93, 149.20, 149.09,
148.77, 143.79, 144.33, 147.13, 144.83, 144.98, 142.09, 141.65, 141.47, 145.10,
141.26, 153.04, 145.75, 153.43, 146.77, 138.52, 140.77, 139.01, 136.81, 117.88,
115.06, 116.02, 115.98, 115.87, 121.62, 116.72, 147.33, 147.12, 133.89, 130.89,
132.38, 130.97],
 "Location_Elevation" : [164, 81, 260, 4, 213, 33, 19, 112, 25, 19, 43, 6, 212,
8, 10, 577, 587, 1760, 435, 209, 5, 113, 8, 50, 139, 81, 66, 51, 8, 2, 3, 4, 2,
63, 275, 167, 3, 85, 40, 15, 25, 249, 73, 51, 5, 546, 30, 134, 492]})
    df = pd.merge(df, df_temp, on='Location', how='left')
    del df_temp
    # towns with many occurences in Australia such that the station might
    # be wrong:
    # Richmond, (Mount Ginini, Portland, PearceRAAF, Perth
    # richmond RAAF 067105  33.60 150.78 19
    # richmond post office 030045 20.73 143.14  211
    #display(Markdown(df_temp.to_markdown(index=False)))
    
    # replace wind direction labels by angles (in radians from 0 to 2*pi):
    for i in df[["WindGustDir","WindDir9am","WindDir3pm"]]:
        df[i]=df[i].replace(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"],
                            [j*np.pi/8 for j in range(16)])
    # create two new variables representing the wind direction as cosinus and
    # sinus part
    for i in df[["WindGustDir","WindDir9am","WindDir3pm"]]:
        df[str(i)+"_cos"] = np.round(np.cos(df[i]),2)
        df[str(i)+"_sin"] = np.round(np.sin(df[i]),2)
    
    # replace RainX by numerical value:
    # float type, cause there are NaN in vars: cannot create int type (yet)
    df.RainToday=df.RainToday.replace(["Yes","No"],[1,0])
    df.RainTomorrow=df.RainTomorrow.replace(["Yes","No"],[1,0])
    
    """
    2nd part of preprocessing:
    -adding features based on weighted-mean from surrounding stations
    -train/test split
    -NAN treatment (removal, simple replacement by mode/mean or sophisticated
     replacement by local (in space and time) mean)
    -scaling (standardscaler or minmaxscaler)
    -sampling (counteracting imbalance)
    """
    
    # remove locations with few entries: Katherine, Nhil and Uluru:
    df = df[(df.Location!="Katherine") & (df.Location!="Nhil")
            & (df.Location!="Uluru")]
    display("data shape after removing Locations with few entries:",df.shape)
    
    # sort values by date, before train test split
    #display(df.head(20))
    df = df.sort_values(by='Date').reset_index(drop=True)
    #display(df.head(20))
    
    # remove some object type vars and old vars not needed anymore
    df = df.drop(["WindGustDir",
                "WindDir9am", "WindDir3pm"],axis=1)
    
    # remove all entires with NaN in target variable and related variables:
    # (RainToday and Rainfall)
    df = df.dropna(axis = 0, how = 'any', subset = ['RainToday','RainTomorrow',
                                                  'Rainfall'])
    df.RainToday = df.RainToday.astype(int)
    df.RainTomorrow = df.RainTomorrow.astype(int)
    display("data shape after removing rows with NaN in target variables:",
            df.shape)    
    
    # determine locations that have at least one feature, that contains only
    # NaN values (no meaningful data at all) and remove these locations from
    # the df, as the location cannot be modeled with good performance:
    # These NaN features are typically found to be most important for
    # a randomforest classifier presented in report 2 (feature_importance)).
    # Note that there are still locations with large NaN contents in some
    # features, which have to be filled by nearby station data or dropped.
    locs_with_missing_feature = df.groupby("Location").agg(lambda x:\
                                                 x.isna().sum()/x.shape[0]//1)
    #print(locs_with_missing_feature)
    locs_with_missing_feature = [locs_with_missing_feature.index[i]\
                 for i in np.unique(np.where(locs_with_missing_feature==1)[0])]
    df = df[~df.Location.isin(locs_with_missing_feature)]
    del locs_with_missing_feature
    #display(df.Location.unique())
    display("data shape after removing locations with at least one feature\
     containing exclusively NaN values:", df.shape)  
    
    # remove all entries before 01.01.2009, as most stations do not have data
    # for these early years:
    locs_dates_min_max = df.groupby("Location").agg({"Date" : [min, max]})
    print(locs_dates_min_max.astype(str).sort_values(\
        [("Date", "min"), "Location"]).to_markdown())
    del locs_dates_min_max
    df = df.loc[df.Date > "2008-12-31"]
    display("data shape after removing data from before 01.01.2009:", df.shape)
    print("NaN fractions [%] of most important features (according to\
     RandomForest of report2) per location")
    print(df.groupby("Location").agg(lambda x: np.round(x.isna().sum()\
                                                        /x.shape[0]*100,1))\
          [["Sunshine", "Humidity3pm", "Cloud3pm", "Pressure3pm",\
            "WindGustSpeed", "Pressure9am"]].to_markdown())
    NaNs_per_row = df.isna().sum(axis=1)
    for index in NaNs_per_row.index:
        if NaNs_per_row[index] > 0 : NaNs_per_row[index] = 1
    locs_fraction_of_rows_w_NaN = dict.fromkeys(df.Location.unique())
    for location in df.Location.unique():
        locs_fraction_of_rows_w_NaN[location] = np.round(\
            1 - (len(df[df.Location == location])\
             - NaNs_per_row.where(df.Location == location).sum())\
             /len(df[df.Location == location]),3)
    del NaNs_per_row
    locs_fraction_of_rows_w_NaN = pd.DataFrame(
                                locs_fraction_of_rows_w_NaN.items(),
                                columns=["Location", "Fraction_of_rows_w_NaN"])
    locs_fraction_of_rows_w_NaN.sort_values("Fraction_of_rows_w_NaN",
                                            inplace=True)
    locs_fraction_of_rows_w_NaN.set_index("Location", inplace = True)
    print("Fraction of entries with at least one feature having a NaN value\
     per Location")
    print(locs_fraction_of_rows_w_NaN.to_markdown())
    del locs_fraction_of_rows_w_NaN
    
    if nearby_locations != 0.:
        """
        calculate weighted-average for each feature including nearby stations:
        - distance metric with weight = 1/(distance + 200)*200
        -> 100km 2/3 of central weight
            - all stations within 'nearby_locations'[km] radius or closest one,
              if no station is located within given radius
        - correlation of Rainfall metric with weight = corr coeff
            - all stations with corr. coeff. > threshold [0,1] or largest one,
              if no station has a corr. coeff. above threshold
        - action: df extended by averaged features (excludes time data)
        """
        if nearby_locations < 1.: # use corr. coeff. between stations as metric
            # create df with dates as rows and location names as columns
            # entries == values of  Rainfall for location of corresponding col.
            df_date_vs_locs_temp = []
            variable = "Rainfall"
            for (i,location) in enumerate(df.Location.unique()):
                temp = pd.DataFrame(df.loc[df.Location == location,
                                                           ["Date", variable]])
                temp.columns = ["Date", location]
                df_date_vs_locs_temp.append(temp)
            del temp
            df_date_vs_locs = df_date_vs_locs_temp[0]
            for i in range(0, len(df_date_vs_locs_temp) - 1):
                df_date_vs_locs = df_date_vs_locs.merge(\
                             right = df_date_vs_locs_temp[i+1], on = "Date",\
                             how = "outer")
            del df_date_vs_locs_temp
            df_date_vs_locs = df_date_vs_locs.sort_values(by="Date")\
               .reindex(["Date", *sorted(df_date_vs_locs.columns[1:])], axis=1)
            #display(df_date_vs_locs)
            # calculate correlation coefficients for all stations with respect
            # to the 'variable' (Rainfall)
            # and replace corr coeffs. below threshold by zero (set weight of 
            # stations outside of the defined corr. coeff. range to zero)
            loc_corrs = np.round(df_date_vs_locs.iloc[:,1:].corr(), 1)
            # ToDo: new check if correct!
            for central_loc in loc_corrs.index:
                temp = list(loc_corrs.index[np.where(\
                        loc_corrs.loc[central_loc,:] >= nearby_locations)])
                if len(temp) <= 1: # if no station above corr. coeff. thr.
                    temp = df_date_vs_locs.iloc[:,1:].corr() # more precision
                     # needed (loc_corrs was defined with np.round(...,1)),
                     # otherwise more than one station will be chosen
                     # take station w/ largest cor coef and station itself->(2)
                    loc_corrs.loc[central_loc,:] = np.where(\
                       temp.loc[central_loc,:].isin(temp.\
                       loc[central_loc,:].nlargest(2)),\
                       loc_corrs.loc[central_loc,:], 0.)
                else:
                    loc_corrs.loc[central_loc,:] = np.where(\
                        loc_corrs.loc[central_loc,:] >= nearby_locations,\
                        loc_corrs.loc[central_loc,:], 0.)
            loc_weights = loc_corrs # save obtained weight df (loc vs loc)
            del loc_corrs, df_date_vs_locs, temp
        
        elif nearby_locations >= 1: # distance[km] between stations as metric:
            # create df w/ distances as vals and Locations as index and column
            Location_unique_indices = df.reset_index().groupby(['Location'])\
                ['index'].min().to_list()
            names = df.loc[Location_unique_indices, "Location"]
            lats = df.loc[Location_unique_indices, "Location_Latitude"]
            lons = df.loc[Location_unique_indices, "Location_Longitude"]
            #elevs = df.loc[Location_unique_indices,"Location_Elevation"]
            distances_2D = np.zeros((len(names), len(names)))
            for i in range(len(names)):
                for j in range(len(names)):
                    distances_2D[i][j] = distance.distance([-lats.values[i],
                        lons.values[i]], [-lats.values[j], lons.values[j]]).km
            dist_metric = pd.DataFrame(distances_2D,
                                       columns = names, index = names)
            del lats, lons, distances_2D, Location_unique_indices, names
            # ToDo: new check if correct!
            # replace distance values with weight (anti-prop to distance)
            # replace distances larger than threshold with 0. (weight = zero).
            # if no location within given radius (<threshold): use next closest
            # station
            for central_loc in dist_metric.index:
                temp = list(dist_metric.index[np.where(\
                        dist_metric.loc[central_loc,:] < nearby_locations)])
                if len(temp) <= 1:
                    # use loc and closest loc
                    dist_metric.loc[central_loc,:] = np.where(\
                       dist_metric.loc[central_loc,:].isin(dist_metric.\
                       loc[central_loc,:].nsmallest(2)),\
                       1./(dist_metric.loc[central_loc,:] + 200)*200,0.)
                else:
                    dist_metric.loc[central_loc,:] = np.where(\
                        dist_metric.loc[central_loc,:] < nearby_locations,\
                        1./(dist_metric.loc[central_loc,:] + 200)*200,0.)
            loc_weights = dist_metric
            del dist_metric, temp
        print(loc_weights.to_markdown())
        # ToDo: check for bugs in weighted average calculation!
        # - do this more efficiently!
        # - save preprocessed data (also scaler!)
        # add new "avg_..." features to the df and fill their values with the
        # weighted-averages for each location and date over the nearby locs
        # found in the previous code blocks (also use weights from there)
        Location_unique_indices = df.reset_index().groupby(['Location'])\
                ['index'].min().to_list()
        names = df.loc[Location_unique_indices, "Location"]
        del Location_unique_indices
        feats_to_be_averaged = ["Location_Latitude", "Location_Longitude",
            "Location_Elevation", "MinTemp","MaxTemp", "Evaporation",
            "Sunshine", "WindGustSpeed", "WindGustDir_cos", "WindGustDir_sin",
            "WindDir9am_cos", "WindDir9am_sin", "WindDir3pm_cos",
            "WindDir3pm_sin", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
            "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
            "Cloud3pm", "Temp9am", "Temp3pm", "Rainfall", "RainToday"]
        for name in feats_to_be_averaged:
            df["avg_"+name] = np.nan
        for location in tqdm(names):
            for date in df[df["Location"] == location].Date:
                # create small dataframe "temp" containing all entries for the
                # currently treated date
                # calculate the weighted-average for each feature on this
                # temporary dataframe by using weights found previously
                temp = df[df["Date"] == date].sort_values(by = "Location")
                for feature in feats_to_be_averaged:
                    df.loc[temp.Location[temp.Location == location].index,\
                           "avg_"+feature] = np.ma.average(\
                        np.ma.MaskedArray(temp[feature],\
                                          mask = np.isnan(temp[feature])),\
                        weights = loc_weights.loc[location,\
                                                  temp.Location.values].values)
        df.loc[df.avg_Rainfall > 0,"avg_RainToday"] = 1
        df.avg_RainToday = df.avg_RainToday.astype(int)
        del loc_weights, names, temp, feats_to_be_averaged
        #display(df.sort_values(by=["Date","Location"]))
    
    if nan_treatment_features == "drop":
        df = df.dropna(axis = 0, how = 'any')
        display("data shape after NaN treatment", df.shape)
    elif nan_treatment_features == "mean_mode_local":
        for NaN_col in df.columns:
            if NaN_col in ["Location_Latitude", "Location_Longitude",
            "Location_Elevation", "MinTemp","MaxTemp", "Evaporation",
            "Sunshine", "WindGustSpeed", "WindGustDir_cos", "WindGustDir_sin",
            "WindDir9am_cos", "WindDir9am_sin", "WindDir3pm_cos",
            "WindDir3pm_sin", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
            "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
            "Cloud3pm", "Temp9am", "Temp3pm"]:
                df[NaN_col].fillna(df["avg_" + NaN_col], inplace=True)
        df = df.dropna(axis = 0, how = 'any')
        display("data shape after NaN treatment", df.shape)
    
    if select_location == "None": pass
    else:
        if isinstance(select_location,str): select_location = [select_location]
        df = df.loc[df.Location.isin(select_location)]
    
    df = df.drop(["Date", "Location", "NearestBureauStationNr"], axis=1)
    """
    df=df.drop(["Day","Month","Year","Location_Latitude", "Location_Longitude",
    "Location_Elevation", "MinTemp","MaxTemp", "Evaporation", "WindGustSpeed",
    "WindGustDir_cos", "WindGustDir_sin", "WindDir9am_cos","WindDir9am_sin",
    "WindDir3pm_cos", "WindDir3pm_sin","WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Pressure9am","Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "Rainfall","RainToday","Date","Location",
    "NearestBureauStationNr"],axis=1)
    df=df.loc[:,["Humidity3pm", "Sunshine", "RainTomorrow"]]
    """
    
    # change the order of the variables/columns for better structure of df
    if nearby_locations != 0.:
        df = df.loc[:,["Day","Month","Year",
                       "Location_Latitude", "avg_Location_Latitude",
                       "Location_Longitude", "avg_Location_Longitude",
                       "Location_Elevation", "avg_Location_Elevation",
                       "MinTemp", "avg_MinTemp",
                       "MaxTemp", "avg_MaxTemp",
                       "Evaporation", "avg_Evaporation",
                       "Sunshine", "avg_Sunshine",
                       "WindGustSpeed", "avg_WindGustSpeed",
                       "WindGustDir_cos", "avg_WindGustDir_cos",
                       "WindGustDir_sin", "avg_WindGustDir_sin",
                       "WindDir9am_cos", "avg_WindDir9am_cos",
                       "WindDir9am_sin", "avg_WindDir9am_sin",
                       "WindDir3pm_cos", "avg_WindDir3pm_cos",
                       "WindDir3pm_sin", "avg_WindDir3pm_sin",
                       "WindSpeed9am", "avg_WindSpeed9am",
                       "WindSpeed3pm", "avg_WindSpeed3pm",
                       "Humidity9am", "avg_Humidity9am",
                       "Humidity3pm", "avg_Humidity3pm",
                       "Pressure9am", "avg_Pressure9am",
                       "Pressure3pm", "avg_Pressure3pm",
                       "Cloud9am", "avg_Cloud9am",
                       "Cloud3pm", "avg_Cloud3pm",
                       "Temp9am", "avg_Temp9am",
                       "Temp3pm", "avg_Temp3pm",
                       "Rainfall", "avg_Rainfall",
                       "RainToday", "avg_RainToday",
                       "RainTomorrow"]]
    else:
        df = df.loc[:,["Day","Month","Year", "Location_Latitude",
                       "Location_Longitude", "Location_Elevation", "MinTemp",
                       "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed",
                       "WindGustDir_cos", "WindGustDir_sin", "WindDir9am_cos",
                       "WindDir9am_sin", "WindDir3pm_cos", "WindDir3pm_sin",
                       "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
                       "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
                       "Cloud3pm", "Temp9am", "Temp3pm", "Rainfall",
                       "RainToday", "RainTomorrow"]]
    display("data shape after reordering the columns", df.shape)
    display(df)
    # ToDo change data/target variables, in case the modeling goal changes
    data = df.drop('RainTomorrow', axis=1)
    target = df.RainTomorrow
    #display(df.head(),df.tail())
    X_train, X_test, y_train, y_test = sklearn.model_selection.\
    train_test_split(data, target, test_size = 0.2,
                     random_state = 120, shuffle = False)
    #display(X_train.iloc[0],X_train.iloc[-1])
    #display(X_test.iloc[0],X_test.iloc[-1])
    if nan_treatment_features == "mean_mode_all":
        # replace NaN in discrete variables with the mode:
        if nearby_locations != 0.:
            print("wrong value in 'nan_treatment_features' parameter given.\n\
             Due to the choice of nearby_locations>0 only possible values are:\
             mean_mode_local, drop")
            return -1
        for i in ["Cloud9am",
                  "Cloud3pm", "WindGustDir_cos", "WindGustDir_sin",
                  "WindDir9am_cos", "WindDir9am_sin", "WindDir3pm_cos",
                  "WindDir3pm_sin"]:
            X_train_mode_i = X_train[i].mode()[0]
            X_train[i] = X_train[i].fillna(X_train_mode_i)
            X_test[i] = X_test[i].fillna(X_train_mode_i)
        del X_train_mode_i
        
        # replace NaN in continuous variables with the mean
        for i in ["MinTemp", "MaxTemp", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"]:
            X_train_mean_i = X_train[i].mean()
            X_train[i] = X_train[i].fillna(X_train_mean_i)
            X_test[i] = X_test[i].fillna(X_train_mean_i)
        del X_train_mean_i
    elif (nan_treatment_features in ["drop", "mean_mode_local"]):
        pass
    else:
        print("wrong value in 'nan_treatment_features' parameter given.\n\
              Possible values are:\n mean_mode_all, mean_mode_local, drop")
        return -1
    
    if scaling == "std":
        cols = X_train.columns
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns = cols)
        cols = X_test.columns
        X_test = pd.DataFrame(scaler.transform(X_test), columns = cols)
    elif scaling == "minmax":
        cols = X_train.columns
        scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns = cols)
        cols = X_test.columns
        X_test = pd.DataFrame(scaler.transform(X_test), columns = cols)
    elif scaling == "None":
        pass
    else:
        print("wrong value in 'scaling' parameter given. Possible values are:\
              \n std, minmax, None")
        return -1
    
    if sampling == "over":
        cols_X = X_train.columns
        cols_y = y_train.name
        X_train, y_train = imblearn.over_sampling.RandomOverSampler(
            random_state = 120).fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns = cols_X)
        y_train = pd.Series(y_train, name = cols_y)
        # sort X_train and y_train by Date, as the sampling affects the sorting
        temp = pd.concat([X_train,y_train],axis=1)
        temp = temp.sort_values(by=["Year", "Month", "Day"])
        X_train = temp.drop("RainTomorrow",axis=1)
        y_train = temp.RainTomorrow
        del temp
    elif sampling == "under":
        cols_X = X_train.columns
        cols_y = y_train.name
        X_train, y_train = imblearn.under_sampling.RandomUnderSampler(
            random_state = 120).fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns = cols_X)
        y_train = pd.Series(y_train, name = cols_y)
        # sort X_train and y_train by Date, as the sampling affects the sorting
        temp = pd.concat([X_train,y_train],axis=1)
        temp = temp.sort_values(by=["Year", "Month", "Day"])
        X_train = temp.drop("RainTomorrow",axis=1)
        y_train = temp.RainTomorrow
        del temp
    elif sampling == "None":
        pass
    else:
        print("wrong value in 'sampling' parameter given. Possible values are:\
              \n over, under, None")
        return -1
    
    columns = df.columns
    if ((scaling == "std") or (scaling == "minmax")):
        df = pd.DataFrame(np.column_stack((np.vstack((
                                            scaler.inverse_transform(X_train),
                                            scaler.inverse_transform(X_test))),
                                            np.concatenate((y_train,y_test)))),
                          columns = columns)
    else:
        df = pd.DataFrame(np.column_stack((np.vstack((X_train,X_test)),
                                           np.concatenate((y_train,y_test)))),
                          columns = columns)
    del columns
    display("final data shape:", df.shape)
    if save_output == "None": pass
    else:
        df.to_pickle("../data/" + save_output + "_df.pkl")
        X_train.to_pickle("../data/" + save_output + "_Xtrain.pkl")
        X_test.to_pickle("../data/" + save_output + "_Xtest.pkl")
        y_train.to_pickle("../data/" + save_output + "_ytrain.pkl")
        y_test.to_pickle("../data/" + save_output + "_ytest.pkl")
        if ((scaling == "std") or (scaling == "minmax")):
            pickle.dump(scaler, open("../data/" + save_output + "_scaler.pkl", 
                                 'wb'))
    return df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # usually df's with many cols/vars/features are not displayed completely,
    # omit this by: showing all columns of a df:
    pd.set_option('display.max_columns', None)
    """
    print("Example for using the preprocessing function defined above:")
    df=pd.read_csv("../data/weatherAUS.csv")
    df, X_train, X_test, y_train, y_test = \
        data_preprocessing(df,
                       nearby_locations = 0.3,
                       nan_treatment_features = "mean_mode_local",
                       scaling = "std",
                       sampling = "over",
                       save_output = "03corr_NaNrep_local_std_over")
    display(df)
    """
    df=pd.read_csv("../data/weatherAUS.csv")
    df, X_train, X_test, y_train, y_test = \
        data_preprocessing(df,
                       nearby_locations = 100,
                       nan_treatment_features = "mean_mode_local",
                       scaling = "std",
                       sampling = "over",
                       save_output = "100km_NaNrep_local_std_over")
    display(df)
    """
    df=pd.read_csv("../data/weatherAUS.csv")
    df, X_train, X_test, y_train, y_test = \
        data_preprocessing(df,
                       nearby_locations = 300,
                       nan_treatment_features = "mean_mode_local",
                       scaling = "std",
                       sampling = "over",
                       save_output = "300km_NaNrep_local_std_over")
    display(df)
    df=pd.read_csv("../data/weatherAUS.csv")
    df, X_train, X_test, y_train, y_test = \
        data_preprocessing(df,
                       nearby_locations = 0.5,
                       nan_treatment_features = "mean_mode_local",
                       scaling = "std",
                       sampling = "over",
                       save_output = "05corr_NaNrep_local_std_over")
    display(df)
    df=pd.read_csv("../data/weatherAUS.csv")
    df, X_train, X_test, y_train, y_test = \
    data_preprocessing(df,
                       nearby_locations = 100,
                       select_location = "MelbourneAirport",
                       nan_treatment_features = "mean_mode_local",
                       scaling = "std",
                       sampling = "over",
                       save_output = "MelbourneAirport_100km_\
NaNrep_local_std_over")
    display(df)
    """
    print("do the modeling on the obtained train and test sets.")
    
    print("to read preprocessed data from previously created pkl files w/o\
     running the preprocessing function again, use this code and replace\
     'save_output' with the root name of the respective files:")
    """
    save_output="alicesprings_03_nanlocal_std_over"
    df_reload = pd.read_pickle("../data/"+save_output+"_df.pkl")
    X_train_reload = pd.read_pickle("../data/"+save_output+"_Xtrain.pkl")
    X_test_reload = pd.read_pickle("../data/"+save_output+"_Xtest.pkl")
    y_train_reload = pd.read_pickle("../data/"+save_output+"_ytrain.pkl")
    y_test_reload = pd.read_pickle("../data/"+save_output+"_ytest.pkl")
    """


"""
np.savez_compressed(path_to_data+fname,
    data=data_array,	...)
reloaded_file=np.load("/data/""+str(fname)+".npz")
data=reloaded_file["data"]
reloaded_file.close()

#use df.Date.dt.date to get only date (not time HH:MM:SS)
# dtype('<M8[ns]') or as str(): 'datetime64[ns]'
# handeling date and time data:
#pd.to_datetime(df.Date).describe(datetime_is_numeric=True)#
#dt.datetime(2017,1,17)
#df["year_added"] = pd.to_datetime(df["date_added"]).dt.year.astype(int)
#pdSeries_datetime_type.year / month / day to get respective part of date
#transactions['day'] = transactions['tran_date'].apply(
# lambda date: date.split('-')[0])
#df["weekday"]=pd.to_datetime(
# df[["month","day","year"]].astype(str).agg("-".join, axis=1)).dt.weekday

#from tqdm import tqdm
#tqdm.pandas()
#df.progress_apply("kk")
def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]
#print(pearsonr(empty_df.dropna(),empty_df.dropna()))

list_location_raintomorrow_equal={}
amnt_of_days=np.array(len(df.Location.unique()),len(df.Location.unique()))
def locations(x):
    for i in x.index:
        for j in x.index:
            if df.RainTomorrow[j]==df.RainTomorrow[i]: 
                list[df.Location[i],df.Location[j]]+=1
            amnt_of_days[df.Location[i],df.Location[j]]+=1
df.groupby("Date").agg({"RainTomorrow" : locations})

# code template/examples to get distances between locations and calculate mean
of locations nearby.

distance_2d=[distance.distance([-lats.values[a],lons.values[a]],
 [-lats.values[b],lons.values[b]]).km for b in range(0,len(lats.values))]

geolocator = Nominatim(user_agent="specify_your_app_name_here")
location = geolocator.geocode("175 5th Avenue NYC")
# or reverse:
location = geolocator.reverse("52.509669, 13.376294")
print(location.address, location.latitude, location.longitude, location.raw)

Location_unique_indices=df.reset_index().groupby(['Location'])\
['index'].min().to_list()
lats = df.loc[Location_unique_indices,"Location_Latitude"]
lons = df.loc[Location_unique_indices,"Location_Longitude"]
elevs = df.loc[Location_unique_indices,"Location_Elevation"]

geolocator = Nominatim(user_agent="my_locator")
locations = [geolocator.geocode(location+", Australia")\
 for location in df.Location.unique()]
locations = [geolocator.reverse(geopy.point.Point(lat, lon))\
 for lat,lon in zip (-lats.values,lons.values)]
locs_lats=[]
locs_lons=[]
for i in range(len(locations)):
    try:
        locs_lats.append(locations[i].latitude)
        locs_lons.append(locations[i].longitude)
    except:
        print("no long/lat found for Location",i,":",df.Location.unique()[i])
print("Looks like Locations not separated by spaces are not found:\
 add spaces using regex? (ToDo)")
"""