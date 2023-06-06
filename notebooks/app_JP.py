import numpy as np
import pandas as pd
import datetime as dt
import sklearn
import pickle
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from geopy import distance 
from geopy.geocoders import Nominatim
import geopy



def predict_weather(input_date, input_location):
    """
    Obtaines geographical location from given string (city name, organization,
    address...) and finds next neighboring weather station in dataset for a
    given date.
    Uses a model loaded from a pickle file to predict the weather on the next
    day.
    
    Arguments:
    
    - input_date [string]:
      the current date, or the date, for which you want to predict, if it will
      rain on the next day.
      - the format should be: YYYY-MM-DD
      - the date must be in the interval [2015-05-12, 2017-06-25]
        (test set of model)
    - input_location [string]:
      Enter the your location, or the location, for which you want to predict,
      if it will rain on the next day.
      - the format should be: the name of the location and if possible,
        the Australian state separated by a comma (or an address).
    
    returns:
    - text [string] giving the weather prediction and data used for the
      prediction: input location / found location and respective coordinates,
                  input date, location and name of next neighboring weather
                  station, and the predicted weather
    - plot [matplotlib figure]: map of Australia with Arrow pointing at input
                                location, and dots for all weather stations,
                                which are color coded to show the values of the
                                prediction
    - data [pandas DataFrame] used for the prediction (w/o scaling applied)
    """
    pd.set_option('display.max_columns', None)
    save_output = "100km_NaNrep_local_std_over"
    filename = save_output + '_rf_model.pkl'

    df = pd.read_pickle("../data/"+save_output+"_df.pkl")
    X_test = pd.read_pickle("../data/"+save_output+"_Xtest.pkl")
    y_test = pd.read_pickle("../data/"+save_output+"_ytest.pkl")
    scaler = pickle.load(open("../data/" + save_output + "_scaler.pkl", 'rb'))
    loaded_model = pickle.load(open("../data/" + filename, 'rb'))
    cols = X_test.columns
    X_test_rescaled = pd.DataFrame(scaler.inverse_transform(X_test),
                                   columns = cols)
    
    ###
    ### check which weather stations have data for this date
    ###
    avail_data = X_test_rescaled.loc[pd.to_datetime(
        X_test_rescaled[["Year", "Month", "Day"]]) == input_date,:]
    avail_locs = X_test_rescaled.loc[pd.to_datetime(
        X_test_rescaled[["Year", "Month", "Day"]]) == input_date,
        ['Location_Latitude', 'Location_Longitude']].drop_duplicates()
    df_temp = pd.read_csv("../data/weatherAUS.csv")
    locs = pd.DataFrame({"Location":df_temp.Location.unique(),
     "NearestBureauStationNr" : [
"072160", "067108", "048027", "059151", "053115",
"061055", "061366", "200288", "067113", "067105", "066214", "066037", "072150",
"061078", "068228", "070351", "070339", "070349", "089002", "081123", "085072",
"086282", "086338", "076031", "078015", "090171", "086068", "090194", "040913",
"031011", "040764", "032040", "023034", "026021", "023373", "016001", "009500",
"109521", "009053", "009021", "009225", "012071", "009998", "094029", "091237",
"015590", "014015", "014932", "015635"],
     "Location_Latitude" : [
36.07, 33.90, 31.48, 30.32, 29.49, 32.92, 33.28, 29.04,
33.72, 33.60, 33.86, 33.95, 35.16, 32.79, 34.37, 35.31, 35.42, 35.53, 37.51,
36.74, 38.12, 37.67, 37.83, 34.24, 36.31, 38.31, 37.74, 37.92, 27.48, 16.87,
27.94, 19.25, 34.95, 37.75, 34.48, 31.16, 35.03, 34.03, 31.67, 31.93, 31.92,
32.99, 34.95, 42.89, 41.42, 23.80, 12.42, 14.52, 25.19],
     "Location_Longitude" : [
146.95, 150.73, 145.83, 153.12, 149.85, 151.8, 151.58,
167.94, 150.68, 150.78, 151.20, 151.17, 147.46, 151.84, 150.93, 149.20, 149.09,
148.77, 143.79, 144.33, 147.13, 144.83, 144.98, 142.09, 141.65, 141.47, 145.10,
141.26, 153.04, 145.75, 153.43, 146.77, 138.52, 140.77, 139.01, 136.81, 117.88,
115.06, 116.02, 115.98, 115.87, 121.62, 116.72, 147.33, 147.12, 133.89, 130.89,
132.38, 130.97],
     "Location_Elevation" : [
164, 81, 260, 4, 213, 33, 19, 112, 25, 19, 43, 6, 212,
8, 10, 577, 587, 1760, 435, 209, 5, 113, 8, 50, 139, 81, 66, 51, 8, 2, 3, 4, 2,
63, 275, 167, 3, 85, 40, 15, 25, 249, 73, 51, 5, 546, 30, 134, 492]})
    avail_locs = avail_locs.merge(right=locs,
                                 how="left",
                                 on=["Location_Latitude","Location_Longitude"])
    del locs, df_temp
    ###
    ### find the closest station
    ###
    geolocator = Nominatim(user_agent = "Australian_locations")
    input_loc = geolocator.geocode(input_location+", Australia")
    # or reverse: input_loc = geolocator.reverse("52.509669, 13.376294")
    # print(input_loc.address, , input_loc.longitude, input_loc.raw)
    """
    print("The weather will be predicted for the day following:",input_date,
          "and the location",input_location,
          "with coordinates [latitude, longitude]:", input_loc.latitude,
          input_loc.longitude, "\n")
    """
    distances = [distance.distance([-avail_locs.Location_Latitude.values[i],
                            avail_locs.Location_Longitude.values[i]],
                      [input_loc.latitude, input_loc.longitude]).km
                 for i in range(len(avail_locs))]
    nearest_station = avail_locs.Location[np.argmin(distances)]
    station_long = avail_locs.Location_Longitude[
        avail_locs.Location == nearest_station].values[0]
    station_lat = avail_locs.Location_Latitude[
        avail_locs.Location == nearest_station].values[0]
    """
    print("The nearest weather station is:",nearest_station,
          "with coordinates [latitude, longitude]:", -station_lat, ",",
          station_long)
    """
    ###
    ### Do predictions for the given date and location
    ###
    # prediction for closest station
    pred_idx = X_test_rescaled.loc[((pd.to_datetime(
        X_test_rescaled[["Year", "Month", "Day"]]) == input_date)
            & (X_test_rescaled.Location_Latitude == station_lat)
            & (X_test_rescaled.Location_Longitude == station_long)),:].index
    y_pred = loaded_model.predict(X_test.loc[pred_idx,:])
    """
    print("The prediction is:",y_pred, "whereas the true value is:",
          y_test.iloc[pred_idx].values,",where 0/1 = no/yes rain tomorrow.")
    """
    # prediction for all stations of that date
    pred_idcs = X_test_rescaled.loc[(pd.to_datetime(
        X_test_rescaled[["Year", "Month", "Day"]]) == input_date),:].index
    y_preds = loaded_model.predict(X_test.loc[pred_idcs,:])
    y_tests = y_test.iloc[pred_idcs].values
    """
    print("predictions and true values for all available weather stations on\
     that day:",y_preds,y_tests)
    """
    pred_date = dt.datetime.strptime(input_date, "%Y-%m-%d")
    pred_date += dt.timedelta(days=1)
    pred_date = pred_date.strftime('%Y-%m-%d')
    if y_pred == 1: pred_weather = "WILL RAIN"
    else: pred_weather = "WILL NOT RAIN"
    if y_test.iloc[pred_idx].values == 1: real_weather = "WILL RAIN"
    else: real_weather = "WILL NOT RAIN"
    pred_output = "\nThe weather should be predicted for this date: "\
    + str(pred_date) + "\nand this location: " + str(input_location)\
    + ", with coordinates [latitude, longitude]: "\
    + str(np.round(input_loc.latitude,1)) + ", "\
    + str(np.round(input_loc.longitude,1))\
    + ".\n\nThe prediction is based on the data of the nearest weather"\
    + " station in " + str(np.round(np.amin(distances),1))\
    + "km distance, which is: " + str(nearest_station)\
    + ", with coordinates [latitude, longitude]: " + str(-station_lat) + ", "\
    + str(station_long) + ".\n\n"\
    + "The prediction is: IT " + pred_weather + ", "\
    + "whereas the true value is: IT "+ real_weather + "."
    ###
    ### plotting
    ###
    # cs = map.contour(x,y,y_preds,15,linewidths=1.5)
    # cs = map.contour(x,y,y_test.iloc[pred_idcs].values,15,linewidths=1.5)
    df_temp = pd.read_csv("../data/weatherAUS.csv")
    df_temp1=pd.DataFrame({"Location":df_temp.Location.unique(),
     "NearestBureauStationNr":[
 "072160", "067108", "048027", "059151", "053115",
"061055", "061366", "200288", "067113", "067105", "066214", "066037", "072150",
"061078", "068228", "070351", "070339", "070349", "089002", "081123", "085072",
"086282", "086338", "076031", "078015", "090171", "086068", "090194", "040913",
"031011", "040764", "032040", "023034", "026021", "023373", "016001", "009500",
"109521", "009053", "009021", "009225", "012071", "009998", "094029", "091237",
"015590", "014015", "014932", "015635"],
     "Location_Latitude":[
36.07, 33.90, 31.48, 30.32, 29.49, 32.92, 33.28, 29.04,
33.72, 33.60, 33.86, 33.95, 35.16, 32.79, 34.37, 35.31, 35.42, 35.53, 37.51,
36.74, 38.12, 37.67, 37.83, 34.24, 36.31, 38.31, 37.74, 37.92, 27.48, 16.87,
27.94, 19.25, 34.95, 37.75, 34.48, 31.16, 35.03, 34.03, 31.67, 31.93, 31.92,
32.99, 34.95, 42.89, 41.42, 23.80, 12.42, 14.52, 25.19],
     "Location_Longitude":[
146.95, 150.73, 145.83, 153.12, 149.85, 151.80, 151.58,
167.94, 150.68, 150.78, 151.20, 151.17, 147.46, 151.84, 150.93, 149.20, 149.09,
148.77, 143.79, 144.33, 147.13, 144.83, 144.98, 142.09, 141.65, 141.47, 145.10,
141.26, 153.04, 145.75, 153.43, 146.77, 138.52, 140.77, 139.01, 136.81, 117.88,
115.06, 116.02, 115.98, 115.87, 121.62, 116.72, 147.33, 147.12, 133.89, 130.89,
132.38, 130.97],
     "Location_Elevation":[
164, 81, 260, 4, 213, 33, 19, 112, 25, 19, 43, 6, 212,
8, 10, 577, 587, 1760, 435, 209, 5, 113, 8, 50, 139, 81, 66, 51, 8, 2, 3, 4, 2,
63, 275, 167, 3, 85, 40, 15, 25, 249, 73, 51, 5, 546, 30, 134, 492]})
    df_temp = pd.merge(df_temp, df_temp1, on='Location', how='left')
    Location_unique_indices=df_temp.reset_index().groupby(['Location'])\
    ['index'].min().to_list()
    names = df_temp.loc[Location_unique_indices,"Location"]
    lats = df_temp.loc[Location_unique_indices,"Location_Latitude"]
    lons = df_temp.loc[Location_unique_indices,"Location_Longitude"]
    elevs = df_temp.loc[Location_unique_indices,"Location_Elevation"]
    lowerleft_corner_lon = np.min(df_temp.Location_Longitude)-2
    lowerleft_corner_lat = -np.max(df_temp.Location_Latitude)-1
    upperright_corner_lon = np.max(df_temp.Location_Longitude)+2
    upperright_corner_lat = -np.min(df_temp.Location_Latitude)+2
    del df_temp, df_temp1

    lons = X_test_rescaled.loc[pred_idcs].Location_Longitude
    lats = X_test_rescaled.loc[pred_idcs].Location_Latitude
    
    #f, axs = plt.subplots() #figsize=fig_size)
    """
    m = Basemap(projection='merc',
                llcrnrlon=lowerleft_corner_lon,
                urcrnrlon=upperright_corner_lon,
                llcrnrlat=lowerleft_corner_lat,
                urcrnrlat=upperright_corner_lat)
    x, y = m(lons,-lats)
    x1,y1 = m(input_loc.longitude,input_loc.latitude)
    m.shadedrelief() #
    def find_colors(data):
        colors=[]
        for i in data:
            if i<0.5:
                colors.append("red")
            else: colors.append("blue")
        return colors
    m.scatter(x,y,marker='o',s=16,color=find_colors(y_tests),alpha=0.5)
     # y_preds
    #m.scatter(x1,y1,marker='o',s=16,color="tab:orange",alpha=0.5)
    plt.annotate(input_location, xy=(x1, y1), xytext=m(input_loc.longitude+3,
                                                       input_loc.latitude-3),
                arrowprops=dict(facecolor='orange',shrink=0.05))
    #m.contourf(x,y,y_preds)
    plt.title('Real Station Data (Blue: Rain, Red: No Rain)\
     and Your Location (Orange)')
    plt.tight_layout()
    true_plot = m
    plt.show()
    plt.clf()
    """
    f, axs = plt.subplots()
    m = Basemap(projection='merc',
                llcrnrlon=lowerleft_corner_lon,
                urcrnrlon=upperright_corner_lon,
                llcrnrlat=lowerleft_corner_lat,
                urcrnrlat=upperright_corner_lat)
    x, y = m(lons,-lats)
    x1,y1 = m(input_loc.longitude,input_loc.latitude)
    m.shadedrelief() #
    def find_colors(data):
        colors=[]
        for i in data:
            if i<0.5:
                colors.append("red")
            else: colors.append("blue")
        return colors
    m.scatter(x,y,marker='o',s=16,color=find_colors(y_preds),alpha=0.5)
    plt.text(upperright_corner_lon+100000, upperright_corner_lat+100000,
           "Blue: Rain\nRed: No Rain\nYour Location: Orange Arrow")
    #m.scatter(x1,y1,marker='o',s=16,color="tab:orange",alpha=0.5)
    plt.annotate(input_location, xy=(x1, y1), xytext=m(input_loc.longitude+3,
                                                       input_loc.latitude-3),
                arrowprops=dict(facecolor='orange',shrink=0.05))
    #m.contourf(x,y,y_preds)
    plt.title('Predicted Weather at Stations on ' + str(pred_date))
    plt.tight_layout()
    #plt.show()
    pred_plot = f
    return pred_output, pred_plot, avail_data


if __name__ == "__main__":
    ###
    ### let the user give a date and location for prediction
    ###
    input_date = str(input("Enter the date, for which you want to predict,\
 if it will rain on the next day.\nThe format should be: YYYY-MM-DD\n\
 The date must be in the interval [2015-05-12, 2017-06-25]\n"))
    input_location = str(input("Enter the location, for which you want to\
 predict, if it will rain on the next day.\n\
 The format should be: the name of the location and the Australian state\
 separated by a comma. An address is also possible."))
    pred_output,pred_plot,avail_data = predict_weather(input_date,
                                                       input_location)
    print("Prediction:")
    print(pred_output)
    print("Plot:")
    plt.show()
    plt.clf()
    print("Available data for prediction:")
    display(avail_data)


    """
    y_train = pd.read_pickle("../data/"+save_output+"_ytrain.pkl")
    X_train = pd.read_pickle("../data/"+save_output+"_Xtrain.pkl")
    # sort X_train and y_train by Date,
    # as the over/undersampling destroyed the sorting
    temp = pd.concat([X_train,y_train],axis=1)
    temp = temp.sort_values(by=["Year", "Month", "Day"])
    X_train = temp.drop("RainTomorrow",axis=1)
    y_train = temp.RainTomorrow
    del temp
    cols = X_train.columns
    X_train_rescaled = pd.DataFrame(scaler.inverse_transform(X_train),
                                    columns = cols)
    print("Amount of entries in preprocessed dataframe\
     (including duplicates from oversampling): ",len(df))
    print("Amount of entries in preprocessed dataframe\
     (excluding duplicates from oversampling): ",
          (len(df)-df.duplicated().sum()))
    print("Amount of train samples: ",
          (len(df)-df.duplicated().sum())*0.8)
    print("Amount of test samples: ",
          (len(df)-df.duplicated().sum())*0.2)
    index1_of_testset = int(len(df)-(len(df)-df.duplicated().sum())*0.2)
    X_test_rescaled.index+=index1_of_testset
    df.drop("RainTomorrow",axis=1).iloc[index1_of_testset:
    ].equals(X_test_rescaled)
    #df.loc[:,"Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    #df = df.loc[:,["Date", *df.columns[:-1]]]
    """