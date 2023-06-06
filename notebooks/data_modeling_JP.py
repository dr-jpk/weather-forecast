#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_preprocessing_JP import *


# In[8]:


def model_from_preproc(description,df, X_train, X_test, y_train, y_test):
    """
    RandomForest w/ / w/o balanced class_weights

    https://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.RandomForestClassifier.html
    """
    rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                                 random_state = 120)
    print()
    print(description,": randomforest w/ default parameter values")
    print()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(np.round(pd.crosstab(y_test, y_pred,
                               rownames=['real/pred'],
                               normalize=True),2).to_markdown())
    print()
    print(np.round(pd.DataFrame(
        sklearn.metrics.classification_report(y_test, y_pred,
                                           output_dict=True)),2).to_markdown())
    print()
    #print(pd.DataFrame(
    #imblearn.metrics.classification_report_imbalanced(y_test,
    #y_pred,
    #output_dict=True)).to_markdown())
    print(np.round(pd.DataFrame(
        rf.feature_importances_,
        index=df.columns[:-1],
        columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
    print()
    
    rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                                 class_weight = "balanced",
                                                 random_state = 120)
    print()
    print(description,": randomforest w/ class_weight=balanced")
    print()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(np.round(pd.crosstab(y_test,
                               y_pred,
                               rownames=['real/pred'],
                               normalize=True),2).to_markdown())
    print()
    print(np.round(pd.DataFrame(
        sklearn.metrics.classification_report(y_test,
                                              y_pred,output_dict=True)),
                   2).to_markdown())
    print()
    #print(pd.DataFrame(imblearn.metrics.classification_report_imbalanced(
    #y_test, y_pred,output_dict=True)).to_markdown())
    print(np.round(pd.DataFrame(
        rf.feature_importances_,
        index=df.columns[:-1],
        columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
    print()
    
    rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                           class_weight = "balanced_subsample",
                                           random_state = 120)
    print()
    print(description,": randomforest w/ class_weight=balanced_subsample")
    print()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(np.round(pd.crosstab(y_test,
                               y_pred,
                               rownames=['real/pred'],
                               normalize=True),2).to_markdown())
    print()
    print(np.round(pd.DataFrame(
        sklearn.metrics.classification_report(y_test,
                                              y_pred,
                                              output_dict=True)),
                   2).to_markdown())
    print()
    #print(pd.DataFrame(imblearn.metrics.classification_report_imbalanced(
    #y_test, y_pred,output_dict=True)).to_markdown())
    print(np.round(pd.DataFrame(
        rf.feature_importances_,
        index=df.columns[:-1],
        columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
    print()
    #y_probas = rf.predict_proba(X_test) # P(X_test sample) in  mod/class
    # different way of evaluation of perf: "cumulative lift curve" / Gain curve
    # score for prob of being in target class (rain)
    #skplt.metrics.plot_cumulative_gain(y_test, y_probas, figsize=(12,8))
    #plt.show() # sorted with decending prob to be in target class
    # curve with x axis= percentage of entries 0 to 1 for test set
    #  y axis "gain" / amount of rainy days

    # counteract imbalance: BalancedRandomForestClassifier
    # https://imbalanced-learn.org/dev/references/generated/
    # imblearn.ensemble.BalancedRandomForestClassifier.html
    import warnings
    warnings.filterwarnings('ignore')
    from imblearn.ensemble import BalancedRandomForestClassifier
    brf = BalancedRandomForestClassifier(random_state=120)
    print()
    print(description,": balancedrandomforest from imblearn")
    print()
    brf.fit(X_train, y_train)
    y_pred = brf.predict(X_test)
    print(np.round(pd.crosstab(y_test,
                               y_pred,
                               rownames=['real/pred'],
                               normalize=True),2).to_markdown())
    print()
    #print(sklearn.metrics.classification_report(y_test, y_pred))
    print(np.round(pd.DataFrame(
        sklearn.metrics.classification_report(y_test,
                                              y_pred,
                                              output_dict=True)),
                   2).to_markdown())#,output_dict=True
    #print(imblearn.metrics.classification_report_imbalanced(y_test, y_pred))
    #print(pd.DataFrame(imblearn.metrics.classification_report_imbalanced(
    #y_test, y_pred,output_dict=True)).transpose().to_markdown())
    print()
    print(np.round(pd.DataFrame(
        brf.feature_importances_,
        index=df.columns[:-1],
        columns=["feature_importance"]).sort_values(
                                                by="feature_importance")[-8:],
                   2).to_markdown())
    print()
    #print(pd.DataFrame(brf.feature_importances_,index=df.columns[:-1],
    #columns=["feature_importance"]).sort_values(
    #by="feature_importance").to_markdown())


# # basic random forest using new (after report2) preprocessing,
  # which includes weighted-average data from nearby stations

# In[10]:


for save_output in["100km_NaNrep_local_std_over"]:
    """["MelbourneAirport_100km_NaNrep_local_std_over",
                   "05corr_NaNrep_local_std_over",
                   "100km_NaNrep_local_std_over",
                   "300km_NaNrep_local_std_over",
                   "03corr_NaNrep_local_std_over"]:"""
    # try to reduce the data to only loc / only loc average features:
    for addition in ["all"]: #,"only_nonavg","only_avg"
        df = pd.read_pickle("../data/"+save_output+"_df.pkl")
        X_train = pd.read_pickle("../data/"+save_output+"_Xtrain.pkl")
        X_train
        X_test = pd.read_pickle("../data/"+save_output+"_Xtest.pkl")
        y_train = pd.read_pickle("../data/"+save_output+"_ytrain.pkl")
        y_test = pd.read_pickle("../data/"+save_output+"_ytest.pkl")
        """
        # sort X_train and y_train by Date,
        # as the over/undersampling destroyed the sorting
        temp = pd.concat([X_train,y_train],axis=1)
        temp = temp.sort_values(by=["Year", "Month", "Day"])
        X_train = temp.drop("RainTomorrow",axis=1)
        y_train = temp.RainTomorrow
        del temp
        """
        if addition == "all": pass
        else:
            locs=["Location_Latitude", "Location_Longitude",
                "Location_Elevation", "MinTemp","MaxTemp", "Evaporation",
                "Sunshine", "WindGustSpeed", "WindGustDir_cos",
                "WindGustDir_sin", "WindDir9am_cos", "WindDir9am_sin",
                "WindDir3pm_cos", "WindDir3pm_sin", "WindSpeed9am",
                "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
                "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
                "Rainfall", "RainToday"]
            if addition == "only_nonavg":
                
                X_train = X_train.loc[:,["Day","Month","Year",*locs]]
                X_test = X_test.loc[:,["Day","Month","Year",*locs]]
                df = df.loc[:,["Day","Month","Year",*locs,"RainTomorrow"]]
            if addition == "only_avg":
                locs=["avg_"+loc for loc in locs] # avg
                X_train = X_train.loc[:,["Day","Month","Year",*locs]]
                X_test = X_test.loc[:,["Day","Month","Year",*locs]]
                df = df.loc[:,["Day","Month","Year",*locs,"RainTomorrow"]]
        # model
        rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                                     random_state = 120)
        print()
        print(save_output,
              addition,
              ": randomforest w/ default parameter values")
        print()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print(np.round(pd.crosstab(y_test, y_pred,
                                   rownames=['real/pred'],
                                   normalize=True),2).to_markdown())
        print()
        print(np.round(pd.DataFrame(
            sklearn.metrics.classification_report(y_test, y_pred,
                                           output_dict=True)),2).to_markdown())
        print()
        print(np.round(pd.DataFrame(
            rf.feature_importances_,
            index=df.columns[:-1],
            columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
        print()
        if ((save_output == "100km_NaNrep_local_std_over") and
            (addition == "all")):
            print("save, reload and check model\n")
            #create pkl file with fitted model
            filename = '100km_NaNrep_local_std_over_rf_model.pkl'
            pickle.dump(rf, open("../data/" + filename, 'wb'))
            # load model from pkl file and use it
            loaded_model = pickle.load(open("../data/" + filename, 'rb'))
            y_pred = loaded_model.predict(X_test)
            print(np.round(pd.crosstab(y_test, y_pred,
                                   rownames=['real/pred'],
                                   normalize=True),2).to_markdown())
            print()
            print(np.round(pd.DataFrame(
                sklearn.metrics.classification_report(y_test, y_pred,
                                           output_dict=True)),2).to_markdown())
            print()
            print(np.round(pd.DataFrame(
                loaded_model.feature_importances_,
                index=df.columns[:-1],
                columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
            print()
"""
OUTPUT for alicesprings data:

alicesprings_03_nanlocal_std_over all :randomforest w/ default parameter values

|   real/pred |    0 |    1 |
|------------:|-----:|-----:|
|           0 | 0.91 | 0.02 |
|           1 | 0.03 | 0.04 |

|           |      0 |     1 |   accuracy |   macro avg |   weighted avg |
|:----------|-------:|------:|-----------:|------------:|---------------:|
| precision |   0.97 |  0.63 |       0.95 |        0.8  |           0.95 |
| recall    |   0.98 |  0.58 |       0.95 |        0.78 |           0.95 |
| f1-score  |   0.97 |  0.6  |       0.95 |        0.79 |           0.95 |
| support   | 480    | 33    |       0.95 |      513    |         513    |

|                 |   feature_importance |
|:----------------|---------------------:|
| avg_Humidity9am |                 0.03 |
| avg_Cloud9am    |                 0.04 |
| Cloud3pm        |                 0.05 |
| avg_Humidity3pm |                 0.06 |
| avg_Sunshine    |                 0.08 |
| Humidity3pm     |                 0.09 |
| Sunshine        |                 0.09 |
| avg_Cloud3pm    |                 0.1  |
"""

'''
# ## testing to improve randomforest and logistic regression with gridsearchcv
  ## (hyperparameter optimization)

# In[86]:


save_output="MelbourneAirport_100km_NaNrep_local_std_over"
# "05corr_NaNrep_local_std_over"
# "100km_NaNrep_local_std_over"
# "300km_NaNrep_local_std_over"
# "03corr_NaNrep_local_std_over"
# "MelbourneAirport_100km_NaNrep_local_std_over"
df = pd.read_pickle("../data/"+save_output+"_df.pkl")
X_train = pd.read_pickle("../data/"+save_output+"_Xtrain.pkl")
X_train
X_test = pd.read_pickle("../data/"+save_output+"_Xtest.pkl")
y_train = pd.read_pickle("../data/"+save_output+"_ytrain.pkl")
y_test = pd.read_pickle("../data/"+save_output+"_ytest.pkl")

# try to reduce the data to only loc / only loc average features:
"""
locs=["Location_Latitude", "Location_Longitude",
            "Location_Elevation", "MinTemp","MaxTemp", "Evaporation",
            "Sunshine", "WindGustSpeed", "WindGustDir_cos", "WindGustDir_sin",
            "WindDir9am_cos", "WindDir9am_sin", "WindDir3pm_cos",
            "WindDir3pm_sin", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
            "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
            "Cloud3pm", "Temp9am", "Temp3pm", "Rainfall", "RainToday"]
avg_locs=["avg_"+loc for loc in locs] # avg_
X_train = X_train.loc[:,["Day","Month","Year",*locs]]
X_test = X_test.loc[:,["Day","Month","Year",*locs]]
df = df.loc[:,["Day","Month","Year",*locs,"RainTomorrow"]]
"""
# model

###
### rf
###
rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                             random_state = 120)
print()
print(": randomforest w/ default parameter values")
print()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(np.round(pd.crosstab(y_test, y_pred,
                           rownames=['real/pred'],
                           normalize=True),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    sklearn.metrics.classification_report(y_test, y_pred,
                                          output_dict=True)),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    rf.feature_importances_,
    index=df.columns[:-1],
    columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
print()

###
### lr
###
lr = sklearn.linear_model.LogisticRegression(random_state=120, n_jobs = -1)
print()
print(": lr w/ default parameter values")
print()
lr.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(np.round(pd.crosstab(y_test, y_pred,
                           rownames=['real/pred'],
                           normalize=True),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    sklearn.metrics.classification_report(y_test, y_pred,
                                          output_dict=True)),2).to_markdown())
print()
feat_imp = abs(lr.coef_[0])
feat_imp =feat_imp / feat_imp.max()
idx = np.argsort(feat_imp)
feats = np.array(X_test.columns)[idx]
print(np.round(pd.DataFrame(
    {"importance" : feat_imp[idx][-8:]}, index = feats[idx][-8:]),
               2).to_markdown())
print()
print()

###
### gridsearch on rf
###
# scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score,pos_label=1)
param_grid_rf = [{'n_estimators': [10, 50, 100, 250, 500, 1000],
                  'criterion' : ["entropy", "gini"],
                  'min_samples_leaf': [1, 3, 5],
                  'max_features': ['sqrt', 'log2']}]
best_rf = sklearn.model_selection.GridSearchCV(rf,
                                               param_grid_rf,
                                               scoring='f1_macro', # or "f1"
                                               cv=3, n_jobs=-1)
best_rf = best_rf.fit(X_train, y_train)
print(best_rf.best_params_)
print(best_rf.best_estimator_)
y_pred = best_rf.predict(X_test)
display(pd.DataFrame.from_dict(best_rf.cv_results_).loc[:,
                                                ['params', 'mean_test_score']])
print(np.round(pd.crosstab(y_test, y_pred,
                           rownames=['real/pred'],
                           normalize=True),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    sklearn.metrics.classification_report(y_test, y_pred,
                                          output_dict=True)),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    best_rf.best_estimator_.feature_importances_,
    index=df.columns[:-1],
    columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
print()

###
### gridsearch on lr
###
param_grid_lr = {'solver': ['liblinear', 'lbfgs'],
                 'C': np.logspace(-4, 2, 7),
                 'max_iter' : [100,1000]}
best_lr = sklearn.model_selection.GridSearchCV(lr,
                                               param_grid_lr,
                                               scoring='f1_macro', # or "f1"
                                               cv=3, n_jobs=-1)
best_lr = best_lr.fit(X_train, y_train)
print(best_lr.best_params_)
print(best_lr.best_estimator_)
y_pred = best_lr.predict(X_test)
display(pd.DataFrame.from_dict(best_lr.cv_results_).loc[:,
                                                ['params', 'mean_test_score']])
print(np.round(pd.crosstab(y_test, y_pred,
                           rownames=['real/pred'],
                           normalize=True),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    sklearn.metrics.classification_report(y_test, y_pred,
                                          output_dict=True)),2).to_markdown())
print()
feat_imp = abs(best_lr.best_estimator_.coef_[0])
feat_imp =feat_imp / feat_imp.max()
idx = np.argsort(feat_imp)
feats = np.array(X_test.columns)[idx]
print(np.round(pd.DataFrame(
    {"importance" : feat_imp[idx][-8:]}, index = feats[idx][-8:]),
               2).to_markdown())


# # basic random forest using only two most important features
# ## (temporarily changed the preprocessing function to make this possible)
# 

# In[ ]:


df=pd.read_csv("../data/weatherAUS.csv")
df, X_train, X_test, y_train, y_test = data_preprocessing(df,
                                        nan_treatment_features = "drop",
                                        sampling = "over")
rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                             random_state = 120)
print()
print(": randomforest w/ default parameter values")
print()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(np.round(pd.crosstab(y_test, y_pred,
                           rownames=['real/pred'],
                           normalize=True),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    sklearn.metrics.classification_report(y_test, y_pred,
                                          output_dict=True)),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    rf.feature_importances_,
    index=df.columns[:-1],
    columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
print()
"""
OUTPUT:
'final data shape (after treating NaNs in features):'

(81650, 3)


: randomforest w/ default parameter values

|   real/pred |    0 |    1 |
|------------:|-----:|-----:|
|           0 | 0.58 | 0.2  |
|           1 | 0.08 | 0.14 |

|           |       0 |       1 |   accuracy |   macro avg |   weighted avg |
|:----------|--------:|--------:|-----------:|------------:|---------------:|
| precision |    0.88 |    0.41 |       0.72 |        0.65 |           0.78 |
| recall    |    0.74 |    0.64 |       0.72 |        0.69 |           0.72 |
| f1-score  |    0.81 |    0.5  |       0.72 |        0.65 |           0.74 |
| support   | 8810    | 2474    |       0.72 |    11284    |       11284    |

|             |   feature_importance |
|:------------|---------------------:|
| Humidity3pm |                 0.46 |
| Sunshine    |                 0.54 |
"""


# # basic random forest using all features
# ## (same preprocessing function as above, but w/o temporal changes)

# In[ ]:


df=pd.read_csv("../data/weatherAUS.csv")
df, X_train, X_test, y_train, y_test = data_preprocessing(df,
                                        nan_treatment_features = "drop",
                                        sampling = "over")
rf = sklearn.ensemble.RandomForestClassifier(n_jobs = -1,
                                             random_state = 120)
print()
print(": randomforest w/ default parameter values")
print()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(np.round(pd.crosstab(y_test, y_pred,
                           rownames=['real/pred'],
                           normalize=True),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    sklearn.metrics.classification_report(y_test, y_pred,
                                          output_dict=True)),2).to_markdown())
print()
print(np.round(pd.DataFrame(
    rf.feature_importances_,
    index=df.columns[:-1],
    columns=["feature_importance"]).sort_values(
                                by="feature_importance")[-8:],2).to_markdown())
print()
"""
OUTPUT:

'final data shape (after treating NaNs in features):'

(81650, 30)


: randomforest w/ default parameter values

|   real/pred |    0 |    1 |
|------------:|-----:|-----:|
|           0 | 0.74 | 0.05 |
|           1 | 0.1  | 0.12 |

|           |       0 |       1 |   accuracy |   macro avg |   weighted avg |
|:----------|--------:|--------:|-----------:|------------:|---------------:|
| precision |    0.88 |    0.73 |       0.86 |        0.8  |           0.85 |
| recall    |    0.94 |    0.55 |       0.86 |        0.74 |           0.86 |
| f1-score  |    0.91 |    0.62 |       0.86 |        0.77 |           0.85 |
| support   | 8810    | 2474    |       0.86 |    11284    |       11284    |

|               |   feature_importance |
|:--------------|---------------------:|
| Rainfall      |                 0.04 |
| Humidity9am   |                 0.04 |
| Pressure9am   |                 0.05 |
| WindGustSpeed |                 0.05 |
| Pressure3pm   |                 0.05 |
| Cloud3pm      |                 0.06 |
| Humidity3pm   |                 0.13 |
| Sunshine      |                 0.14 |
"""


# # modeling for report2

# In[ ]:


df=pd.read_csv("../data/weatherAUS.csv")

model_from_preproc("default",*data_preprocessing(df))
 # (stdscaler, no sampling, mean/mode over whole col with NaN)
model_from_preproc("NaN_drop",
                   *data_preprocessing(df, nan_treatment_features = "drop"))
model_from_preproc("oversampling",
                   *data_preprocessing(df, sampling = "over"))
model_from_preproc("undersampling",
                   *data_preprocessing(df, sampling = "under"))
model_from_preproc("NaN_drop oversampling",
                   *data_preprocessing(df, nan_treatment_features = "drop",
                                       sampling = "over"))
model_from_preproc("NaN_drop undersampling",
                   
                   *data_preprocessing(df, nan_treatment_features = "drop",
                                       sampling = "under"))


# # model selection with nested cross-validation

# In[ ]:


"""ToDo: add code here! Or find better way to optimize model performance"""


# # save model for future application
# 

# In[ ]:


# fit model
#model = RandomForestClassifier()
#model.fit(X_train, y_train)
#create pkl file with fitted model
filename = 'model.pkl'
pickle.dump(rf, open(filename, 'wb'))
# load model from pkl file and use it
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict(X_test)
loaded_model.score(X_test, y_test)


# # tests

# In[ ]:


df=pd.read_csv("../data/weatherAUS.csv")
# de = default parameter values of data_preprocessing()
#      (stdscaler, no sampling, mean/mode over whole col with NaN)
# ov = over-sampling
# un = under-sampling
# dr = drop all rows with NaN
df_de, X_train_de, X_test_de, y_train_de, y_test_de = data_preprocessing(df)
df_dr, X_train_dr, X_test_dr, y_train_dr, y_test_dr = \
    data_preprocessing(df, nan_treatment_features = "drop")
df_ov, X_train_ov, X_test_ov, y_train_ov, y_test_ov = \
    data_preprocessing(df, sampling = "over")
df_un, X_train_un, X_test_un, y_train_un, y_test_un = \
    data_preprocessing(df, sampling = "under")
df_drov, X_train_drov, X_test_drov, y_train_drov, y_test_drov = \
    data_preprocessing(df, nan_treatment_features = "drop", sampling = "over")
df_drun, X_train_drun, X_test_drun, y_train_drun, y_test_drun = \
    data_preprocessing(df, nan_treatment_features = "drop", sampling = "under")




# check if randomover/undersampling does not destroy the temporal
# order of train and test set

print(pd.Series(y_train_de).value_counts())
print(pd.Series(y_train_ov).value_counts())
print(pd.Series(y_train_un).value_counts())
print(pd.Series(y_train_drov).value_counts())

# check if sampler always uses same rows and
# does not e.g. mix up Location longs&lats
display(df_de[df_de.Location_Latitude==35.31]) 
display(df_ov[df_ov.Location_Latitude==35.31])
display(df_un[df_un.Location_Latitude==35.31])

# check if sampler does not generate new dates extending to temporal range
# of the test set

dates_de = pd.to_datetime(dict(year=X_train_de.Year,
                               month=X_train_de.Month,
                               day=X_train_de.Day))
dates_un = pd.to_datetime(dict(year=X_train_un.Year,
                               month=X_train_un.Month,
                               day=X_train_un.Day))
dates_ov = pd.to_datetime(dict(year=X_train_ov.Year,
                               month=X_train_ov.Month,
                               day=X_train_ov.Day))
print(len(dates_de),len(dates_ov),len(dates_un))
# check if sampler keeps same feature value ranges
print(X_train_de.min(),X_train_de.max()) 
print(X_train_ov.min(),X_train_ov.max())
print(X_train_un.min(),X_train_un.max())


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df=pd.read_csv("../data/weatherAUS.csv")
print(
     np.round(df.groupby("Location").RainTomorrow.value_counts(normalize=True),
         2).sort_values())
print("modeling only the Location 'AliceSprings' yields high accuracy,\
 just because it is one of the driest locations in Australia (only 8% raining\
 days.)")

'''