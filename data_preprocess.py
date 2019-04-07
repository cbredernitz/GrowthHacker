import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import ast
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing
from sklearn import ensemble


############# Helper functions ####################

# Load data into Pandas df and return the df
def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = ["{0}.{1}".format(column,subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Load complete.")
    print(df.shape)
    return df

# perform lable encoding (for categorical columns only)
def label_encoding(df):
    for each in df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[each].values))
        df[each] = lbl.transform(list(df[each].values))
    return df

# Return columns with just one value
def unique_valued_cols(df):
    ### unique valued columns
    ones = []
    for each in df.columns:
        try:
            if df[each].nunique() == 1:
                ones.append(each)
        except:
            pass
    return ones

# function to preprocess data
def preprocess(df):
    # Boolean
    df['trafficSource.adwordsClickInfo.isVideoAd'] = df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(1).astype(np.float)
    df['trafficSource.isTrueDirect'] = df['trafficSource.isTrueDirect'].fillna(0).astype(np.float)

    # Text numbers to int
    df['totals.hits'] = df['totals.hits'].fillna(0).astype(np.float)
    df['totals.pageviews'] = df['totals.pageviews'].fillna(0).astype(np.float)
    df['totals.sessionQualityDim'] = df['totals.sessionQualityDim'].fillna(0).astype(np.float)
    df['totals.timeOnSite'] = df['totals.timeOnSite'].fillna(0).astype(np.float)
    df['totals.visits'] = df['totals.visits'].fillna(0).astype(np.float)
    df['totals.newVisits'] = df['totals.newVisits'].fillna(0).astype(np.float)
    df['totals.timeOnSite'] = df['totals.timeOnSite'].fillna(0).astype(np.float)
    df['totals.pageviews'] = df['totals.pageviews'].fillna(0).astype(np.float)
    df['trafficSource.adwordsClickInfo.page'] = df['trafficSource.adwordsClickInfo.page'].fillna(0).astype(np.float)
    df['visitNumber'] = df['visitNumber'].fillna(0).astype(np.float)
    df['visitStartTime'] = df['visitStartTime'].fillna(0).astype(np.float)

    #EXTRACTING DAY_OF_WEEK, HOUR, DAY, MONTH FROM DATE 
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df = df.drop('date', axis=1)
    return df

# remove columns
def drop_cols(df,cols):
    df = df.drop(cols,axis=1)
    return df

# Train model and predict
def train_and_predict(model, X_train, X_test, y_train, y_test):
    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    RMSE_test = sqrt(mean_squared_error(y_test, y_pred))
    RMSE_train = sqrt(mean_squared_error(y_train, y_pred_train))

    return model, RMSE_test, RMSE_train