import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import ast
from ChiSquare import ChiSquare
import scipy.stats as stats
from scipy.stats import chi2_contingency
from data_preprocess import load_df, unique_valued_cols

# Load all data.
### Loading TRAIN Data
df_train = load_df("train_v2.csv")


# create a new dummy column logTransaction which is the log of all totalTransactionRevenue
df_train['logTransaction']= df_train['totals.totalTransactionRevenue'].fillna(0).astype(float).apply(lambda x: np.log1p(x))
std_dev = df_train.logTransaction.std()
mean_val = df_train.logTransaction.mean()
df_train['logTransaction'] = np.where(np.abs(df_train.logTransaction-mean_val) > 3*std_dev,3*std_dev,df_train.logTransaction)

# Get the categorical variables
df_categorical = df_train.select_dtypes(include=['object'])

# Remove colums that contain no data
ones = unique_valued_cols(df_train)
cols_to_remove = [x for x in ones if set(df_train[x].unique()) == set(['not available in demo dataset'])]
df_categorical = df_categorical.drop(cols_to_remove, axis=1)

# add logTransaction (dependent variable) column
df_categorical['logTransaction'] = df_train['logTransaction']

# delete train set as we don't need it anymore
del df_train

#Initialize ChiSquare Class
cT = ChiSquare(df_categorical)

# Test independence of categorical variables on logTransaction - that we are predicting.
# print which variables are important and which ones are not important
for var in df_categorical.columns:
    if var is not "logTransaction":
        cT.TestIndependence(colX=var,colY="logTransaction" ) 


