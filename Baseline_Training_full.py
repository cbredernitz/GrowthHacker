import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import ast
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.metrics import mean_squared_error


def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    print(df.shape)
    return df

def flatten_hits(df):
    df_ = pd.DataFrame()
    for index, row in df.iterrows():
        initial_id = df['fullVisitorId'][index]
        s = json.dumps(df['hits'][index])
        r = json.loads(s)
        d = ast.literal_eval(r)
        for each in d:
            each['fullVisitorId'] = initial_id
        column_as_df = json_normalize(d)
        if 'product' in column_as_df.columns:
            column_as_df['v2ProductName'] = column_as_df['product'].apply(lambda x: [p['v2ProductName'] for p in x] if type(x) == list else [])
            column_as_df['v2ProductCategory'] = column_as_df['product'].apply(lambda x: [p['v2ProductCategory'] for p in x] if type(x) == list else [])
            del column_as_df['product']
        if 'promotion' in column_as_df.columns:
            column_as_df['promoId']  = column_as_df['promotion'].apply(lambda x: [p['promoId'] for p in x] if type(x) == list else [])
            column_as_df['promoName']  = column_as_df['promotion'].apply(lambda x: [p['promoName'] for p in x] if type(x) == list else [])
            del column_as_df['promotion']
        df_ = df_.append(column_as_df)
    df = df.merge(df_, on='fullVisitorId')
    print(df.shape)
    return df

### Loading TRAIN Data
df_train = load_df()
df_train = flatten_hits(df_small)

### Loading TEST Data
df_test = load_df('D:\\Documents\\ga-customer-revenue-prediction\\test_v2.csv')
df_test = flatten_hits(df_test)

cols_to_remove = [x for x in ones if set(df_small[x].unique()) == set(['not available in demo dataset'])]

y = df_small['totals.totalTransactionRevenue'].fillna(0).astype(float)
y = y.apply(lambda x: np.log1p(x))
df_train = df_small.drop('totals.totalTransactionRevenue', axis=1)

### Removing columns that contain no data
df_train = df_train.drop(list(cols_to_remove), axis=1)
df_test = df_test.drop(list(cols_to_remove), axis=1)

cat_columns = ['channelGrouping',
               'socialEngagementType',
               'device.browser',
               'device.deviceCategory',
               'device.operatingSystem',
               'geoNetwork.city',
               'geoNetwork.continent',
               'geoNetwork.country',
               'geoNetwork.metro', 
               'geoNetwork.networkDomain',
               'geoNetwork.region', 
               'geoNetwork.subContinent',
               'trafficSource.adContent',
               'trafficSource.adwordsClickInfo.adNetworkType',
               'trafficSource.adwordsClickInfo.gclId',
               'trafficSource.adwordsClickInfo.page',
               'trafficSource.adwordsClickInfo.slot', 
               'trafficSource.campaign',
               'trafficSource.keyword',
               'trafficSource.referralPath',
               'trafficSource.source',
               'trafficSource.medium'
              ]

from sklearn import preprocessing
for each in cat_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[each].values) + list(df_test[each].values))
    df_train[each] = lbl.transform(list(df_train[each].values))
    df_test[each] = lbl.transform(list(df_test[each].values))

# Getting our vailidation y for scoring
y_true = df_test['totals.totalTransactionRevenue'].fillna(0).astype(float)
y_true = y_true.apply(lambda x: np.log1p(x))
df_test = df_test.drop('totals.totalTransactionRevenue', axis=1)

y_mean = np.mean(y)
y_base = np.full_like(y_true, y_mean)

df_train = df_train.drop('customDimensions', axis=1)
df_test = df_test.drop('customDimensions', axis=1)
df_train = df_train.drop('totals.transactionRevenue', axis=1)
df_test = df_test.drop('totals.transactionRevenue', axis=1)
df_train = df_train.drop('totals.transactions', axis=1)
df_test = df_test.drop('totals.transactions', axis=1)

def preprocess(df):
    df['totals.bounces'] = df['totals.bounces'].fillna(0).astype(np.float)
    df['totals.newVisits'] = df['totals.newVisits'].fillna(0).astype(np.float)
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].fillna(0).astype(np.float)
    df['totals.transactions'] = df['totals.transactions'].fillna(0).astype(np.float)
    df['trafficSource.adwordsClickInfo.isVideoAd'] = df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(0).astype(np.float)
    df['trafficSource.isTrueDirect'] = df['trafficSource.isTrueDirect'].fillna(0).astype(np.float)
    
    return df

df_train = preprocess(df_train)
df_test = preprocess(df_test)

#### IMPORTANT ####
# Find the issue here before proceeding on Flux

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

from sklearn import tree
clf_tree = tree.DecisionTreeRegressor()
clf_tree = clf_tree.fit(df_train, y)

y_pred = clf_tree.predict(df_test)

RMSE = sqrt(mean_squared_error(y_true, y_pred))

for idx, each in enumerate(clf_tree.feature_importances_):
    print(idx, each*1e5)

print('-'*10)

for idx, each in enumerate(df_train.columns):
    print(idx, each)

