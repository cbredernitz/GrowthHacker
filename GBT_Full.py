import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import ast
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = ["{0}.{1}".format(column, subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

### Loading TRAIN Data
df_train = load_df("train_v2.csv")

### Loading TEST Data
df_test = load_df('test_v2.csv')

### unique valued columns
ones = []
for each in df_train.columns:
    try:
        if df_train[each].nunique() == 1:
            ones.append(each)
    except:
        print(each)
cols_to_remove = [x for x in ones if set(df_train[x].unique()) == set(['not available in demo dataset'])]
cols_to_remove.append('hits')
cols_to_remove.append('customDimensions')

y = df_train['totals.totalTransactionRevenue'].fillna(0).astype(float)
y = y.apply(lambda x: np.log1p(x))
df_train = df_train.drop('totals.totalTransactionRevenue', axis=1)

### Removing columns that contain no data
cols_to_remove_train = list(cols_to_remove)
df_train = df_train.drop(list(cols_to_remove_train), axis=1)
df_test = df_test.drop(list(cols_to_remove), axis=1)

# Getting our vailidation y for scoring
y_true = df_test['totals.totalTransactionRevenue'].fillna(0).astype(float)
y_true = y_true.apply(lambda x: np.log1p(x))
df_test = df_test.drop('totals.totalTransactionRevenue', axis=1)

df_train = df_train.drop('totals.transactionRevenue', axis=1)
df_test = df_test.drop('totals.transactionRevenue', axis=1)
df_train = df_train.drop('totals.transactions', axis=1)
df_test = df_test.drop('totals.transactions', axis=1)
df_train = df_train.drop('trafficSource.campaignCode', axis=1)
df_train = df_train.drop('fullVisitorId', axis=1)
df_test = df_test.drop('fullVisitorId', axis=1)

df_categorical = df_train.select_dtypes(include=['object'])
cat_columns = [x for x in df_categorical.columns]

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

df_train = preprocess(df_train)
df_test = preprocess(df_test)

df_train = preprocess(df_train)
df_test = preprocess(df_test)

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

from sklearn import preprocessing
for each in cat_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[each].values) + list(df_test[each].values))
    df_train[each] = lbl.transform(list(df_train[each].values))
    df_test[each] = lbl.transform(list(df_test[each].values))

from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

##### Can comment out the code between when running all columns #####
low_features = [
    'socialEngagementType',
    'totals.bounces',
    'trafficSource.adwordsClickInfo.adNetworkType',
    'trafficSource.adwordsClickInfo.isVideoAd',
    'trafficSource.adwordsClickInfo.page',
    'trafficSource.adwordsClickInfo.slot',
    'trafficSource.campaign',
    'trafficSource.source'
]

#df_train = df_train.drop(low_features)
#df_test = df_test.drop(low_features)

clf_tree = ensemble.GradientBoostingRegressor()

clf_tree = clf_tree.fit(df_train, y)

y_pred = clf_tree.predict(df_test)
y_pred_train = clf_tree.predict(df_train)

RMSE_test = sqrt(mean_squared_error(y_true, y_pred))
RMSE_train = sqrt(mean_squared_error(y, y_pred_train))

for idx, each in enumerate(clf_tree.feature_importances_):
    print(idx, each*1e5)

print('-'*10)

for idx, each in enumerate(df_train.columns):
    print(idx, each)
print('\n')
print("-- Scores --")
print("RMSE on test: ", RMSE_test)
print("RMSE on train: ", RMSE_train)
print('\n\n')
#import matplotlib.pyplot as plt
#fig = plt.gcf()
#fig.set_size_inches(18.5, 10.5)

print('-- Getting list of columns --\n')
for each in df_train.columns:
    print(each)
for imp in clf_tree.feature_importances_:
    print(imp)
#indices = np.argsort(importances)

#plt.title('Feature Importances')
#plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#plt.yticks(range(len(indices)), [features[i] for i in indices])
#plt.xlabel('Relative Importance')
# plt.show()
#plt.savefig("Feature_Importance_initial.png")

from sklearn.externals import joblib
joblib.dump(clf_tree, "modl_default_Less_Params.joblib")
