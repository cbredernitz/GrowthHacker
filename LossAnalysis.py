from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt
from data_preprocess import load_df, unique_valued_cols, drop_cols, preprocess, label_encoding
import numpy as np
import pandas as pd

## Loading Model
mdl = joblib.load('modl_GBT_full.joblib')

### Loading TEST Data
df_test = load_df('test_v2.csv')

### Fix outliers that do not fall within +/-3 std dev from mean of log transaction value

# create a new dummy column logTransaction which is the log of all totalTransactionRevenue
df_test['logTransaction']= df_test['totals.totalTransactionRevenue'].fillna(0).astype(float).apply(lambda x: np.log1p(x))
std_dev = df_test.logTransaction.std()
mean_val = df_test.logTransaction.mean()
df_test['logTransaction'] = np.where(np.abs(df_test.logTransaction-mean_val) > 3*std_dev,3*std_dev,df_test.logTransaction)

### Extract Labels 

# Get true values from test set
y_true = df_test['logTransaction']

### Removing unnecessary columns 

# colums that contain no data
ones = unique_valued_cols(df_test)
cols_to_remove = [x for x in ones if set(df_test[x].unique()) == set(['not available in demo dataset'])]
cols_to_remove.extend(['hits', 'customDimensions', 'device.isMobile'])

# Drop them
df_test = drop_cols(df_test, list(cols_to_remove))

# Remove transaction related columns
transaction_cols = ['totals.totalTransactionRevenue', 'totals.transactionRevenue', 'totals.transactions', 'fullVisitorId', 'logTransaction']
df_test = drop_cols(df_test, transaction_cols)

### Preprocess the data before we start training
# print(df_test.iloc[0])
df_test = df_test.fillna(0)
df_test = label_encoding(df_test)
df_test = preprocess(df_test)

### Get Predictions
pred = mdl.predict(df_test)

vals = []
acc = []

for idx, each in enumerate(list(pred)):
    vals.append((idx, (each-y_true[idx])))

for idx, each in enumerate(list(pred)):
    if float(y_true[idx]) != 0.0:
        acc.append((idx, abs(each-y_true[idx])))


df_top = pd.DataFrame()
df_bot = pd.DataFrame()
df_acc = pd.DataFrame()

vals_sort_top = sorted(vals, key=lambda tup: tup[1], reverse=True)
vals_sort_bot = sorted(vals, key=lambda tup: tup[1])
vals_sort_acc = sorted(acc, key=lambda tup: tup[1])

print('--- TOP ---')
for each in vals_sort_top[:50]:
    print(each[1])
    print(y_true[0])
    print('-'*20)
    df_top = df_top.append(df_test.iloc[each[0]])
    # print(df_test.iloc[each[0]])

print('--- BOTTOM ---')
for each in vals_sort_bot[:50]:
    print(each[1])
    print(y_true[0])
    print('-'*20)
    df_bot = df_bot.append(df_test.iloc[each[0]])

print('--- Accurate ---')
for each in vals_sort_acc[:50]:
    print(each[1])
    print(y_true[each[0]])
    print('-'*20)
    df_acc = df_acc.append(df_test.iloc[each[0]])


df_top.to_csv('Top_error.csv')
df_bot.to_csv('Bottom_error.csv')
df_acc.to_csv('Accurate_pred.csv')
