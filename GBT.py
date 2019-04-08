import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.externals import joblib
from data_preprocess import load_df, label_encoding, drop_cols, preprocess, train_and_predict
from data_preprocess import unique_valued_cols


#################### MAIN ###########################

def main():
    ### Loading TRAIN Data
    df_train = load_df("train_v2.csv")

    ### Loading TEST Data
    df_test = load_df('test_v2.csv')

    ### Fix outliers that do not fall within +/-3 std dev from mean of log transaction value
    
    # create a new dummy column logTransaction which is the log of all totalTransactionRevenue
    df_train['logTransaction']= df_train['totals.totalTransactionRevenue'].fillna(0).astype(float).apply(lambda x: np.log1p(x))
    std_dev = df_train.logTransaction.std()
    mean_val = df_train.logTransaction.mean()
    df_train['logTransaction'] = np.where(np.abs(df_train.logTransaction-mean_val) > 3*std_dev,3*std_dev,df_train.logTransaction)

    # create a new dummy column logTransaction which is the log of all totalTransactionRevenue
    df_test['logTransaction']= df_test['totals.totalTransactionRevenue'].fillna(0).astype(float).apply(lambda x: np.log1p(x))
    std_dev = df_test.logTransaction.std()
    mean_val = df_test.logTransaction.mean()
    df_test['logTransaction'] = np.where(np.abs(df_test.logTransaction-mean_val) > 3*std_dev,3*std_dev,df_test.logTransaction)

    ### Extract Labels 
    y = df_train['logTransaction']
    # Get true values from test set
    y_true = df_test['logTransaction']

    ### Removing unnecessary columns 

    # colums that contain no data
    ones = unique_valued_cols(df_train)
    cols_to_remove = [x for x in ones if set(df_train[x].unique()) == set(['not available in demo dataset'])]
    cols_to_remove.append(['hits', 'customDimensions'])

    # Drop them
    df_train = drop_cols(df_train, list(cols_to_remove))
    df_test = drop_cols(df_test, list(cols_to_remove))

    # Remove transaction related columns
    transaction_cols = ['totals.totalTransactionRevenue', 'totals.transactionRevenue', 'totals.transactions', 'fullVisitorId', 'logTransaction']
    df_train = drop_cols(df_train, transaction_cols)
    df_test = drop_cols(df_test, transaction_cols)

    # Remove extra column in training
    df_train = df_train.drop('trafficSource.campaignCode', axis=1)

    ### Preprocess the data before we start training
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    ### Create categorical and numeric features dataframe
    df_categorical = df_train.select_dtypes(include=['object'])
    df_categorical_test = df_test.select_dtypes(include=['object'])

    # Numeric
    df_numeric = df_train.select_dtypes(include=['float64', 'int64'])
    df_numeric_test = df_test.select_dtypes(include=['float64', 'int64'])

    # Label encoding
    df_categorical = label_encoding(df_categorical)
    df_categorical_test = label_encoding(df_categorical_test)

    '''
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

    df_train = df_train.drop(low_features)
    df_test = df_test.drop(low_features)'''

    ### Training and Predictions

    ################### Categorical ###############
    gbrt = ensemble.GradientBoostingRegressor()
    model_cat, RMSE_test, RMSE_train = train_and_predict(gbrt, df_categorical, df_categorical_test, y, y_true)

    for idx, each in enumerate(df_categorical.columns):
        print(idx, each)

    for idx, each in enumerate(model_cat.feature_importances_):
        print(idx, each*1e5)

    print('-'*10)
    print('\n')
    print("-- Scores for Categorical --")
    print("RMSE on test: ", RMSE_test)
    print("RMSE on train: ", RMSE_train)
    print('\n\n')

    print('-- Getting list of columns for categorical model --\n')
    for each in df_categorical.columns:
        print(each)
    for imp in model_cat.feature_importances_:
        print(imp)

    # Save Categorical model
    joblib.dump(model_cat, "modl_GBT_cat.joblib")

    ###################### Numerical #####################
    df_numeric = df_numeric.fillna(0)
    df_numeric_test = df_numeric_test.fillna(0)

    model_num, RMSE_test, RMSE_train = train_and_predict(gbrt, df_numeric, df_numeric_test, y, y_true)

    for idx, each in enumerate(df_numeric.columns):
        print(idx, each)

    for idx, each in enumerate(model_num.feature_importances_):
        print(idx, each*1e5)

    print('-'*10)
    print('\n')
    print("-- Scores for Numerical --")
    print("RMSE on test: ", RMSE_test)
    print("RMSE on train: ", RMSE_train)
    print('\n\n')

    print('-- Getting list of columns for Numerical Model --\n')
    for each in df_numeric.columns:
        print(each)
    for imp in model_num.feature_importances_:
        print(imp)

    # Save Numerical model
    joblib.dump(model_num, "modl_GBT_num.joblib")

    ###################### Full #####################
    df_train = pd.concat([df_numeric,df_categorical],axis=1)
    df_test = pd.concat([df_numeric_test,df_categorical_test],axis=1)

    model_full, RMSE_test, RMSE_train = train_and_predict(gbrt, df_train, df_test, y, y_true)

    for idx, each in enumerate(df_train.columns):
        print(idx, each)

    for idx, each in enumerate(model_full.feature_importances_):
        print(idx, each*1e5)

    print('-'*10)
    print('\n')
    print("-- Scores for Full --")
    print("RMSE on test: ", RMSE_test)
    print("RMSE on train: ", RMSE_train)
    print('\n\n')

    print('-- Getting list of columns for Full model --\n')
    for each in df_train.columns:
        print(each)
    for imp in model_full.feature_importances_:
        print(imp)

    # Save full model
    joblib.dump(model_full, "modl_GBT_full.joblib")


if __name__ == "__main__":
    main()
