import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from data_preprocess import load_df, label_encoding, drop_cols, preprocess, train_and_predict
from data_preprocess import unique_valued_cols

#################### MAIN ###########################

def main():
    ### Loading TRAIN Data
    df_train = load_df("train_v2.csv",10000)

    ### Loading TEST Data
    df_test = load_df('test_v2.csv', 10000)

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
    cols_to_remove.extend(['hits','customDimensions'])


    # Drop them from both the sets
    df_train = drop_cols(df_train, list(cols_to_remove))
    df_test = drop_cols(df_test, list(cols_to_remove))

    # Remove transaction related columns
    transaction_cols = ['totals.totalTransactionRevenue', 'totals.transactionRevenue', 'totals.transactions', 'fullVisitorId', 'logTransaction']
    df_train = drop_cols(df_train, transaction_cols)
    df_test = drop_cols(df_test, transaction_cols)

    # Remove extra column in training
    #df_train = df_train.drop('trafficSource.campaignCode', axis=1)

    ### Preprocess the data before we start training
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    ### Create categorical and numeric features dataframe
    df_categorical = df_train.select_dtypes(include=['object'])
    df_categorical_test = df_test.select_dtypes(include=['object'])

    # Numeric
    df_numeric = df_train.select_dtypes(include=['float64', 'int64'])
    df_numeric_test = df_test.select_dtypes(include=['float64', 'int64'])

    # Label encoding on categorical
    df_categorical = label_encoding(df_categorical)
    df_categorical_test = label_encoding(df_categorical_test)

    # Predict on categorical
    lm = LinearRegression()
    categorize_model, RMSE_test, RMSE_train = train_and_predict(lm, df_categorical, df_categorical_test, y, y_true)

    print("In-sample RMSE categorical:", RMSE_train)
    print("Out-sample RMSE categorical:", RMSE_test)
    print("Model parameters: ", categorize_model.get_params())

    joblib.dump(categorize_model, "modl_LR_cat.joblib")

    print('-'*10)


    # Predict on numerical
    df_numeric = df_numeric.fillna(0)
    df_numeric_test = df_numeric_test.fillna(0)

    num_model, RMSE_test, RMSE_train = train_and_predict(lm, df_numeric, df_numeric_test, y, y_true)

    print("In-sample RMSE numeric:", RMSE_train)
    print("Out-sample RMSE numeric:", RMSE_test)
    print("Model parameters: ", num_model.get_params())

    joblib.dump(num_model, "modl_LR_cat.joblib")

    print('-'*10)

    # Predict on all features
    df_train = pd.concat([df_numeric,df_categorical],axis=1)
    df_test = pd.concat([df_numeric_test,df_categorical_test],axis=1)
    full_model, RMSE_test, RMSE_train = train_and_predict(lm, df_train, df_test, y, y_true)

    print("In-sample RMSE:", RMSE_train)
    print("Out-sample RMSE:", RMSE_test)
    print("Model parameters: ", full_model.get_params())

    joblib.dump(full_model, "modl_LR_cat.joblib")

    print('-'*10)


if __name__ == "__main__":
    main()



