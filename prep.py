#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import feather

# Load Training Data
train_raw = pd.read_csv('data/train.csv', dtype={'id': np.int32})
train = train_raw
train['log_loss'] = np.log(train['loss'])
ntrain = train.shape[0]

# Load Test Data
test_raw = pd.read_csv('data/test.csv', dtype={'id': np.int32})
test = test_raw
ntest = test.shape[0]

features = train.columns

# factorize categorical features
for feat in [feat for feat in features if 'cat' in feat]:
    a = train['log_loss'].groupby([train[feat]]).mean()
    a[feat] = a.index
    train[feat] = pd.merge(left=train, right=a, how='left', on=feat)['log_loss']
    test[feat] = pd.merge(left=test, right=a, how='left', on=feat)['log_loss']

features_numeric = train.dtypes[train.dtypes != "object"].index

# compute skew and do Box-Cox transformation
# transform features with skew > 0.25 (this can be varied to find optimal value)
features_skewed = train[features_numeric].apply(lambda x: skew(x.dropna()))
features_skewed = features_skewed[features_skewed > 0.25]
for feat in features_skewed.index:
    train[feat] = train[feat] + 1
    train[feat], lam = boxcox(train[feat])
    if feat in test.columns:
        test[feat] = test[feat] + 1
        test[feat], lam = boxcox(test[feat])

# Scale the Data
scaler = StandardScaler().fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# Split data back into the training and test sets
train_labels = np.array(train['log_loss'])
train_ids = train_raw['id'].values.astype(np.int32)
test_ids = test_raw['id'].values.astype(np.int32)

# Save the dataframes in files to be picked up by other scripts
feather.write_dataframe(train, "temp/preparedtraining.feather")
feather.write_dataframe(test, "temp/preparedtest.feather")
