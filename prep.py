#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
import feather

# Load Training Data
train = pd.read_csv('data/train.csv', dtype={'id': np.int32})
train['log_loss'] = np.log(train['loss'])

# Load Test Data
test = pd.read_csv('data/test.csv', dtype={'id': np.int32})

# Replace Categorical Features with the Category's mean log loss
features_categorical = [feat for feat in test.columns if 'cat' in feat]

for feat in features_categorical:
    a = pd.DataFrame(train['loss'].groupby([train[feat]]).mean())
    a[feat] = a.index
    train[feat] = pd.merge(left=train, right=a, how='left', on=feat)['loss_y']
    test[feat] = pd.merge(left=test, right=a, how='left', on=feat)['loss']

features_numeric = test.dtypes[test.dtypes != "object"].index

# compute skew and do Box-Cox transformation
# transform features with skew > 0.75 (this can be varied to find optimal value)
# features_skewed = train[features_numeric].apply(lambda x: skew(x.dropna()))
# features_skewed = features_skewed[features_skewed > 0.75]
# for feat in features_skewed.index:
#     train[feat], lam = boxcox(train[feat] + 1)
#     test[feat] = boxcox(test[feat] + 1, lam)

## Scale the Data
#scaler = StandardScaler().fit(train)
#train_scaled = scaler.transform(train)
#test_scaled = scaler.transform(test)

# Save the dataframes in files to be picked up by other scripts
feather.write_dataframe(train, "temp/preparedtraining.feather")
feather.write_dataframe(test, "temp/preparedtest.feather")
