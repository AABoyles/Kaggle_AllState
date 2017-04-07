#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
import statsmodels.formula.api as smf
import feather

# Load Training Data
train = pd.read_csv('data/train.csv', dtype={'id': np.int32})
train['log_loss'] = np.log(train['loss'])

# Load Test Data
test = pd.read_csv('data/test.csv', dtype={'id': np.int32})

def mean_corr(log=False):
    var = 'loss'
    if log: var = 'log_loss'
    mean_correlation = 0
    for feat in test.dtypes[test.dtypes != "object"].index:
        mean_correlation = mean_correlation + (np.corrcoef(train[feat], train[var])**2)[0,1]
    mean_correlation = mean_correlation/len(features_numeric)
    return(mean_correlation)

# compute skew and do Box-Cox transformation
# transform features with skew > 0.25 (this can be varied to find optimal value)
features_numeric = test.dtypes[test.dtypes != "object"].index
features_skewed = train[features_numeric].apply(lambda x: skew(x.dropna()))
features_skewed = features_skewed[features_skewed > 0.25]
for feat in features_skewed.index:
    train[feat], lam = boxcox(train[feat] + 1)
    test[feat] = boxcox(test[feat] + 1, lam)

# Replace Categorical Features with the Category's mean log loss
features_categorical = [feat for feat in test.columns if 'cat' in feat]

for feat in features_categorical:
    a = pd.DataFrame(train['log_loss'].groupby([train[feat]]).mean())
    a[feat] = a.index
    train[feat] = pd.merge(left=train, right=a, how='left', on=feat)['log_loss_y']
    test[feat] = pd.merge(left=test, right=a, how='left', on=feat)['log_loss']

temp = test.dropna(1)
model = smf.ols('log_loss ~ ' + ' + '.join(temp.columns), data=train.drop('loss', 1)).fit()
yhat = np.exp(model.predict(test))

result = pd.DataFrame({'id': test['id'].values, 'loss': yhat})
result = result.set_index('id')
result.to_csv('outputs/simplelm.csv', index=True, index_label='id')

# Save the dataframes in files to be picked up by other scripts
feather.write_dataframe(train, "temp/preparedtraining.feather")
feather.write_dataframe(test, "temp/preparedtest.feather")
