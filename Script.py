#!/usr/bin/env python3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb # XGBoost implementation

# Load the dataset.

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

features = [x for x in train.columns if x not in ['id','loss']]

train_x = train[features]
test_x = test[features]

for c in set(train.select_dtypes(include=['object']).columns) - set(['id','loss']):
    a = pd.DataFrame(train['loss'].groupby([train[c]]).mean())
    a.loc[:, c] = a.index
    train_x.loc[:, c] = pd.merge(left=train_x, right=a, how='left', on=c)['loss']
    test_x.loc[:, c] = pd.merge(left=test_x, right=a, how='left', on=c)['loss']

bst = xgb.train({
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 5,
    'min_child_weight': 1,
    'eval_metric': 'logloss'
    },
    xgb.DMatrix(train_x, train['loss']),
    num_boost_round = 1000)

test_x['loss'] = bst.predict(xgb.DMatrix(test_x))
test_x.drop(test_x.columns[[i for i in range(1,test_x.shape[1]-1)]], axis = 1, inplace = True)
test_x.to_csv('output.csv', index=None)
