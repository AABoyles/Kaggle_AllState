#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import feather

# if you're paranoid about overfitting, increase this.
n_folds = 10

# if you see metrics dropping precipitously until the end, increase this.
n_rounds = 100

# Set this to anything you want.
seed = 10

# Load Training Data
train = feather.read_dataframe("temp/preparedtraining.feather")
train_labels = np.array(train['log_loss'])
train.drop(train.columns[[-1,-2]], 1, inplace = True)
train_d = xgb.DMatrix(train, label=train_labels)

# Load Test Data
test = feather.read_dataframe("temp/preparedtest.feather")
test_d = xgb.DMatrix(test)

# Discovered by the hyperoptimize.py script
params = {
    "colsample_bytree": .9921,
    "eta": .0995,
    "gamma": 3.8581,
    "max_depth": 11,
    "min_child_weight": 1.0065,
    "subsample": 1
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

res = xgb.cv(params, train_d, n_rounds, n_folds, early_stopping_rounds = 15, seed = seed, feval = xg_eval_mae)

n_rounds = res.shape[0] - 1

model = xgb.train(params, train_d, n_rounds)

# Write the Results
result = pd.DataFrame(np.exp(model.predict(test_d)), columns=['loss'])
result["id"] = test['id'].values.astype(np.int32)
result = result.set_index("id")
result.to_csv('outputs/hyperoptimizedxgb.csv', index=True, index_label='id')
