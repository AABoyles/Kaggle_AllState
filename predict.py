#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
import feather

# Load Training Data
train = feather.read_dataframe("temp/preparedtraining.feather")
train_labels = np.array(train['log_loss'])
train_ids = train['id'].values.astype(np.int32)
train.drop(train.columns[[-1,-2]], 1, inplace = True)
train_d = xgb.DMatrix(train, label=train_labels)

# Load Test Data
test = feather.read_dataframe("temp/preparedtest.feather")
test_ids = test['id'].values.astype(np.int32)
test_d = xgb.DMatrix(test)

n_rounds = 100

# Discovered by the hyperoptimize.py script
params = {
    "colsample_bytree": 1,
    "eta": .5,
    "gamma": 1.7684,
    "max_depth": 12,
    "min_child_weight": 5,
    "subsample": 1
}

model = xgb.train(params, train_d, n_rounds, [(train_d, 'train')], verbose_eval = False,)

# Write the Results
result = pd.DataFrame(model.predict(test_d), columns=['loss'])
result["id"] = test_ids
result = result.set_index("id")
result.to_csv('outputs/hyperoptimizedxgb.csv', index=True, index_label='id')
