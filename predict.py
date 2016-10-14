#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

n_rounds = 1000

# Discovered by the hyperoptimize.py script
params = {
    "colsample_bytree": 1,
    "eta": .01,
    "gamma": 1.7281,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 1
}

clf_full = xgb.train(params, d_train_full, n_rounds, [(d_train_full, 'train')],
    verbose_eval = False,)

# Write the Results
result_full = pd.DataFrame(clf_full.predict(d_test), columns=['loss'])
result_full["id"] = test_ids
result_full = result_full.set_index("id")
result_full.to_csv('outputs/hyperoptimizedxgb.csv', index=True, index_label='id')
