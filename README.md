# Kaggle Competition: AllState

## USE

1. Execute `prep.py`. This will generate feather files in the temp/ directory, on which the other scripts will rely.
2. Execute `hyperoptimize.py`. This will execute a Bayesian Optimizer, which will then attempt to estimate the hyperparameters which yield the best outcomes from XGBoost on this data. It doesn't generate any other files, but it will print a log of hyperparameters and scores to STDOUT. You should note the best score and transcribe the hyperparameters that yielded it into the `predict.py` script.
3. Execute `predict.py`. This will output a CSV file in the outputs/ directory. Submit this file to Kaggle for scoring.
