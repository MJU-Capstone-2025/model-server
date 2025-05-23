from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(preds, targets):
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return {'rmse': rmse, 'mae': mae}