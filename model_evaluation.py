###############################################################################
# MODEL EVALUATION ############################################################
###############################################################################

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    """
    https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def eval_model_nontext(estimator, x_train, x_val, y_train, y_val, model_name=''):
    
    y_pred = estimator.predict(x_val)
    y_train_pred = estimator.predict(x_train)
    
    print('Evaluating estimator:',model_name)
    print('='*40)
    print('RMSE  train: %.3f,  test: %.3f' %(np.sqrt(mean_squared_error(y_train, y_train_pred)),np.sqrt(mean_squared_error(y_val, y_pred))))
    print('R^2   train: %.3f,  test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_val, y_pred)))

def eval_model_text(model, X_val, y_val, model_name=''):
    """
    X_val in the form of [X_val_name, X_val_summary, X_val_description].
    """
    y_pred = model.predict(X_val)

    preds = pd.DataFrame(y_pred, columns=["name_pred", "summary_pred", "description_pred"])

    preds['actual'] = y_val
    
    print('Evaluating model:',model_name)
    print('='*40)
    print(f"RMSE name: {root_mean_squared_error(preds.name_pred, preds.actual):.3f}")
    print(f"RMSE summary: {root_mean_squared_error(preds.summary_pred, preds.actual):.3f}") 
    print(f"RMSE description: {root_mean_squared_error(preds.description_pred, preds.actual):.3f}")
    
def compute_metrics(model, X_val, y_val):
    """
    Compute RMSE and MAE for a given model.
    """
    y_pred = model.predict(X_val)
    
    rmse = round(np.sqrt(mean_squared_error(y_pred, y_val)), 6)
    mae = round(mean_absolute_error(y_pred, y_val), 6)
    
    return rmse, mae
    