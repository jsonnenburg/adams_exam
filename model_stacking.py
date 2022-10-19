###############################################################################
# MODEL STACKING ##############################################################
###############################################################################

import numpy as np
from sklearn.linear_model import RidgeCV

# approach following https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
def stacked_dataset(model_nontext, model_text, X_nontext, X_text):
    """
    Create the input dataset for the meta model as outputs of the ensemble members.
    """
    stackX = None
    
    yhat_nontext = model_nontext.predict(X_nontext)
    yhat_text = model_text.predict(X_text)
    stackX = np.hstack((yhat_nontext.reshape(-1, 1), yhat_text))
    
    return stackX
 
def fit_stacked_model(model_nontext, model_text, stackedX, y):
    """
    Fit a meta model based on the predictions of the ensemble members. 
    """
    meta_model = RidgeCV(scoring='neg_root_mean_squared_error', cv=5, fit_intercept=False)
    meta_model.fit(stackedX, y)
    
    print('Ridge Regression coefficients:', meta_model.coef_)
    
    return meta_model
 
def stacked_prediction(model_nontext, model_text, X_tr_nontext, X_tr_text, y_tr, X_ts_nontext, X_ts_text):
    """
    Create the input dataset for the meta model, fit the meta model,
    and predict on the provided test set.
    
    @param model_nontext The fitted model trained on non-text features.
    @param model_text The fitted model trained on text_features, accepting [X_name, X_summary, X_description] as input.
    @param X_tr_nontext The training set of the non-text features.
    @param X_tr_text The training set of the text features in the format required by the text model.
    @param y_tr The training set labels.
    @param X_ts_nontext The test set of the non-text features in the format required by the text model.
    @param X_ts_text The test set of the text features in the format required by the text model.
    
    @returns yhat The predictions on the test set.
    """
    # create dataset using ensemble
    stackedX_tr = stacked_dataset(model_nontext, model_text, X_tr_nontext, X_tr_text)
    # fit the meta regressor
    meta_model = fit_stacked_model(model_nontext, model_text, stackedX_tr, y_tr)
    
    stackedX_ts = stacked_dataset(model_nontext, model_text, X_ts_nontext, X_ts_text)
    
    yhat = meta_model.predict(stackedX_ts)
    
    return yhat
