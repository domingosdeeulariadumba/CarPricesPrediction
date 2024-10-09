

'''
        IMPORTING MODULES
        ------------
'''

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split as tts, cross_val_score as cv, GridSearchCV
from sklearn.linear_model import LinearRegression as lreg, Lasso, Ridge, ElasticNet
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor as rfr
import xgboost as xgb



'''
        Implementation
        --------------
'''


def eval_reg(candidate_models, X_train, y_train):
    
    """
    Evaluate candidate regression models using cross-validation and return sorted scores.

    This function loops through a set of candidate models, applies scaling and feature selection,
    performs cross-validation, and computes the mean cross-validated score for each model. For specific
    models like ElasticNet and XGBoost, it also performs hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    candidate_models : dict
        A dictionary of model names (as keys) and corresponding instantiated model objects (as values).
        Example: {'OLS': LinearRegression(), 'Decision Tree': DecisionTreeRegressor(), 'XGB': XGBRegressor(), etc.}
    
    X_train : array-like
        Training data features. This is the matrix of independent variables (predictors).
    
    y_train : array-like
        Target variable corresponding to X_train. This is the dependent variable (response) that the models
        aim to predict.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with two columns:
            - 'Model': The name of each model.
            - 'Mean Score': The mean cross-validated score for each model (negative mean squared error).
        The DataFrame is sorted by the 'Mean Score' in descending order.

    Notes
    -----
    - RobustScaler is used to scale the features.
    - RFE (Recursive Feature Elimination) is applied for feature selection.
    - Models like ElasticNet and XGBoost undergo hyperparameter tuning using GridSearchCV.
    - For other models, only the alpha parameter is tuned via GridSearchCV.
    - Cross-validation is performed using the 'neg_mean_squared_error' scoring metric.

    Example
    -------
    >>> from sklearn.linear_model import ElasticNet
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from xgboost import XGBRegressor
    >>> models = {
    >>>     'OLS': LinearRegression(),
    >>>     'Decision Tree': DecisionTreeRegressor(),
    >>>     'XGB': XGBRegressor(),
    >>>     'ElasticNet': ElasticNet()
    >>> }
    >>> df_results = eval_reg(models, X_train, y_train)
    >>> print(df_results)
    
    """
    
    dict_scores = {}

    param_grid = {'regressor__alpha': [0.1, 1, 10],
                  'regressor__l1_ratio': [0.1, 0.5, 0.9]}

    param_grid_xgb = {'regressor__lambda': [0.1, 1, 10],
                      'regressor__alpha': [0.1, 1, 10]}

    for model_name, model_instance in candidate_models.items():
        model_pipeline = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('feature_selection', RFE(model_instance)),
            ('regressor', model_instance)
        ])

        if model_name in ['OLS', 'Decision Tree', 'Random Forest']:
            scores = cv(model_pipeline, X_train, y_train, scoring='neg_mean_squared_error')
            dict_scores[model_name] = scores.mean()

        elif model_name == 'XGB':
            gridsearch_cv = GridSearchCV(model_pipeline, param_grid=param_grid_xgb,
                                         scoring='neg_mean_squared_error')
            gridsearch_cv.fit(X_train, y_train)
            best_model = gridsearch_cv.best_estimator_
            scores = cv(best_model, X_train, y_train, scoring='neg_mean_squared_error')
            dict_scores[model_name] = scores.mean()

        elif model_name == 'ElasticNet':
            gridsearch_cv = GridSearchCV(model_pipeline, param_grid=param_grid, scoring='neg_mean_squared_error')
            gridsearch_cv.fit(X_train, y_train)
            best_model = gridsearch_cv.best_estimator_
            scores = cv(best_model, X_train, y_train, scoring='neg_mean_squared_error')
            dict_scores[model_name] = scores.mean()

        else:
            gridsearch_cv = GridSearchCV(model_pipeline, param_grid={'regressor__alpha': param_grid['regressor__alpha']},
                                         scoring='neg_mean_squared_error')
            gridsearch_cv.fit(X_train, y_train)
            best_model = gridsearch_cv.best_estimator_
            scores = cv(best_model, X_train, y_train, scoring='neg_mean_squared_error')
            dict_scores[model_name] = scores.mean()

    df_scores = pd.DataFrame(list(dict_scores.items()), columns=['Model', 'Mean Score'])
    df_scores.sort_values(by = 'Mean Score', ascending = False, inplace = True)

    return df_scores