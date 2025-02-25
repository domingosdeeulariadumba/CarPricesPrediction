'''
        IMPORTING MODULES
        ------------
'''

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV



    # Regression Problems
    
# Function to return scores of candidate regressors and the best pipeline
def eval_reg(candidate_models, X_train, y_train):
    
    '''
    Evaluate candidate regression models and return the best model pipeline.
    
    Parameters:
    candidate_models (dict): A dictionary of candidate regression models.
    X_train (DataFrame): Training features.
    y_train (Series): Training target.

    Returns:
    tuple: DataFrame of model scores and the best pipeline.
    '''
    
    # Empty dictionary to store candidate models scores
    dict_scores = {} 
    
    # Function to retrieve the appropriate feature reduction
    def feat_red_strategy(X_train):
        
        if (X_train.shape[1] > 99) or (X_train.shape[1] / len(X_train) > 0.1):
            pca = PCA().fit(X_train)   
            n_comp = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.95) + 1
            return 'pca', PCA(n_components = n_comp)
        else:
            return 'feature selection', ''

    # Looping through different candidate models for evaluation
    for model_name, model_instance in candidate_models.items(): 
        feat_red_strategy_name, feat_red_strategy_model = feat_red_strategy(X_train)        
        if feat_red_strategy_name == 'pca':
            feat_red_strategy_step = ('pca', feat_red_strategy_model)
        else:            
            feat_red_strategy_step = ('feature selection', RFE(model_instance))
            
        # Building the pipeline for each model    
        model_pipeline = Pipeline(steps = [
            feat_red_strategy_step,
            ('regressor', model_instance)
        ])        
        
        # Conditions for each candidate model
        if model_name == 'OLS':
            scores = cross_val_score(model_pipeline, 
                                     X_train, y_train, 
                                     scoring ='neg_mean_squared_error')
            dict_scores[model_name] = scores.mean()

        else:
            # Parameters setup
            param_reg = {'regressor__alpha': [0.1, 1, 10],
                          'regressor__l1_ratio': [0.1, 0.5, 0.9]}    
            tree_depths, min_imp_decs = [3, 5, 7], [0.001, 0.01, 0.1]     
            param_tree = {'regressor__max_depth': tree_depths,
                         'regressor__min_impurity_decrease': min_imp_decs}    
            param_forest = {'regressor__max_depth': tree_depths}   
            param_xgb = {'regressor__lambda': [0.1, 1, 10],
                              'regressor__alpha': param_reg['regressor__alpha'],
                              'regressor__learning_rate': [0.1, 0.2, 0.3],
                              'regressor__max_depth': tree_depths,
                              'regressor__gamma': [0, 0.1, 0.2, 0.5, 1]}            
            if model_name in ['Lasso', 'Ridge']:
                param_grid_ = {'regressor__alpha': param_reg['regressor__alpha']}
            elif model_name == 'Elastic Net':
                param_grid_ = param_reg
            elif model_name == 'Decision Tree':
                param_grid_ = param_tree
            elif model_name == 'Random Forest':
                    param_grid_ = param_forest
            else:
                param_grid_ = param_xgb       
                
            gridsearch_cv = GridSearchCV(model_pipeline, 
                                         param_grid = param_grid_,
                                         scoring = 'neg_mean_squared_error')
            gridsearch_cv.fit(X_train, y_train)
            best_model = gridsearch_cv.best_estimator_
            scores = cross_val_score(best_model, X_train, y_train, 
                        scoring = 'neg_mean_squared_error')
            dict_scores[model_name] = scores.mean()

    # Dataframe to store the candidate models results in descending order
    df_scores = pd.DataFrame(list(dict_scores.items()), 
                             columns = ['Model', 'Mean Score (MSE)'])
    df_scores.sort_values(by = df_scores.columns[-1],
                          ascending = False, inplace = True)

    # Extracting the best model and its instance from sorted scores
    best_model_name = df_scores.iat[0, 0]
    best_model_instance = candidate_models[best_model_name]
    
    # Selecting the reduction strategy given the function above   
    if feat_red_strategy_name == 'pca':
        best_feat_red_strategy_step = feat_red_strategy_step
    else:
        best_feat_red_strategy_step = ('feature selection', 
                                       RFE(best_model_instance))

    # Building the best model pipeline with a valid feature reduction strategy
    best_model_pipeline = Pipeline(steps = [
        best_feat_red_strategy_step,
        ('regressor', best_model_instance)
    ])

    # Tuple containing the candidate models result and the best pipeline
    eval_data = (df_scores, best_model_pipeline)

    return eval_data




    # Classification Problems
    
# Function to return scores of candidate classifiers and the best pipeline
def eval_class(candidate_models, X_train, y_train):
    
    '''
    Evaluate candidate classification models and return the best model pipeline.
    
    Parameters:
    candidate_models (dict): A dictionary of candidate regression models.
    X_train (DataFrame): Training features.
    y_train (Series): Training target.

    Returns:
    tuple: DataFrame of model scores and the best pipeline.
    '''
        
    # Empty dictionary to store candidate models scores    
    dict_scores = {}    
    
    # Function to retrieve the appropriate feature reduction
    def feat_red_strategy(X_train):
        
        if (X_train.shape[1] > 99) or (X_train.shape[1] / len(X_train) > 0.1):
            pca = PCA().fit(X_train)   
            n_comp = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.95) + 1
            return 'pca', PCA(n_components = n_comp)
        else:
            return 'feature selection', ''

    # Looping through different candidate models for evaluation
    for model_name, model_instance in candidate_models.items(): 
        feat_red_strategy_name, feat_red_strategy_model = feat_red_strategy(X_train)        
        if feat_red_strategy_name == 'pca':
            feat_red_strategy_step = (feat_red_strategy_name, feat_red_strategy_model)
        else:            
            feat_red_strategy_step = (feat_red_strategy_name, RFE(model_instance))
            
        # Building the pipeline for each model    
        model_pipeline = Pipeline(steps = [
            feat_red_strategy_step,
            ('classifier', model_instance)
            ])
       
        # Scoring condition
        if len(set(y_train)) > 2:
            scoring_ = 'f1_micro'
        else:
            scoring_ = 'f1'       
        
        # Conditions for each candidate model
        if model_name == 'Logistic Regression':
            scores = cross_val_score(model_pipeline, X_train,
                                     y_train, scoring = scoring_)
            dict_scores[model_name] = scores.mean()
        else:
            # Parameters setup
                ## Tree-Based Models
            tree_depths, min_imp_decs = [3, 5, 7], [0.001, 0.01, 0.1]     
            param_tree = {'classifier__max_depth': tree_depths,
                         'classifier__min_impurity_decrease': min_imp_decs}    
            param_forest = {'classifier__max_depth': tree_depths}
            param_xgb = {'classifier__lambda': [0.1, 1, 10],
                              'classifier__alpha': [0.1, 1, 10],
                              'classifier__learning_rate': [0.1, 0.2, 0.3],
                              'classifier__max_depth': tree_depths,
                              'classifier__gamma': [0, 0.1, 0.2, 0.5, 1]}
            
                ## Distance-Based Models
            dist_pipeline = Pipeline(steps = [step for i, step
                                                 in enumerate(model_pipeline.steps) if i != 0])
            param_svm = {'classifier__kernel': ['linear', 'poly', 'rbf',
                                                'sigmoid'],
                'classifier__C': [0.1, 1, 10, 100]}
              
            param_knn = {
                'classifier__n_neighbors': [1, 2, 3, 5, 7, 9, 11, 13, 15],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['minkowski']
}
            
            if model_name == 'Decision Tree':
                param_grid_, model_pipeline = param_tree, model_pipeline
            elif model_name == 'SVM':
                param_grid_, model_pipeline = param_svm, dist_pipeline
            elif model_name == 'XGB':
                param_grid_, model_pipeline = param_xgb, model_pipeline
            elif model_name == 'KNN':
                param_grid_, model_pipeline = param_knn, dist_pipeline                      
            else:
                param_grid_, model_pipeline = param_forest, model_pipeline 
                
            # Grid Search Cross Validation
            gridsearch_cv = GridSearchCV(model_pipeline, 
                param_grid = param_grid_, scoring = scoring_)
            gridsearch_cv.fit(X_train, y_train)
            best_model = gridsearch_cv.best_estimator_
            scores = cross_val_score(best_model, X_train, y_train,
                        scoring = scoring_)
            dict_scores[model_name] = scores.mean()
            
    # Dataframe to store the candidate models results in descending order
    df_scores = pd.DataFrame(list(dict_scores.items()), 
                             columns = ['Model',
                                        f'Mean Score ({scoring_.replace("_", " ").title()})'])
    df_scores.sort_values(by = df_scores.columns[-1],
                          ascending = False, inplace = True)

    # Extracting the best model and its instance from sorted scores
    best_model_name = df_scores.iat[0, 0]
    best_model_instance = candidate_models[best_model_name]
    
    # Selecting the reduction strategy given the function above   
    if feat_red_strategy_name == 'pca':
        best_feat_red_strategy_step = feat_red_strategy_step
    else:
        best_feat_red_strategy_step = ('feature selection', 
                                       RFE(best_model_instance))

    # Building the best model pipeline with a valid feature reduction strategy
    if best_model_name in ('SVM', 'KNN'):
        best_model_pipeline = dist_pipeline         
    else:     
        best_model_pipeline = Pipeline(steps = [
            best_feat_red_strategy_step,
            ('classifier', best_model_instance)
            ])

    # Tuple containing the candidate models result and the best pipeline
    eval_data = (df_scores, best_model_pipeline)

    return eval_data
