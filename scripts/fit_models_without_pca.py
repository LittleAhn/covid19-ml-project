import build_master_df
import pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_and_split_master_df(save_path):
    """
    Function to call other scripts that build the final dataset.
    This also does some processing, such as doing a train-test
    split. (where test data = 20% most recent dates observed)
    """

    raw = build_master_df.build_df()

    # Identify variables
    index_vars   = ['StateName','CountyName','fips','date']
    target_vars  = [col for col in raw.columns if (col.endswith('change_from_baseline'))]
    main_target  = 'retail_and_recreation_percent_change_from_baseline'
    features     = [col for col in raw.columns if (col not in index_vars) and (col not in target_vars)]

    # Get full dataset for use and drop unnecessary variables
    df = raw.dropna(subset=[main_target])[features+[main_target]+['date']]

    # Split train test 
    train_full,validation_full,test_full = pipeline.get_train_test(df,train_size=0.8,
                                                                time_series=True,validation=True)
    train_target = train_full[main_target]
    validation_target = validation_full[main_target]
    test_target = test_full[main_target]
    train_features = train_full.drop(columns=[main_target])
    validation_features = validation_full.drop(columns=[main_target])
    test_features = test_full.drop(columns=[main_target])

    # Impute and normalize
    train_features,validation_features,test_features = pipeline.impute_missing(train_features,test_features,
                                                                               validation_features,how='median')
    train_features,validation_features,test_features = pipeline.normalize_vars(train_features,test_features,
                                                                               validation_features)

    # Save output
    dump(train_features, save_path+"/Data - Train Features.joblib")
    dump(validation_features, save_path+"/Data - Validation Features.joblib")
    dump(test_features, save_path+"/Data - Test Features.joblib")
    dump(train_target, save_path+"/Data - Train Target.joblib")
    dump(validation_target, save_path+"/Data - Validation Target.joblib")
    dump(test_target, save_path+"/Data - Test Target.joblib")                              

    return train_features,validation_features,test_features,train_target,validation_target,test_target

def fit_and_eval_models(train_features,train_target,
                        validation_features,validation_target,
                        save_path):
    """
    Function to fit a number of regression models,
    evaluate, and return the results. This takes a while to run.
    """

    # Config: Dictionaries of models and hyperparameters
    MODELS = {
        'LinearRegression': LinearRegression(), 
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor(),
        'AdaBoostRegressor':AdaBoostRegressor(),
    }
    GRID = {
        'LinearRegression': [{}],
        'Lasso': [{'alpha':x, 'random_state':0, 'max_iter':10000} for x in [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]],
        'RandomForestRegressor': [{'n_estimators':x, 'max_features':'log2',
                                   'n_jobs':-1} \
                                    for x in [100,500,1000]],
        'AdaBoostRegressor': [{'n_estimators':y} for y in [50,75,100,125,150,175,200]],
    }

    # Fit and get results
    model_results = pipeline.build_regressors(MODELS, GRID,
                                              train_features, train_target,
                                              validation_features, validation_target,
                                              save_path)
    return model_results


if __name__ == "__main__":

    save_path = "../output"
    train_features,validation_features,test_features,train_target,validation_target,test_target = build_and_split_master_df(save_path)
    model_results = fit_and_eval_models(train_features,train_target,
                                        validation_features,validation_target,
                                        save_path)
    
    # Sort and save results
    model_results = model_results.sort_values('MAE')
    model_results.to_csv("../output/model_validation_results_without_pca.csv")
