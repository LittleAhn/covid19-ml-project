import build_master_df
import pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_and_split_master_df():
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

    # Get full dataset for use
    df = raw.dropna(subset=[main_target])[features+[main_target]+['date']]

    # Split train test 
    train_full,test_full = pipeline.get_train_test(df,train_size=0.8,time_series=True)
    train_target = train_full[main_target]
    test_target = test_full[main_target]
    train_features = train_full.drop(columns=[main_target])
    test_features = test_full.drop(columns=[main_target])

    # Impute and normalize
    train_features,test_features = pipeline.impute_missing(train_features,test_features,how='median')
    train_features,test_features = pipeline.normalize_vars(train_features,test_features)

    return train_features,test_features,train_target,test_target


def get_pca_reduction(train_features,test_features):
    """
    Gets a dimensionality-reduced version of datasets
    using PCA. Chooses number of components so as to 
    explain 95% of the variance in the data.
    """

    # Identify number of components that explain 95% of variance
    pca = PCA(n_components=75)
    pca.fit(train_features)
    cumsums = np.cumsum(pca.explained_variance_ratio_)
    num_components = next(x for x,val in enumerate(cumsums) if val > 0.95)+1

    # Get final PCA-reduced datasets
    pca = PCA(n_components=num_components)
    pca.fit(train_features)
    train_pca_features = pca.transform(train_features)
    test_pca_features  = pca.transform(test_features)

    return train_pca_features,test_pca_features


def fit_and_eval_models(train_pca_features,train_target,
                        test_pca_features,test_target):
    """
    Function to fit a number of regression models,
    evaluate, and return the results. This takes a while to run.
    """

    # Config: Dictionaries of models and hyperparameters
    MODELS = {
        'LinearRegression': LinearRegression(), 
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'LinearSVR': LinearSVR(), 
        'RandomForestRegressor': RandomForestRegressor(),
        'AdaBoostRegressor':AdaBoostRegressor(),
        'KNeighborsRegressor':KNeighborsRegressor()
    }
    GRID = {
        'LinearRegression': [{}],
        'Lasso': [{'alpha':x, 'random_state':0, 'max_iter':10000} for x in [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]],
        'Ridge': [{'alpha':x, 'random_state':0, 'max_iter':10000} for x in [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]],
        'LinearSVR': [{'C': x, 'epsilon':y, 'random_state': 0, 'max_iter':10000} \
                    for x in [0.01,0.05,0.1,0.5,1,5]
                    for y in [0.01,0.1,1]],
        'RandomForestRegressor': [{'n_estimators':x, 'max_features':y,
                                'n_jobs':-1} \
                                for y in ['auto','log2','sqrt']
                                for x in [100,500,1000]],
        'AdaBoostRegressor': [{'n_estimators':y} for y in [50, 100, 150]],
        'KNeighborsRegressor': [{'n_neighbors':x} for x in np.arange(5,20)]
    }

    # Fit and get results
    model_results = pipeline.build_regressors(MODELS, GRID,
                                            train_pca_features, train_target,
                                            test_pca_features, test_target)
    return model_results


if __name__ == "main":

    train_features,test_features,train_target,test_target = build_and_split_master_df()
    train_pca_features,test_pca_features = get_pca_reduction(train_features,test_features)
    model_results = fit_and_eval_models(train_pca_features,train_target,
                                        test_pca_features,test_target)
    model_results.to_csv("../data/model_results.csv")