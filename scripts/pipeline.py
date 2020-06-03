from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,mean_squared_error,mean_absolute_error
from joblib import dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime
import os
import pickle

def read_data(df, csvsep=","):
    '''
    Returns a DataFrame, either directly (if passed) or read as a CSV.
    df: a DataFrame or file path to CSV file 
    (further file types other than CSV to be implemented later)
    '''
    # Return plain pandas dataframe
    if isinstance(df, pd.DataFrame):
        return df

    # Read file path 
    elif os.path.exists(df):
        _,ext = os.path.splitext(df)

        # CSVs
        if ext.lower() == '.csv':
            return (pd.read_csv(df, sep=csvsep))

        # Other file types implemented later (maybe pickled files?)
        else:
            raise TypeError("Please pass a valid file type.")
    else:
        raise TypeError("Please pass a valid file path.")


def get_numeric_correlations(df, method='pearson'):
    '''
    Prints a table of pairwise correlation for all numeric
    columns in a pandas DataFrame object. Returns null. 
    '''
    # Check pandas df
    if isinstance(df, pd.DataFrame):
        numeric_df = df.select_dtypes(include=['number'])
        print(numeric_df.corr(method=method))
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def get_numeric_histograms(df, nbins=40, log=False):
    '''
    Displays histograms of all numeric columns in a 
    pandas DataFrame. Returns null.
    '''
    # Check pandas df
    if isinstance(df, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['number'])
        for col in numeric_cols.columns:

            # Remove NaN values
            toplot = numeric_cols[col].dropna()

            # Make histogram
            plt.figure(figsize=(10,5))
            plt.hist(toplot, bins=nbins, log=log)
            plt.title(f"Histogram of {col}", fontsize=15)
            plt.xlabel(f"Value for {col}", fontsize=10)
            plt.ylabel("Number of Observations", fontsize=10)
            plt.yticks(fontsize=9)
            plt.xticks(fontsize=9)
            plt.show()
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def get_categorical_tables(df, pairwise=False, maxlevels=20):
    '''
    Displays oneway or twoway frequency tables of all object-type
    columns in a pandas DataFrame. Returns null. 
    Parameters:
        pairwise - default False. If True, displays each pairwise
         (i.e., twoway) frequency table. 
        maxlevels - default 20. An integer describing the max number
         of levels displayed for a variable in any frequency table. 
         (e.g., limiting the size of the table if there are more than
         a certain number of levels for a variable)
    '''
    # Check pandas df
    if isinstance(df, pd.DataFrame):
        cat_cols = df.select_dtypes(include=['object']).columns

        # Display all one-way tables
        if pairwise == False:
            for cat_col in cat_cols:
                numlevels = df[cat_col].nunique()

                # Deal with cases when we exceed maxlevels
                if numlevels < maxlevels:
                    print(f"{df[cat_col].value_counts()}\n")
                else:
                    print(f"{df[cat_col].value_counts()[:maxlevels]}\n")

        # Display all two-way tables
        else:
            for cat_col1,cat_col2 in itertools.combinations(cat_cols,2):
                numlevels1 = df[cat_col1].nunique()
                numlevels2 = df[cat_col2].nunique()
                printdf = df[[cat_col1,cat_col2]]

                # Deal with cases when we exceed maxlevels
                if (numlevels1 > maxlevels) and (numlevels2 <= maxlevels):
                    kept_vals1 = df[cat_col1].value_counts()[:maxlevels]
                    printdf = df[df[cat_col1] in kept_vals1]
                elif (numlevels1 <= maxlevels) and (numlevels2 > maxlevels):
                    kept_vals2 = df[cat_col2].value_counts()[:maxlevels]
                    printdf = df[df[cat_col2] in kept_vals2]
                elif (numlevels1 > maxlevels) and (numlevels2 > maxlevels):
                    kept_vals1 = df[cat_col1].value_counts()[:maxlevels]
                    kept_vals2 = df[cat_col2].value_counts()[:maxlevels]
                    printdf = df[(df[cat_col1] in kept_vals1) and (df[cat_col2] in kept_vals2)]
                
                # Print twoway table (adjusted for maxvals, if necessary)
                print(pd.crosstab(printdf[cat_col1], printdf[cat_col2]))
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def get_train_test(df, train_size=0.8, random_state=1, time_series=False):
    '''
    Calls sklearn's train_test_split on an input df and returns 
    the resulting train and test set DataFrames. Takes in parameters
    for training size and random seed. Ignores the potential need
    for a validation set.
    '''
    # Check pandas df
    if isinstance(df, pd.DataFrame):

        # Non-time series split
        if not time_series:
            train, test = train_test_split(df, train_size=train_size, 
                                            test_size=1-train_size, 
                                            random_state=random_state)
        elif time_series:
            df['date'] = df['date'].astype('datetime64')
            cutoff = df['date'].astype('datetime64').quantile(0.8)
            train = df[df['date'] <= cutoff].drop(columns='date')
            test  = df[df['date'] >  cutoff].drop(columns='date')
                    
        return (train, test)
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def impute_missing(train, test, how='median', verbose=False):
    '''
    Imputes missing values for numeric columns. Returns
    train and test data with missing values imputed.
    (Imputation done based on train columns)
    Parameter:
    df - pandas DataFrame to impute
    how - how to impute. default median. also accepts
     'mean' or some numeric value.
    '''
    # Check pandas df
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
        numeric_cols = train.select_dtypes(include=['number']).columns

        # Loop over numeric columns
        for col in numeric_cols:

            # Get missing value counts
            trainnas = train[col].isna().sum()
            testnas = test[col].isna().sum()

            # Only impute if no missing values (saves time)
            if (trainnas > 0) or (testnas > 0):

                # Verbosity
                if verbose == True:
                    print(f"Imputing {trainnas+testnas:,} missing values for {col}")

                # Impute
                if how=='median':
                    train.loc[:,col] = train.loc[:,col].fillna(train.loc[:,col].median())
                    test.loc[:,col] = test.loc[:,col].fillna(train.loc[:,col].median())
                elif how=='mean':
                    train.loc[:,col] = train.loc[:,col].fillna(train.loc[:,col].mean())
                    test.loc[:,col] = test.loc[:,col].fillna(train.loc[:,col].mean())
                elif isinstance(how,(int,float)):
                    train.loc[:,col] = train.loc[:,col].fillna(how)
                    test.loc[:,col] = test.loc[:,col].fillna(how)
                else:
                    raise TypeError("Please pass a valid 'how' value.")
            
            else:
                if verbose == True:
                    print(f"No missing values for {col}")

        return train,test
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def normalize_vars(train, test, verbose=False):
    '''
    Normalizes numeric variables in train and test dataset to 
    have mean 0 and standard deviation of 1. Returns the train and
    test dataset with normalized features.
    '''
    # Check pandas df
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):            

        # Split into numeric and non-numeric
        train_numeric = train.select_dtypes(include=['number'])
        test_numeric = test.select_dtypes(include=['number'])
        train_nonnumeric = train.select_dtypes(exclude=['number'])
        test_nonnumeric = test.select_dtypes(exclude=['number'])

        # Verbosity
        if verbose == True:
            print(f"The following columns will be normalized: {list(train_numeric.columns)}")

        # Scale numeric data
        scaler = StandardScaler()
        trainscale = pd.DataFrame(scaler.fit_transform(train_numeric))
        testscale = pd.DataFrame(scaler.transform(test_numeric))

        # Fix columns and indices
        trainscale.columns = train_numeric.columns
        trainscale.index = train_numeric.index
        testscale.columns = test_numeric.columns
        testscale.index = test_numeric.index

        # Merge back onto non-numeric data
        train = trainscale.join(train_nonnumeric)
        test = testscale.join(test_nonnumeric)

        return train, test
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def one_hot_encode(train, test, verbose=False):
    '''
    One-hot encodes training and testing data, returning
    new train and test data. Note - if a training data value is not in the
    test data, a new column with all zeros is added. If a test data value is not
    in the training data, that column is not included in the one hot encodings.
    '''
    # Check pandas df
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):            
        
        # Split into numeric and non-numeric
        train_numeric = train.select_dtypes(include=['number'])
        test_numeric = test.select_dtypes(include=['number'])
        train_nonnumeric = train.select_dtypes(exclude=['number'])
        test_nonnumeric = test.select_dtypes(exclude=['number'])

        # Verbosity
        if verbose == True:
            print(f"The following training columns will be one-hot encoded: {list(train_nonnumeric.columns)}")

        # Scale numeric data
        train_1he = pd.get_dummies(train_nonnumeric)
        test_1he = pd.get_dummies(test_nonnumeric)

        # Add 0 columns to test data, as necessary
        testcols = set(test_1he.columns)
        traincols = set(train_1he.columns)
        for col in train_1he.columns:
            if col not in testcols:
                test_1he[col] = 0
                if verbose == True:
                    print(f"Added 0s column for {col} in test, present in training")
        for col in test_1he.columns:
            if col not in traincols:
                test_1he = test_1he.drop(columns=col)
                if verbose == True:
                    print(f"Dropped {col} from test, not in training")
        assert(test_1he.columns.shape[0] == train_1he.columns.shape[0])

        # Merge back onto numeric data
        train = train_1he.join(train_numeric)
        test = test_1he.join(test_numeric)

        return train, test

    else: 
        raise TypeError("df must be a pandas DataFrame object")

def discretize(dfcol, bins=10, right=True):
    '''
    Discretizes a pandas series into a number of bins.
    Bins can be an integer or a series of bin cut values.
    Note - this function should be run before 'one_hot_encode', 
    since it returns a categorical Pandas Series and not
    a one-hot encoded set of series.
    '''
    return (pd.cut(dfcol, bins=bins, right=right))


def build_classifiers(models, params_grid, 
                      train_features, train_outcome, 
                      test_features, test_outcome):
    '''
    Trains a number of models and returns a DataFrame 
    of these models and their resulting evaluation metrics.
    Parameters:
     models - dictionary of sklearn models to fit
     parameters - dictionary of parameters to test for each of above models
     train_features, train_outcome - training data (pd.DataFrame)
     test_features, test_outcome - test data (pd.DataFrame) 
    '''
    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    results = pd.DataFrame(columns=["Model","Parameters",
                                    "Accuracy","F1 Score",
                                    "Precision","Recall"])

    # Loop over models 
    for model_key in models.keys(): 
        
        # Loop over parameters 
        for params in params_grid[model_key]: 
            
            # Create model 
            model = models[model_key]
            model.set_params(**params)

            # Start timing for fit
            startmodel = datetime.datetime.now()
            print("\tTraining:", model_key, "|", params)

            # Fit model on training set 
            model.fit(train_features, train_outcome)

            # Finish timing for fit
            endmodel = datetime.datetime.now()
            print("\tTime elapsed to train: ",endmodel-startmodel,"\n")
            
            # Predict on testing set 
            test_pred = model.predict(test_features)
            
            # Evaluate predictions 
            accuracy = evaluate_classifier('accuracy',test_pred,test_outcome)
            f1score = evaluate_classifier('f1score',test_pred,test_outcome)
            precision = evaluate_classifier('precision',test_pred,test_outcome)
            recall = evaluate_classifier('recall',test_pred,test_outcome)

            # Store results in your results data frame 
            newrow = pd.DataFrame([[model_key,params,accuracy,
                                   f1score,precision,recall]], 
                                   columns=["Model","Parameters",
                                            "Accuracy","F1 Score",
                                             "Precision", "Recall"])
            results = results.append(newrow)
            
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed For All Fitting and Prediction:", stop - start)

    return results        

def build_regressors(models, params_grid, 
                      train_features, train_outcome, 
                      test_features, test_outcome,
                      save_path):
    '''
    Trains a number of models and returns a DataFrame 
    of these models and their resulting evaluation metrics.
    Parameters:
     models - dictionary of sklearn models to fit
     parameters - dictionary of parameters to test for each of above models
     train_features, train_outcome - training data (pd.DataFrame)
     test_features, test_outcome - test data (pd.DataFrame) 
    '''
    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    results = pd.DataFrame(columns=["Model","Parameters",
                                    "MSE","MAE"])

    # Loop over models 
    for model_key in models.keys(): 
        
        # Loop over parameters 
        for idx,params in enumerate(params_grid[model_key]): 
            
            # Create model 
            model = models[model_key]
            model.set_params(**params)

            # Start timing for fit
            startmodel = datetime.datetime.now()
            print("\tTraining:", model_key, "|", params)

            # Fit model on training set 
            model.fit(train_features, train_outcome)

            # Predict on testing set 
            test_pred = model.predict(test_features)

            # Save results
            dump(model, save_path+f"/{model_key} - Model {idx}.joblib")
            dump(test_pred, save_path+f"/{model_key} - Predictions {idx}.joblib")

            # Finish timing
            endmodel = datetime.datetime.now()
            print("\tTime elapsed to train and predict: ",endmodel-startmodel,"\n")

            # Evaluate predictions 
            MSE = mean_squared_error(test_outcome,test_pred)
            MAE = mean_absolute_error(test_outcome,test_pred)

            # Store results in your results data frame 
            newrow = pd.DataFrame([[model_key,params,MSE,MAE]],
                                   columns=["Model","Parameters",
                                            "MSE","MAE"])
            results = results.append(newrow)
            
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed For All Fitting and Prediction:", stop - start)

    return results        

def evaluate_classifier(metric, test_outcome, test_pred):
    '''
    Returns given evaluation metric for a classifier.
    'metric' indicates which evaluation metric to use. Currently
    supported are accuracy, F1 score, precision, and recall.
    '''
    if metric == "accuracy":
        return(accuracy_score(test_outcome,test_pred))
    elif metric == "f1score":
        return(f1_score(test_outcome,test_pred))
    elif metric == "precision":
        return(precision_score(test_outcome,test_pred))
    elif metric == "recall":
        return(recall_score(test_outcome,test_pred))
    else:
        raise ValueError("Given metric not supported!")
