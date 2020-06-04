import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load test data
test_features = load('../output/Data - Test Features.joblib')
test_target   = load('../output/Data - Test Target.joblib')

# Identify top model and load it
# NOTE - This must be changed manually - currently best model is AdaBoost 6
results = pd.read_csv("../output/model_validation_results_without_pca.csv")
print(f"Validated model is {results.loc[0,'Model']} with parameters {results.loc[0,'Parameters']}")
model = load('../output/AdaBoostRegressor - Model 6.joblib')

# Get test data predictions
predictions = model.predict(test_features)

# Get error results
MSE = mean_squared_error(test_target,predictions)
MAE = mean_absolute_error(test_target,predictions)
print(f"Mean Squared Error: {MSE:,.5}")
print(f"Mean Absolute Error: {MAE:,.5}")