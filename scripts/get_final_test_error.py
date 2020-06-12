import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error



def execute_test_error():

	model = load('../output/models_predictions_pca/KNeighborsRegressor - Model 0.joblib')
	


def get_test_error(model):

	# Load test data
	test_features = load('../output/data/Data - Test Features PCA.joblib')
	test_target   = load('../output/data/Data - Test Target.joblib')
	results = pd.read_csv("../output/model_validation_results_with_pca.csv")
	print(f"Validated model is {results.loc[0,'Model']} with parameters {results.loc[0,'Parameters']}")

	#### TOP KNN RESULT ####

	# Identify top model and load it
	# NOTE - This must be changed manually - currently best model is 5-NN

	# model = load('../output/models_predictions_pca/KNeighborsRegressor - Model 0.joblib')

	# Get test data predictions
	predictions = model.predict(test_features)

	# Get error results
	MSE = mean_squared_error(test_target,predictions)
	MAE = mean_absolute_error(test_target,predictions)
	print(f"Mean Squared Error: {MSE:,.5}")
	print(f"Mean Absolute Error: {MAE:,.5}")

	#### TOP NON-KNN RESULT ####

	# Identify top model and load it
	# NOTE - This must be changed manually - currently best model is Random Forest
	model = load('../output/models_predictions_pca/RandomForestRegressor - Model 4.joblib')

	# Get test data predictions
	predictions = model.predict(test_features)

	# Get error results
	MSE = mean_squared_error(test_target, predictions)
	MAE = mean_absolute_error(test_target, predictions)
	print(f"Mean Squared Error: {MSE:,.5}")
	print(f"Mean Absolute Error: {MAE:,.5}")
	return MSE, MAE