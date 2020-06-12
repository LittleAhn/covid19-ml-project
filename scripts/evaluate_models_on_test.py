import pandas as pd
import numpy as np
import joblib as jl
from os import listdir
from os.path import join

OUTPUT = "../output"
DATA = "../output/data"



def get_models_from_folder(pca=True):
	"""
	returns list of directories to model files
	"""

	folder = 'models_predictions_pca' if pca else 'models_predictions_nopca'
	models = [join(OUTPUT, folder, m) for m in listdir(join(OUTPUT, folder)) if 'Model' in m and '0.9' in m]

	return models


def get_test_predictions_files(pca=True):

	folder = 'models_predictions_pca' if pca else 'models_predictions_nopca'
	predictions = [join(OUTPUT, folder, p) for p in listdir(join(OUTPUT, folder)) \
		if 'Predictions' in p and 'Test' in p]

	return predictions


def load_test_data(pca=True):


	flist = [f for f in listdir(DATA) if 'Test' in f]
	if pca:
		flist = [f for f in flist if 'PCA' in f]


	if len(flist) != 1:
		raise Exception("File is not uniquely identified")

	feat_file = flist[0]
	test_feats = jl.load(join(DATA, feat_file))
	return test_feats


def load_test_target():

	return jl.load(join(DATA, 'Data - Test Target.joblib'))



def generate_prediction(model, test_feats):
	"""
	models is a string that refers to the joblib file for the model
	"""
	print('predicting for', model)
	model = jl.load(model)
	predictions = model.predict(test_feats)
	return predictions


def generate_all_predictions(model_list, test_feats):

	df = pd.DataFrame

	for m in model_list:

		predictions = generate_prediction(m, test_feats)
		path = get_save_path(m)
		jl.dump(predictions, path)


def execute_all_predictions():

	for pca in (True, False):
		test_data = load_test_data(pca)
		model_list = get_models_from_folder(pca)

		generate_all_predictions(model_list, test_data)

	print('done')


def get_save_path(m):

	p = m.replace('Model', 'Predictions')
	p = p[:p.find('0.8')] + 'Test.joblib'

	return p


def calc_MAE(test_target, predictions, var):

	# print('test_target', type(test_target))
	# print('predictions', type(predictions))
	mae = abs(test_target[var] - predictions).mean()

	return mae


def calc_MAE_by_model(prediction_list, test_target, pca=True):

	maes = []
	var = 'retail_and_recreation_percent_change_from_baseline'

	name_cutoff = 33 if pca else 35

	# for p, n in zip(prediction_list, names):
	for p in prediction_list:
		prediction = jl.load(p)
		n = p[name_cutoff:p.find(' - Test')]

		# return prediction
		mae = calc_MAE(test_target, prediction, var)
		maes.append((n, mae))
		# print(mae)

	df = pd.DataFrame.from_records(maes)
	df.columns = ['Model', 'MAE']
	# print(df)

	return df


def execute_MAE_cal():

	test_target = load_test_target()
	prediction_list = get_test_predictions_files(pca=True)
	rv = calc_MAE_by_model(prediction_list, test_target)
	rv['version'] = rv['Model'].str[-1]
	# rv['Model'] = rv['Model'].apply(lambda x: type(x['Model']))
	rv['Model'] = rv['Model'].apply(lambda x: x[:x.find(' - ')])

	print('outputting csv...')
	rv.to_csv(join(OUTPUT, 'test_MAEs.csv'), index=False)

	return rv


# def execute_test_error():

# 	model = load('../output/models_predictions_pca/KNeighborsRegressor - Model 0.joblib')
# 	get_test_error(model)


# def get_test_error(model):

# 	# Load test data
# 	test_features = load('../output/data/Data - Test Features PCA.joblib')
# 	test_target   = load('../output/data/Data - Test Target.joblib')
# 	results = pd.read_csv("../output/model_validation_results_with_pca.csv")
# 	print(f"Validated model is {results.loc[0,'Model']} with parameters {results.loc[0,'Parameters']}")

# 	#### TOP KNN RESULT ####

# 	# Identify top model and load it
# 	# NOTE - This must be changed manually - currently best model is 5-NN

# 	# model = load('../output/models_predictions_pca/KNeighborsRegressor - Model 0.joblib')

# 	# Get test data predictions
# 	predictions = model.predict(test_features)

# 	# Get error results
# 	MSE = mean_squared_error(test_target,predictions)
# 	MAE = mean_absolute_error(test_target,predictions)
# 	print(f"Mean Squared Error: {MSE:,.5}")
# 	print(f"Mean Absolute Error: {MAE:,.5}")

# 	#### TOP NON-KNN RESULT ####

# 	# Identify top model and load it
# 	# NOTE - This must be changed manually - currently best model is Random Forest
# 	model = load('../output/models_predictions_pca/RandomForestRegressor - Model 4.joblib')

# 	# Get test data predictions
# 	predictions = model.predict(test_features)

# 	# Get error results
# 	MSE = mean_squared_error(test_target, predictions)
# 	MAE = mean_absolute_error(test_target, predictions)
# 	print(f"Mean Squared Error: {MSE:,.5}")
# 	print(f"Mean Absolute Error: {MAE:,.5}")
# 	return MSE, MAE


if __name__ == '__main__':


	print('whaddup')





