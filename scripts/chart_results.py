### PLOTTING FUNCTIONS ONLY SUPPORT 1 TARGET VARIABLE PER RUNs

import pandas as pd
import numpy as np
import joblib as jl
# import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from os import listdir
from os.path import join
# import pipeline
import utils
import fit_models_without_pca
import fit_models_with_pca

OUTPUT = "../output"
DATA = "../output/data"

LINE_PLOT_LABELS = {
	'observed_retail_and_recreation_percent_change_from_baseline': 'Retail and Recreation'
	}


fips_lst = ['06059', '36047', '17031', '13121']

pca_mods = ['AdaBoostRegressor - Predictions 2.joblib',
			'KNeighborsRegressor - Predictions 9.joblib',
			'Lasso - Predictions 2.joblib',
			'RandomForestRegressor - Predictions 2.joblib']
nopca_mods = ['AdaBoostRegressor - Predictions 2.joblib',
			  'Lasso - Predictions 1.joblib',
			  'LinearRegression - Predictions 0.joblib',
			  'RandomForestRegressor - Predictions 2.joblib']

def load_df(target, df_type='Validation', pca=False):
	"""
	loads validation or target data from output/data folder
	inputs:
		target: list of target variables
		df_type: string ("Valication" or "Test")
		pca: bool - if you want to load pca or nopca files
	"""

	# print(df_type)
	assert df_type in ('Validation', 'Test'), "df_type must be Validation or Test"
	assert isinstance(target, list), 'target must be a list of strings'

	print("loading df...")
	pca = ' PCA' if pca else ''

	feats = jl.load('../output/data/Data - {} Features{}.joblib'.format(
		df_type, pca))
	targets = jl.load('../output/data/Data - {} Target.joblib'.format(df_type))

	feats.drop(['date', 'fips'], axis=1, inplace=True, errors='ignore')

	# return feats

	for df in [feats, targets]:
		df.drop(['StateName', 'CountyName'], axis=1, inplace=True, errors='ignore')

	# return feats

	print('feat type', type(feats))
	print('targets type', type(targets))

	df = pd.concat([feats, targets], axis=1).reset_index().drop('index', axis=1)
	# df = feats.merge(targets, how='left', on=['fips', 'date'])

	# return df



	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	df = df[['fips', 'date'] + target]
	names = {c: 'observed_' + c for c in target}
	df.rename(names, axis=1, inplace=True, errors='raise')

	# val = val_feat.merge(val_t, how='inner', ldfeft_index=True, right_index=True)
	return df


def load_predictions(folder, model_list, target):
	"""
	loads predictions for specifified models
	this function assumes files are named as follows:
	AdaBoostRegressor - Predictions 4.joblib
	KNeighborsRegressor - Predictions 5.joblib
	LinearSVR - Predictions 0.joblib
	inputs:
		model_list = [list of files in folder that correspond to model predictionss]
		target = [list of target variables]
	"""

	print("loading predictions")
	predictions = {}

	for f in model_list:
	    preds = pd.Series(jl.load(join(OUTPUT, folder, f)))
	    predictions[f[:f.find('.')-2]] = preds

	df = pd.DataFrame(predictions)

	for t in target:
		names = {c: '_'.join([c, t]) for c in predictions}

	df.rename(names, axis=1, inplace=True, errors='raise')

	return df
# if __name__ == '__main__':

# 	print(DATA)

def create_prediction_df(target, model_list, df_type='Validation', pca=False):


	assert df_type in ('Validation', 'Test'), "df_type must be Validation or Test"
	assert isinstance(target, list), 'target must be a list of strings'

	folder = 'models_predictions_pca' if pca else 'models_predictions_nopca'

	observed = load_df(target, df_type=df_type, pca=pca)
	predictions = load_predictions(folder, model_list, target)

	print("merging...")
	merged = observed.merge(predictions, how='inner', left_index=True, right_index=True)

	return merged
		

def load_training_county_mean(target):
	"""
	target is a list of target variables
	"""
	training_df = jl.load(join(DATA, 'Data - Train Target.joblib'))

	df = pd.DataFrame(training_df['fips'].unique())
	df.columns = ['fips']

	# return df
	for t in target:
		means = training_df.groupby('fips')[t].mean()
		df = df.merge(means, how='left', on='fips').rename(
			{t: t+'_cmean'}, axis=1)

	return df


def calc_MAE_by_county(targets, training_means, predictions):
	"""
	target is a list of target variables
	predictions: df that includes date, fips and prediction columns only
	"""

	for t in targets:
	
		df = predictions.merge(training_means, how='left', on='fips').rename(
			{t: t+'_cmean'},
			axis=1)
		df[t + '_error'] = df.apply(lambda x: abs(x['observed_'+t] - x[t+'_cmean']), axis=1)
	
	# return df

	# cols = [t+'_error' for t in targets] + [t+'_cmean' for t in targets]
	# df = df.groupby('fips')[[t+'_error' for t in targets]].mean().reset_index()

	return df


def plot_predictions(df, null_models, fips_list, outpath, df_type='Validation', pca=False):
	"""
	creates by-county line plots of all predictions in df
	"""

	for fips in fips_list:
		county = df[df['fips'] == fips].drop('fips', axis=1)
		# print('null type', type(null))

		for target in [c for c in df.columns if c.startswith('observed')]:
			null = null_models.loc[null_models['fips']==fips, target[9:]+'_cmean'].item()

			#### add the training mean
			for_plot = county[['date'] + [c for c in county.columns if c.endswith(target[9:])]]
			for_plot['date'] = for_plot['date'].astype('O')
			plt.clf()

			fig = plt.figure(figsize=(15, 10))
			ax = fig.add_subplot(111)
			
			ax.hlines(null, for_plot['date'].min(), for_plot['date'].max(),
				color='grey', linestyle='dashed', linewidth=3, label='Null Model')
			ax.plot(for_plot['date'], for_plot[target], color='black', linewidth=4, label='Observed')

			for c in for_plot.columns:
				if 'Predictions' in c:
					label = c[:c.find('Predictions_') - 2]
					ax.plot(for_plot['date'], for_plot[c], linewidth=2, label=label)

			plt.title(
			    'Predictions and Observations for Select Models on Validation Data\nFIPS: {}\nVariable: {}'.format(
			    	fips, LINE_PLOT_LABELS[target]),
			    fontsize=20)
			plt.xlabel('Date', fontsize=12)
			plt.ylabel('Change in Mobility From from Baseline', fontsize=12)
			ax.legend(loc='best')
			ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

			pca_title = 'pca' if pca else 'nopca'
			name = '_'.join([target, fips, df_type, pca_title])




def plot_model_results(results, targets, county_means, outpath, pca=False):
	"""
	creates swarmplots in output/plots
	"""


	for t in targets:
		plt.clf()

		f = plt.figure(figsize=(15, 5))
		sns.swarmplot(x='Model', y='MAE', data=results)
		plt.title('Mean Absolute Error by Model on Validation Data', fontsize=20)
		plt.xlabel('Model', fontsize=14)
		plt.ylabel('Mean Absolute Error', fontsize=14)
		# print(county_means.columns)
		plt.hlines(y=county_means[t + '_error'].mean(), xmin=-1, xmax=results.Model.nunique(),
			linestyle='dashed', linewidth=3, label='County Mean', color='grey')
		plt.text(4.2, county_means[t + '_error'].mean()-.8, 'County Mean',
			fontsize=12, color='grey')

		pca_name = 'pca' if pca else 'nopca'

		plt.show()
		plt.savefig(join(outpath, t + '_MAE_by_model_' + pca_name + '.png'))
		plt.close()


def execute_plots(targets, fips_list):

	model_list = {True: pca_mods, False: nopca_mods}

	null_model = load_training_county_mean(targets)
	outpath = join(OUTPUT, 'plot')

	training_means = load_training_county_mean(targets)

	for p in (True, False):

		pred_df = create_prediction_df(targets, model_list[p], df_type='Validation', pca=p)
		plot_predictions(pred_df, null_model, fips_list, outpath, df_type='Validation', pca=p)

		county_means = calc_MAE_by_county(targets, training_means, pred_df)

		if p:
			results = pd.read_csv(join(OUTPUT, 'model_validation_results_with_pca.csv'))
		else:
			results = pd.read_csv(join(OUTPUT, 'model_validation_results_without_pca.csv'))

		plot_model_results(results, targets, county_means, outpath, pca=p)



def create_county_MAE():


	### first load predictions for best model
	### replace second argument with best performing model
	preds_pca = create_prediction_df(['retail_and_recreation_percent_change_from_baseline'],
									 ['RandomForestRegressor - Predictions 4.joblib'], pca=True)

	preds_pca['MAE'] = preds_pca.apply(
		lambda x: abs(x['observed_retail_and_recreation_percent_change_from_baseline'] -
					  x['RandomForestRegressor - Predictions_retail_and_recreation_percent_change_from_baseline']),
		axis=1)
	preds_pca = preds_pca.groupby('fips').mean()

	preds_nopca = create_prediction_df(['retail_and_recreation_percent_change_from_baseline'],
									 ['RandomForestRegressor - Predictions 2.joblib'], pca=False)
	preds_nopca['MAE'] = preds_nopca.apply(
		lambda x: abs(x['observed_retail_and_recreation_percent_change_from_baseline'] -
					  x['RandomForestRegressor - Predictions_retail_and_recreation_percent_change_from_baseline']),
		axis=1)
	preds_nopca = preds_nopca.groupby('fips').mean()

	return preds_pca, preds_nopca



	