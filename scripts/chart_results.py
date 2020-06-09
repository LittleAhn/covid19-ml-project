### PLOTTING FUNCTIONS ONLY SUPPORT 1 TARGET VARIABLE PER RUNs

import pandas as pd
import numpy as np
import joblib as jl
import seaborn as sns
from matplotlib import pyplot as plt
from os import listdir
from os.path import join
import pipeline
import utils
import fit_models_without_pca
import fit_models_with_pca

OUTPUT = "../output"
DATA = "../output/data"

LINE_PLOT_LABELS = {
	'observed_retail_and_recreation_percent_change_from_baseline': 'Retail and Recreation'
	

}


def load_df(target, df_type='Validation', pca=False):
	"""
	loads validation or target data from output/data folder
	inputs:
		target: list of target variables
		df_type: string ("Valication" or "Test")
		pca: bool - if you want to load pca or nopca files
	"""

	print(df_type)
	assert df_type in ('Validation', 'Test'), "df_type must be Validation or Test"
	assert isinstance(target, list), 'target must be a list of strings'

	print("loading df...")
	pca = ' PCA' if pca else ''

	feats = jl.load('../output/data/Data - {} Features{}.joblib'.format(
		df_type, pca))
	targets = jl.load('../output/data/Data - {} Target.joblib'.format(df_type))

	print('feat type', type(feats))
	print('targets type', type(targets))

	df = pd.concat([feats, targets], axis=1).reset_index().drop('index', axis=1)

	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	df = df[['fips', 'date'] + target]
	names = {c: 'observed_' + c for c in target}
	df.rename(names, axis=1, inplace=True, errors='raise')

	# val = val_feat.merge(val_t, how='inner', left_index=True, right_index=True)
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
		


def plot_predictions(df, fips_list, outpath, df_type='Validation', pca=False):
	"""
	creates by-county line plots of all predictions in df
	"""

	for fips in fips_list:
		county = df[df['fips'] == fips].drop('fips', axis=1)

		for target in [c for c in df.columns if c.startswith('observed')]:

			for_plot = county[['date'] + [c for c in county.columns if c.endswith(target[9:])]]
			# print([c for c in county.columns if c.endswith(target[9:])])
			# for_plot = pd.melt(for_plot, 'date', var_name='Model', value_name='Value')

			# for_plot['widths'] = 1
			# for_plot.loc[for_plot['Model'].str.startswith('observed'), 'widths'] = 5

			plt.clf()
			# fig, ax = plt.subplots()

			fig = plt.figure(figsize=(15, 10))
			
			ax = fig.add_subplot(111)
			for_plot[target].plot(ax=ax, color='black', linewidth=4, label='Observed')
			for c in for_plot.columns:
				if 'Predictions' in c:
					label = c[:c.find('Predictions_')-2]
					for_plot[c].plot(ax=ax, linewidth=2, label=label)



			# sns.lineplot('date', 'Value', hue='Model', size='widths', data=for_plot)
			# for_plot[target].plot(linewidth=3)
			plt.title(
			    'Predictions and Observations for Select Models on Validation Data\nFIPS: {}\nVariable: {}'.format(
			    	fips, LINE_PLOT_LABELS[target]),
			    fontsize=20)
			plt.xlabel('Date', fontsize=12)
			plt.ylabel('Change in Mobility From from Baseline', fontsize=12)
			ax.legend(loc='best')

			pca = 'pca' if pca else 'nopca'
			name = '_'.join([target, fips, df_type, pca])

			plt.show()
			plt.savefig(join(outpath, name + '.png'))
			plt.close()


# def plot_predictions(df, fips_list, outpath, df_type='Validation', pca=False):
# 	"""
# 	creates by-county line plots of all predictions in df
# 	"""



# 	for fips in fips_list:
# 		county = df[df['fips'] == fips].drop('fips', axis=1)

# 		for target in [c for c in df.columns if c.startswith('observed')]:

# 			for_plot = county[['date'] + [c for c in county.columns if c.endswith(target[9:])]]
# 			# print([c for c in county.columns if c.endswith(target[9:])])
# 			for_plot = pd.melt(for_plot, 'date', var_name='Model', value_name='Value')

# 			for_plot['widths'] = 1
# 			for_plot.loc[for_plot['Model'].str.startswith('observed'), 'widths'] = 5

# 			plt.clf()
# 			fig, ax = plt.subplots()
# 			plt.figure(figsize=(15, 10))
# 			sns.lineplot('date', 'Value', hue='Model', size='widths', data=for_plot)
# 			# for_plot[target].plot(linewidth=3)
# 			plt.title(
# 			    'Predictions and Observations for Select Models on Validation Data\nFIPS: {}\nVariable: {}'.format(
# 			    	fips, target),
# 			    fontsize=24)
# 			plt.xlabel('Date', fontsize=14)
# 			plt.ylabel('Change in Mobility From from Baseline', fontsize=14)

# 			pca = 'pca' if pca else 'nopca'
# 			name = '_'.join([target, fips, df_type, pca])

# 			plt.show()
# 			plt.savefig(join(outpath, name + '.png'))
# 			plt.close()



def load_training_data():
	pass

	


def calc_null_scores():
	"""
	Calculates MAE and MSE for 
	"""

	### get training data

	### calc means

	### append to predictions




	