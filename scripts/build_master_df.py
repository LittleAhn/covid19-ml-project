
import numpy as np
import pandas as pd
import read_file
import import_health
import datetime

def build_df():
	"""
	Assembles all components into 1 large dataframe
	"""

	### Target variable
	print('Reading Google mobility data...')
	df = read_file.read_target()

	### NAICS data
	print('Reading / merging NAICS business pattern data...')
	NAICS = read_file.read_NAICS()
	df = df.merge(NAICS, how='left', on='fips')
	
	### Census data
	print('Reading / merging ACS census data...')
	ACS = read_file.read_ACS()
	df = df.merge(ACS, how='left', on='fips')

	### CDC general health data
	print('Reading / merging CDC health data...')
	health_fips,health_st = read_file.read_health()
	df = df.merge(health_fips, how='left', on='fips')
	df = df.merge(health_st, how='left', on='StateFIPS')
	print('    Interpolating missing CDC county data with state data...')
	for c in df.columns:
		if c.startswith('Percent') and not c.endswith('state'):
			df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), c + '_state']
	print('    Dropping extra CDC columns...')
	cols = [c for c in df.columns if c.endswith('_state') or c[:4] in ('CBSA', 'FIPS')]
	df.drop(columns=cols + ['NAME', 'county'], axis=1, inplace=True)

	### CDC case / death data
	print('Reading / merging CDC cases and death data...')
	cases,deaths = read_file.read_CDC()
	df = df.merge(cases, how='left', on=['date', 'fips'])
	df = df.merge(deaths, how='left', on=['date', 'fips'])

	### Kaggle extra vars
	### Right now we're only using stay_at_home from this df
	#print('Reading / merging Kaggle data...')
	#kaggle = read_file.read_kaggle()
	#df = df.merge(kaggle, how='left', on=['fips', 'date'])

	### NOAA
	print('Reading / merging NOAA weather data...')
	noaa = read_file.read_noaa()
	df = df.merge(noaa, how='left', on=['fips', 'date'])
	print('\tInterpolating missing weather data')
	for c in df.columns:
		if c.startswith('TM') or c == 'PRCP':
			print("\tInterpolating {}...".format(c))
			vals = df.groupby(['StateFIPS', 'date'])[c].transform(np.mean)
			df[c].fillna(vals, inplace=True)
	print('\tCreating precipiation dummy...')
	df['precip_dummy'] = 0
	df.loc[df['PRCP'] > .05, 'precip_dummy'] = 1 ### cutoff is 1000% arbitrary
 
	### Interventions
	print('Reading interventions data...')
	interventions = read_file.read_interventions()
	df = df.merge(interventions, on='fips', how='left')
	df.drop(['STATE', 'AREA_NAME', 'StateFIPS'], axis=1, inplace=True, errors='raise')
	print('\tTransforming intervention columns...')
	for c in df.columns:
		if c.startswith("int_date_"):
			print('\tTransforming {}...'.format(c))
			df[c].fillna(800000, inplace=True) ### arbitrary high date
			df[c] = df[c].apply(lambda x: datetime.date.fromordinal(int(x)))
			df[c] = df.apply(lambda x: x[c] <= x['date'], axis=1).astype('int')

	### Vote share
	print('Reading vote share data...')
	votes = read_file.read_votes()
	df = df.merge(votes, how='left', on='fips')

	## Making additional features
	df = make_features(df)

	# Drop excess columns
	df.drop([c for c in df.columns if c.startswith('lag')],
		axis=1, inplace=True, errors='raise')
	df.drop(columns=['state_x','state_y','CountyFIPS',
					 'totalvotes','area'], inplace=True)
	
	print('Outputting csv..')
	df.to_csv('../output/full_df.csv', index=False)

	return df


def make_features(df):
	df['pop_density'] = df['pop'] / df['area']
	df['cases_per_pop'] = df['cases'] / df['pop']
	#df['cases_per_area'] = df['cases'] / df['area']
	df['deaths_per_pop'] = df['deaths'] / df['pop']
	#df['deaths_per_area'] = df['deaths'] / df['area']
	return df