
import numpy as np
import pandas as pd
import read_file
import import_health
import datetime

def build_df():
	"""
<<<<<<< HEAD
	assembles all components into 1 large data frame
=======
	Assembles all components into 1 large dataframe
>>>>>>> 9e61762de9eb096f9510670f6b9272e573a088ad
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
	print('\tInterpolating missing CDC county data with state data...')
	for c in df.columns:
		if c.startswith('Percent') and not c.endswith('state'):
			df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), c + '_state']
	print('\tDropping extra CDC columns...')
	cols = [c for c in df.columns if c.endswith('_state') or c[:4] in ('CBSA', 'FIPS')]
	df.drop(columns=cols + ['NAME', 'county'], axis=1, inplace=True)

	### CDC case / death data
	print('Reading/merging CDC cases and death data...')
	cases,deaths = read_file.read_CDC()
	df = df.merge(cases, how='left', on=['date', 'fips'])
	df = df.merge(deaths, how='left', on=['date', 'fips'])

	### kaggle extra vars
	print('reading kaggle data...')
	weather = read_file.read_kaggle()
	print('merging kaggle data...')
	df = df.merge(weather, how='left', on=['fips', 'date'])
	print('interpolating null with fips mean for density')
	df['area_sqmi'] = df.groupby('fips')['area_sqmi'].transform(np.mean)

	### noaa
	print('reading noaa weather...')
	noaa = read_file.read_noaa()
	print('merging noaa data...')
	df = df.merge(noaa, how='left', on=['fips', 'date'])


	### interventions
	print('reading interventions...')
	interventions = read_file.read_interventions()
	print('merging interventions...')
	df = df.merge(interventions, on='fips', how='left')
	df.drop(['STATE', 'AREA_NAME', 'state', 'StateFIPS'], axis=1, inplace=True, errors='raise')
	print('transforming intervention columns...')
	for c in df.columns:
		if c.startswith("int_date_"):
			print('transforming {}...'.format(c))
			df[c].fillna(800000, inplace=True) ### arbitrary high date
			df[c] = df[c].apply(lambda x: datetime.date.fromordinal(int(x)))
			df[c] = df.apply(lambda x: x[c] <= x['date'], axis=1).astype('int')
			# df['int_' + c] = 0
			# df.loc[df[c] >= df['date'], 'int_' + c] = 1			

	return df
	## making additional features
	df = make_features(df)

	df.drop([c for c in df.columns if c.startswith('lag')],
		axis=1, inplace=True, errors='raise')

	df.drop(['state_x', 'state_y', 'CountyFIPS'], axis=1, inplace=True, errors='raise')

	return df


def make_features(df):

	df['pop_density'] = df['pop'] / df['area']
	df['cases_per_pop'] = df['cases'] / df['pop']
	df['cases_per_area'] = df['cases'] / df['area']
	df['deaths_per_pop'] = df['deaths'] / df['pop']
	df['deaths_per_area'] = df['deaths'] / df['area']

	return df


