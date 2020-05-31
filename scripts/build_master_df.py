import pandas as pd
import read_file
import import_health


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

	### weather + kaggle
	print('reading kaggle weather+ data...')
	weather = read_file.read_kaggle()
	print('merging kaggle weather+ data...')
	df = df.merge(weather, how='left', on=['fips', 'date'])

	### noaa
	#print('reading noaa weather...')
	#noaa = read_file.read_noaa()
	#print('merging noaa data...')
	#df = df.merge(noaa, how='left', on=['fips', 'date'], indicator=True)

	### interventions
	print('reading interventions...')
	interventions = read_file.read_interventions()
	print('merging interventions...')
	df = df.merge(interventions, on='fips', how='left')
	df.drop(['STATE', 'AREA_NAME', 'county', 'state'], axis=1, inplace=True, errors='raise')

	### making additional features
	df = make_features(df)

	return df


def make_features(df):

	df['density'] = df['pop'] / df['area_sqmi']

	return df


