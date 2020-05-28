import pandas as pd
import read_file
import import_health


def build_df():
	"""
	assembles all components into 1 large dataframe
	"""

	### target var as spine
	print('reading google mobility...')
	df = read_file.read_target()

	### business patterns
	print('reading NAICS...')
	NAICS = read_file.read_NAICS()
	print('merging NAICS...')
	df = df.merge(NAICS, how='left', on='fips')
	
	### census
	print('reading ACS...')
	ACS = read_file.read_ACS()
	print('merging ACS...')
	df = df.merge(ACS, how='left', on='fips')

	### cdc wonderdata
	print('reading cdc wonderdata...')
	health_fips, health_st = read_file.read_health()
	print('merging wonderdata...')
	df = df.merge(health_fips, how='left', on='fips')
	df = df.merge(health_st, how='left', on='StateFIPS')
	print('interpolating wonderdata counties with state...')
	for c in df.columns:
		if c.startswith('Percent') and not c.endswith('state'):
			df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), c + '_state']

	### drops extra cols
	print('dropping columns...')
	cols = [c for c in df.columns if c.endswith('_state') or c[:4] in ('CBSA', 'FIPS')]
	df.drop(columns=cols + ['NAME', 'county'], axis=1, inplace=True)

	### cdc cases, deaths
	print('reading cdc cases and deaths...')
	cases, deaths = read_file.read_CDC()
	print('merging cdc cases and deaths...')
	df = df.merge(cases, how='left', on=['date', 'fips'])
	df = df.merge(deaths, how='left', on=['date', 'fips'])

	### weather + kaggle
	print('reading kaggle weather+ data...')
	weather = read_file.read_kaggle()
	print('merging kaggle weather+ data...')
	df = df.merge(weather, how='left', on=['fips', 'date'])

	### noaa
	print('reading noaa weather...')
	noaa = read_file.read_noaa()
	print('merging noaa data...')
	df = df.merge(noaa, how='left', on=['fips', 'date'], indicator=True)

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


