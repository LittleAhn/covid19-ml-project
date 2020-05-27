import pandas as pd
import read_file
import import_health


def build_df():

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
		if c.startswith('Percent') and c[-5:] != 'state':
			df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), c + '_state']

	### drops extra cols
	print('dropping columns...')
	cols = [c for c in df.columns if c[-5:] == 'state' or c[:4] in ('CBSA', 'FIPS')]
	df.drop(columns=cols + ['NAME', 'state', 'county'], axis=1, inplace=True)

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

	### interventions
	print('reading interventions...')
	interventions = read_file.read_interventions()
	print('merging interventions...')
	df = df.merge(interventions, on='fips', how='left')
	# target['in_NAICS'] = target['fips'].isin(NAICS['fips'])
	# NAICS['in_target'] = NAICS['fips'].isin(target['fips'])
	df.drop(['STATE', 'AREA_NAME', 'county', 'state'], axis=1, inplace=True, errors='raise')


	# df.rename({'_merge': 'fips_match'})




			# print(df[df[c].isnull()])
			# df.loc[df[c].isnull(), c] = \
			# 	df.loc[df[c].isnull(), 'StateFIPS'].apply(
			# 	lambda x: import_health.interpolate_nulls(x, c, health_st))




	# df = df.merge(health_st, how='left', on='StateName')

	### interpolating county health with state health
	# cols = [c for c in df.colums if c.startswith('Percent')]
	# df[df['fips_match'] == 'left_only', cols] = df[df['fips_match'] == 'left_only', cols]
	# df.merge



	return df





