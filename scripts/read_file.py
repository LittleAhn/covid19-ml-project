"""
Just some functions that we can use for easiliy loading data into our
jupyter notebooks without referencing all these folders
"""
import pandas as pd
from os.path import join, dirname, abspath
import utils

ROOT = join(dirname(dirname(abspath(__file__))))
INT = join(ROOT, 'data_intermediate')
RAW = join(ROOT, 'data_raw')



def read_target():
	"""
	loads google mobility data merged with FIPS
	"""

	df = pd.read_csv(join(INT, 'us_mobility.csv'))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	# df['CountyFIPS'] = df['CountyFIPS'].apply(lambda x: utils.prepend_0s(str(x), 3))
	df['StateFIPS'] = df['StateFIPS'].apply(lambda x: utils.prepend_0s(str(x), 2))
	# df['fips'] = df['StateFIPS'] + df['CountyFIPS']
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

	return df


def read_ACS():

	df = pd.read_csv(join(INT, "ACS.csv"))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))

	return df


def read_NAICS():
	"""
	loads NAICS file
	"""
	df = pd.read_csv(join(INT, "NAICS.csv"))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))

	return df


def read_noaa():
	"""
	load NOAA data
	"""
	df = pd.read_csv(join(INT, 'noaa.csv'))
	df['fips'] = df['fips'].apply(
		lambda x: utils.prepend_0s(str(x), 5))	
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

	return df

def read_CDC():

	cases = pd.read_csv(join(INT, "CDC_cases.csv"))
	deaths = pd.read_csv(join(INT, "CDC_deaths.csv"))

	for df in (cases, deaths):
		df['fips'] = df['countyFIPS'].apply(
			lambda x: utils.prepend_0s(str(x), 5))
		df.drop('countyFIPS', axis=1, inplace=True)
		df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')

	# cases['fips'] = cases['countyFIPS'].apply(
	# 	lambda x: utils.prepend_0s(str(x), 5))
	# deaths['fips'] = deaths['countyFIPS'].apply(
	# 	lambda x: utils.prepend_0s(str(x), 5))
	
	return cases, deaths


def read_votes():

	df = pd.read_csv(join(INT, 'votes.csv'))
	df['fips'] = df['fips'].apply(
		lambda x: utils.prepend_0s(str(x)[:str(x).find('.')], 5))

	return df


def read_health():

	health_fips = pd.read_csv(join(INT, 'health_fips.csv'))
	# health_fips.rename({'FIPS_ID': 'fips'}, axis=1, inplace=True, errors='raise')
	health_fips['fips'] = health_fips['FIPS_ID'].apply(lambda x: utils.prepend_0s(str(x), 5))

	health_fips.loc[health_fips['fips'] == '15901', 'fips'] = '15009'
	health_fips.loc[health_fips['fips'] == '51918', 'fips'] = '51570'
	health_fips.loc[health_fips['fips'] == '51918', 'fips'] = '51730'
	health_fips.loc[health_fips['fips'] == '51931', 'fips'] = '51830'
	health_fips.loc[health_fips['fips'] == '51941', 'fips'] = '51670'
	health_fips.loc[health_fips['fips'] == '51949', 'fips'] = '51620'
	health_fips.loc[health_fips['fips'] == '51953', 'fips'] = '51520'
	health_fips.loc[health_fips['fips'] == '51958', 'fips'] = '51735'

	health_st = pd.read_csv(join(INT, 'healthdf_st.csv'))
	health_st['StateFIPS'] = health_st['StateFIPS'].apply(
		lambda x: utils.prepend_0s(str(x), 2))

	cols = [c for c in health_st.columns if c.startswith('Percent') or c == 'StateFIPS']
	health_st = health_st[cols]

	name_change = {c: c + '_state' for c in health_st.columns if c.startswith('Percent')}
	health_st.rename(name_change, axis=1, inplace=True, errors='raise')

	return health_fips, health_st


def read_kaggle():

	df = pd.read_csv(join(INT, 'cl_kaggle.csv'))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

	return df

def read_interventions():

	# file = 'https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/data/interventions.csv'
	# df = pd.read_csv(file)

	df = pd.read_csv(join(INT, 'interventions.csv'))

	df.rename({'FIPS': 'fips'}, axis=1, inplace=True)
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))

	return df





# def read_NYT():
# 	"""
# 	always pulls most recent NYT data
# 	"""
# 	new_york_counties = {
# 				'Bronx': 1432132 / 8398748,
# 				'Kings': 2582830 / 8398748,
# 				'New York': 1628701 / 8398748,
# 				'Queens': 2278906 / 8398748,
# 				'Richmond': 476179 / 8398748
# 						}


# 	URL = r'https://raw.githubusercontent.com/nytimes/covid-19-data/c95c96f7793ed6aa0b2fa9e1400cdd587d62bcf7/us-counties.csv'
# 	df = pd.read_csv(URL)
# 	df['fips'] = df['fips'].astype(int)#.apply(lambda x: utils.prepend_0s(str(x), 5))

# 	return df