"""
Just some functions that we can use for easiliy loading data into our
jupyter notebooks without referencing all these folders
"""
import pandas as pd
from os.path import join, dirname, abspath
import utils

ROOT = join(dirname(dirname(dirname(abspath(__file__)))), 'data')
CL = join(ROOT, 'clean_target')
FEAT = join(ROOT, 'clean_features')



def read_target():
	"""
	loads google mobility data merged with FIPS
	"""

	df = pd.read_csv(join(CL, 'us_mobility.csv'))
	df['CountyFIPS'] = df['CountyFIPS'].apply(lambda x: utils.prepend_0s(str(x), 3))
	df['StateFIPS'] = df['StateFIPS'].apply(lambda x: utils.prepend_0s(str(x), 2))

	return df


def read_NAICS():
	"""
	loads NAICS file
	"""
	df = pd.read_csv(join(FEAT, "NAICS.csv"))
	df['FIPS'] = df['FIPS'].apply(lambda x: utils.prepend_0s(str(x), 5))

	return df


def read_NYT():
	"""
	always pulls most recent NYT data
	"""
	new_york_counties = {
				'Bronx': 1432132 / 8398748,
				'Kings': 2582830 / 8398748,
				'New York': 1628701 / 8398748,
				'Queens': 2278906 / 8398748,
				'Richmond': 476179 / 8398748
						}


	URL = r'https://raw.githubusercontent.com/nytimes/covid-19-data/c95c96f7793ed6aa0b2fa9e1400cdd587d62bcf7/us-counties.csv'
	df = pd.read_csv(URL)
	df['fips'] = df['fips'].astype(int)#.apply(lambda x: utils.prepend_0s(str(x), 5))

	return df