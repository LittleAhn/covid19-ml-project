import pandas as pd
from os.path import join, dirname, abspath
import utils
import datetime

ROOT = join(dirname(dirname(abspath(__file__))))
INT = join(ROOT, 'data_intermediate')
RAW = join(ROOT, 'data_raw')

file = 'https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/data/interventions.csv'

def read_file():

	df = pd.read_csv(file)
	# date = datetime.date.fromordinal
	# for c in df.columns:
	# 	if c not in ['FIPS', 'STATE', 'AREA_NAME']:
	# 		df[c] = df[c].apply(lambda x: datetime.date.fromordinal(float(x)))

	return df

