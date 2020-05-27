import pandas as pd
from os.path import join, abspath, dirname

ROOT = join(dirname(dirname(abspath(__file__))))
RAW = join(ROOT, 'data_raw')
INT = join(ROOT, 'data_intermediate')

def read_weather():

	weather = pd.read_csv(join(RAW, 'Weather - US Counties.csv'))
	weather[['date', 'county', 'date']]
	return weather

