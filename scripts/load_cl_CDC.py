import pandas as pd
import requests as re
from os.path import join, abspath, dirname, exists
import utils

ROOT = join(dirname(dirname(abspath(__file__))), 'data')
RAW = join(ROOT, 'raw')
CL = join(ROOT, 'clean_target')
FEAT = join(ROOT, 'clean_features')



def read_CDC():

	cases = pd.read_csv(join(RAW, 'cdc_covid_confirmed_usafacts.csv'))
	deaths = pd.read_csv(join(RAW, 'cdc_covid_deaths_usafacts.csv'))

	return cases, deaths


def cl(df, val_name):

	df = df[df['countyFIPS'] != 0]
	df.drop(['County Name', 'State', 'stateFIPS'], axis=1, inplace=True)
	df = df.melt(id_vars='countyFIPS', var_name='date', value_name=val_name)

	return df


def main():

	cases, deaths = read_CDC()
	cases = cl(cases, 'cases')
	deaths = cl(deaths, 'deaths')

	cases.to_csv(join(FEAT, 'CDC_cases.csv'), index=False)
	deaths.to_csv(join(FEAT, 'CDC_deaths.csv'), index=False)