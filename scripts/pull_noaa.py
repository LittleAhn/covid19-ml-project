import pandas as pd
import geopandas as gpd


def main():

	print('reading...')
	df = read()
	print('fixing date...')
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
	df = df[['id', 'date', 'var', 'value']]
	df = df[df['date'] >= '2020-02-15']
	df = df[df['var'].isin(['TMAX', 'TMIN', 'PRCP'])]
	df = df.set_index(['id', 'date', 'var'])
	df = df.unstack(level=-1)
	df = df.reset_index()
	df.columns = ['id', 'date', 'PRCP', 'TMAX', 'TMIN']

	station_data = read_stations()
	df = df.merge(station_data, how='left', on='id')
	df.to_csv('../data_intermediate/weather.csv', index=False)
	return df


def read():

	df = pd.read_csv('../data_raw/2020.csv', header=None)
	df.columns = (['id', 'date', 'var', 'value',
				  'm_flag', 'q_flag', 's_flag', 'obs_time'])
	return df


def read_stations():

	with open('../data_raw/ghcnd-stations.txt', 'r') as f:
		lines = f.readlines()

	ids = []
	lats = []
	lons = []
	states = []
	names = []
	for line in lines:
		ids.append(line[:11])
		lats.append(line[12:20])
		lons.append(line[21:30])
		states.append(line[38:40])
		names.append(line[41:71])

	data = {'id': ids,
			'lat': lats,
			'lon': lons,
			'state': states,
			'name': names}
	return pd.DataFrame(data)


def read_shape():

	geodf = gpd.read_file('../data_raw/tl_2017_us_county.shp')
	return geodf


def read_mobility():
	df = pd.read_csv('../data_intermediate/us_mobility.csv')
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
