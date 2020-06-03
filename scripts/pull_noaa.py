import pandas as pd
import numpy as np
import geopandas as gpd


def main():

	print('reading...')
	df = read_temps()
	print('fixing date...')
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
	print("filtering and reformatting...")
	df = filter_and_reshape(df)

	print('reading stations...')
	station_data = read_stations()
	df = df.merge(station_data, how='left', on='id')

	print('coverting to gdf...')
	df = gpd.GeoDataFrame(df, crs={'init': 'epsg:4269'},
		geometry=gpd.points_from_xy(x=df['lon'], y=df['lat']))	

	print('merging on counties...')
	df = merge_counties(df)

	print('aggregating by column...')
	df = df.groupby(['fips', 'date'])['TMAX', 'TMIN', 'PRCP'].mean().reset_index()
	df['state'] = df['fips'].str[:2]

	print("convert temperatures to farenheit")
	for c in ['TMAX', 'TMIN']:
		df.loc[:,c] = df[c].apply(lambda x: (x/10) * (5/9) + 32)

	print('merging on areas..')
	counties = read_shape()
	df = df.merge(counties[['fips', 'area']], how='left', on='fips')

	print('interpolate...')
	for c in ['TMAX', 'TMIN', 'PRCP']:
		vals = df.groupby('state')[c].transform(np.mean)
		df[c].fillna(vals, inplace=True)

	print('creating features...')
	df = create_features(df)

	# return df

	# df.drop('interpolate_value', axis=1, inplace=True, errors='raise')

	df.to_csv('../data_intermediate/noaa.csv')



	# df.to_csv('../data_intermediate/weather.csv', index=False)
	return df


def create_features(df):

	df['precip_dummy'] = 0
	df.loc[df['PRCP'] > .05, 'precip_dummy'] = 1 ### arbitrary cutoff

	### rolling avg
	df.set_index('date', inplace=True)
	df.sort_index(inplace=True)
	for var in ['TMIN', 'TMAX']:
		for window in [3, 5, 7, 10]:
			df['{}_{}d_avg'.format(var, window)] = df.groupby('fips')[var].transform(
				lambda x: x.rolling(window, 1).mean())

	return df


def merge_counties(df):

	counties = read_shape()
	counties = counties[['GEOID', 'geometry', 'area']]
	counties.rename({'GEOID': 'fips'}, axis=1, inplace=True, errors='raise')
	df = gpd.sjoin(df, counties, how='inner', op='intersects')
	return df.reset_index()


def filter_and_reshape(df):

	df = df[['id', 'date', 'var', 'value']]
	df = df[df['date'] >= '2020-02-15']
	df = df[df['var'].isin(['TMAX', 'TMIN', 'PRCP'])]
	df = df.set_index(['id', 'date', 'var'])
	df = df.unstack(level=-1)
	df = df.reset_index()
	df.columns = ['id', 'date', 'PRCP', 'TMAX', 'TMIN']

	return df


def read_temps():

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
		lats.append(float(line[12:20]))
		lons.append(float(line[21:30]))
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
	geodf['area'] = geodf.geometry.apply(lambda x: x.area)
	geodf['fips'] = geodf['STATEFP'] + geodf['COUNTYFP']
	return geodf



