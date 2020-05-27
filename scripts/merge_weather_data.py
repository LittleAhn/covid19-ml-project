import pandas as pd
import geopandas as gpd
import pull_noaa
import read_file
import load_weather
import load_interventions


def weather():
	noaa = pull_noaa.main()
	noaa = noaa[['date', 'PRCP', 'TMAX', 'TMIN', 'lat', 'lon']]
	interventions = load_interventions.read_file()
	interventions = interventions[['FIPS', 'AREA_NAME']]
	mobility = pd.read_csv('../data_intermediate/us_mobility.csv')
	mobility = mobility[['CountyName', 'fips']]
	shape = gpd.read_file('../data_raw/tl_2017_us_county/tl_2017_us_county.shp')
	shape = shape[['NAMELSAD', 'geometry']]

	interventions.AREA_NAME = interventions.AREA_NAME.str.upper()
	shape.NAMELSAD = shape.NAMELSAD.str.upper()
	noaa.lat = pd.to_numeric(noaa.lat)
	noaa.lon = pd.to_numeric(noaa.lon)

	weather = shape.merge(interventions, how='left', left_on='NAMELSAD', right_on='AREA_NAME')
	weather = weather.merge(mobility, how='left', left_on='NAMELSAD', right_on='CountyName')
	weather = gpd.GeoDataFrame(weather, crs={'init': 'epsg:4326'}, geometry='geometry')
	noaa = gpd.GeoDataFrame(noaa, crs={'init': 'epsg:4326'}, geometry=gpd.points_from_xy(x=noaa.lon, y=noaa.lat))

	weather = gpd.sjoin(weather, noaa, how='left', op='intersects')
	weather = weather.fillna(weather.mean(), inplace=True)

	return weather



