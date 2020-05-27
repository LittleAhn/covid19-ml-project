import pandas as pd
import geopandas as gpd
import pull_noaa
import read_file
import load_weather
import load_interventions


def noaa():
	noaa = pull_noaa.main()
	noaa = noaa[['date', 'PRCP', 'TMAX', 'TMIN', 'lat', 'lon']]
	noaa.lat = pd.to_numeric(noaa.lat)
	noaa.lon = pd.to_numeric(noaa.lon)
	noaa = gpd.GeoDataFrame(noaa, crs={'init': 'epsg:4269'}, geometry=gpd.points_from_xy(x=noaa.lon, y=noaa.lat))
	return noaa


def shape():
	shape = gpd.read_file('../data_raw/tl_2017_us_county/tl_2017_us_county.shp')
	shape = shape[['NAMELSAD', 'geometry']]
	shape.NAMELSAD = shape.NAMELSAD.str.upper()
	return shape


def weather():
	noaa = noaa.fillna(noaa.mean())
	weather = gpd.sjoin(noaa, shape(), how='inner', op='intersects')
	return weather



