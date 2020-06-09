import geopandas as gpd
import build_master_df
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def graphs_main():
	df = load()
	bar = bars(df)
	line = lines(df)
	matrix = matrixs(df)
	counties_line = counties_lines(df)
	maps = map(df)
	return


def load():
	# df = build_master_df.build_df()
	df = pd.read_csv('full_df.csv')
	return df


def bars(df):
	sns.set(rc={'figure.figsize':(12, 5)})
	Missing_by_Target = df[[c for c in df.columns if c.endswith('baseline')]].isnull().sum() / df.shape[0] * 100
	Missing_by_Target.index = ['Retail & Recreation', 'Grocery & Parmacy', 'Parks', 'Transit', 'Workplace', 'Residential']
	Missing_by_Target = Missing_by_Target.to_frame('Percentage')
	ax = sns.barplot(x=Missing_by_Target.index, y='Percentage', data=Missing_by_Target)
	plt.xlabel("Mobility - Change from Baseline")
	plt.title('Missing Data by Target Variable', fontsize=20)
	plt.savefig('../output/graphs/missing_data/bar.png')
	return


def lines(df):
	df1 = df[df['retail_and_recreation_percent_change_from_baseline'].isnull()].groupby('date')['fips'].count()/ df.groupby('date')['fips'].count() * 100
	df1.plot(linewidth=0.8, label='Retail & Recreation')
	df2 = df[df['grocery_and_pharmacy_percent_change_from_baseline'].isnull()].groupby('date')['fips'].count() / df.groupby('date')['fips'].count() * 100
	df2.plot(linewidth=0.8, label='Grocery & Parmacy')
	df3 = df[df['parks_percent_change_from_baseline'].isnull()].groupby('date')['fips'].count() / df.groupby('date')['fips'].count() * 100
	df3.plot(linewidth=0.8, label='Parks')
	df4 = df[df['transit_stations_percent_change_from_baseline'].isnull()].groupby('date')['fips'].count() / df.groupby('date')['fips'].count() * 100
	df4.plot(linewidth=0.8, label='Transit')
	df5 = df[df['workplaces_percent_change_from_baseline'].isnull()].groupby('date')['fips'].count() / df.groupby('date')['fips'].count() * 100
	df5.plot(linewidth=0.8, label='Workplace')
	df6 = df[df['residential_percent_change_from_baseline'].isnull()].groupby('date')['fips'].count() / df.groupby('date')['fips'].count() * 100
	df6.plot(linewidth=0.8, label='Residential')
	leg = plt.legend()
	plt.xlabel("Dates")
	plt.ylabel("Percentage on Mobility Change")
	plt.title('Missing Data by Mobility Category - Change from Baseline', fontsize=20)
	plt.savefig('../output/graphs/missing_data/line.png')
	return


def matrixs(df):
	df_matrix = df.dropna()
	df_matrix = df_matrix[['TMAX', 'pop_density', 'in_school_pct', 'cases_per_pop', 'voteshare_dem', 'has_broadband_pct', 'med_inc']]
	df_matrix.columns = ['Max Temperature', 'Population Density', 'In School Percentage', 'Cases Per Population',
                     'Democratic Vote Share', 'Has Broadband Percentage', 'Median Inccome']
	sns.set(style="ticks", color_codes=True, font_scale=1.15)
	ax = sns.pairplot(df_matrix, vars=df_matrix.columns)
	ax.savefig('../output/graphs/matrix_predictions_counties/matrix.png')
	return


def map(df):
	shape = gpd.read_file('../data_raw/tl_2017_us_county/tl_2017_us_county.shp')
	shape = shape[['GEOID', 'NAMELSAD', 'geometry']]
	shape.NAMELSAD = shape.NAMELSAD.str.upper()
	df = df[df.iloc[:, 3:9].isnull().any(axis=1)]
	df = df.groupby('CountyName').count()
	df = df.merge(shape, left_on=['CountyName'], right_on='NAMELSAD', how="right")
	df_inter = df.fillna(df.max())
	gdf = gpd.GeoDataFrame(df_inter, geometry=df_inter.geometry)
	fig, ax = plt.subplots(1, figsize=(25, 10))
	gdf.plot(column='fips', cmap='Reds', ax=ax, edgecolor='0.8')
	ax.set_title('Percentage of Missing Mobility Data by County', fontsize=25)
	sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=100))
	sm._A = []
	cbar = fig.colorbar(sm)
	ax.axis((-130, -60, 20, 50))
	plt.xlabel('Longitude', fontsize=15)
	plt.ylabel('Latitude', fontsize=15)
	plt.savefig('../output/graphs/missing_data/map.png')
	return


def counties_lines(df):
	df_line = df[['date', 'fips', 'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline', 'TMAX']]
	df_line.set_index('date', inplace=True)
	df_line.columns = ["fips", "Retail and Recreation", "Grocery and Pharmacy", "Parks", "Transit Stations",
                   "Workplaces", "Residential", "Max Temperature"]
	sns.set(rc={'figure.figsize':(15, 6)})

	df_orange = df_line[df_line['fips'] == 6059]
	df_orange = df_orange[["Retail and Recreation", "Grocery and Pharmacy", "Parks",
                       "Transit Stations", "Workplaces", "Residential", "Max Temperature"]]
	fig, ax = plt.subplots()
	df_orange[["Retail and Recreation"]].plot(ax=ax, linewidth=5)
	df_orange[["Grocery and Pharmacy"]].plot(ax=ax, linewidth=0.8)
	df_orange[["Parks"]].plot(ax=ax, linewidth=0.8)
	df_orange[["Transit Stations"]].plot(ax=ax, linewidth=0.8)
	df_orange[["Workplaces"]].plot(ax=ax, linewidth=0.8)
	df_orange[["Residential"]].plot(ax=ax, linewidth=0.8)
	df_orange[["Max Temperature"]].plot(ax=ax, linewidth=0.8)
	plt.xlabel("Dates", fontsize=13)
	plt.ylabel("Percentage on Mobility Change", fontsize=13)
	plt.title('Mobility Change, Max Temperature and Deaths per Population in Orange County', fontsize=18)
	plt.savefig('../output/graphs/matrix_predictions_counties/orange_line.png')

	df_brooklyn = df_line[df_line['fips'] == 36047]
	df_brooklyn = df_brooklyn[["Retail and Recreation", "Grocery and Pharmacy", "Parks",
	                       "Transit Stations", "Workplaces", "Residential", "Max Temperature"]]
	fig, ax = plt.subplots()
	df_brooklyn[["Retail and Recreation"]].plot(ax=ax, linewidth=5)
	df_brooklyn[["Grocery and Pharmacy"]].plot(ax=ax, linewidth=0.8)
	df_brooklyn[["Parks"]].plot(ax=ax, linewidth=0.8)
	df_brooklyn[["Transit Stations"]].plot(ax=ax, linewidth=0.8)
	df_brooklyn[["Workplaces"]].plot(ax=ax, linewidth=0.8)
	df_brooklyn[["Residential"]].plot(ax=ax, linewidth=0.8)
	df_brooklyn[["Max Temperature"]].plot(ax=ax, linewidth=0.8)
	plt.xlabel("Dates", fontsize=13)
	plt.ylabel("Percentage on Mobility Change", fontsize=13)
	plt.title('Mobility Change, Max Temperature and Deaths per Population in Brooklyn County', fontsize=18)
	plt.savefig('../output/graphs/matrix_predictions_counties/brooklyn_line.png')

	df_cook = df_line[df_line['fips'] == 17031]
	df_cook = df_cook[["Retail and Recreation", "Grocery and Pharmacy", "Parks",
	                       "Transit Stations", "Workplaces", "Residential", "Max Temperature"]]
	fig, ax = plt.subplots()
	df_cook[["Retail and Recreation"]].plot(ax=ax, linewidth=5)
	df_cook[["Grocery and Pharmacy"]].plot(ax=ax, linewidth=0.8)
	df_cook[["Parks"]].plot(ax=ax, linewidth=0.8)
	df_cook[["Transit Stations"]].plot(ax=ax, linewidth=0.8)
	df_cook[["Workplaces"]].plot(ax=ax, linewidth=0.8)
	df_cook[["Residential"]].plot(ax=ax, linewidth=0.8)
	df_cook[["Max Temperature"]].plot(ax=ax, linewidth=0.8)
	plt.xlabel("Dates", fontsize=13)
	plt.ylabel("Percentage on Mobility Change", fontsize=13)
	plt.title('Mobility Change, Max Temperature and Deaths per Population in Cook County', fontsize=18)
	plt.savefig('../output/graphs/matrix_predictions_counties/cook_line.png')

	fig, ax = plt.subplots()
	df_cook_parks = df_cook[["Parks", "Max Temperature"]]
	df_cook_parks[['Parks']].plot(ax=ax, linewidth=3.8)
	df_cook_parks[['Max Temperature']].plot(ax=ax, linewidth=0.8)
	plt.xlabel("Dates", fontsize=13)
	plt.ylabel("Percentage on Mobility Change", fontsize=13)
	plt.title('Mobility Change in Parks, Max Temperatur in Cook County', fontsize=18)
	plt.savefig('../output/graphs/matrix_predictions_counties/park_tmax_line.png')

	return
