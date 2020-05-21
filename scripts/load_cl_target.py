import pandas as pd
from os.path import join, abspath, dirname, exists

ROOT = join(dirname(dirname(abspath(__file__))), 'data')
RAW = join(ROOT, 'raw')
CL = join(ROOT, 'clean_target')

def main():

	path = join(RAW, 'Global_Mobility_Report.csv')
	mobility = pd.read_csv(join(path))

	### filter
	us_mob = filter_counties(mobility)


	county = pd.read_csv(join(RAW, 'FIPS-County List.csv')).astype(object)
	county['CountyFIPS'] = county['CountyFIPS'].apply(lambda x: prepend_0s(str(x), 3))
	county['StateFIPS'] = county['StateFIPS'].apply(lambda x: prepend_0s(str(x), 2))

	merged = us_mob.merge(county, how='left', left_on=('county'), right_on='CountyName')

	merged.to_csv(join(CL, 'us_mobility.csv'), index=False)

	return merged


def filter_counties(df):

	new_df = df[(df['country_region'] == 'United States') & df['sub_region_2'].notnull()]
	new_df.rename({'sub_region_2': 'county'}, axis=1, inplace=True)

	return new_df




if __name__ == "__main__":

	main()
