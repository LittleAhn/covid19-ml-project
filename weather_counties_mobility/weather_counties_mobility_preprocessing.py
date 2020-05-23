
import pandas as pd
import warnings
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pipeline
warnings.filterwarnings('ignore')
mobility = pd.read_csv("Global_Mobility_Report.csv")
google = mobility[mobility['country_region_code'] == 'US'].dropna().iloc[:, 3:]
google.sub_region_2 = google.sub_region_2.str.upper() 


# In[7]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import geopandas as gpd
google.set_index('date', inplace=True)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(21, 4)})


# In[8]:


google_2 = google[google['sub_region_2'] == 'COOK COUNTY']
google_2.plot(linewidth=0.8)
plt.xlabel("Dates")
plt.ylabel("Percentage on Mobility Change")
plt.title('Mobility Change in Cook County')


# In[9]:


google_3 = google[google['sub_region_2'] == 'ADA COUNTY']
google_3.plot(linewidth=0.8)
plt.xlabel("Dates")
plt.ylabel("Percentage on Mobility Change")
plt.title('Mobility Change in Ada County')


# In[10]:


weather_counties = pd.read_csv("US_counties_COVID19_health_weather_data.csv")
weather_counties = weather_counties[['date', 'county', 'state', 'fips', 'cases', 'deaths', 'stay_at_home_announced', 'stay_at_home_effective', 'lat', 'lon', 'area_sqmi', 'mean_temp', 'precipitation']]


# In[11]:


by_date= weather_counties.groupby('date')['mean_temp'].mean()
by_date.plot(kind='bar')


# In[12]:


by_county= weather_counties.groupby('county')['mean_temp'].mean()
by_county.plot()


# In[13]:


interventions = pd.read_csv("interventions.csv")


# In[14]:


weather_counties.fips= weather_counties.fips.convert_objects(convert_numeric=True)
weather_counties_mobility = interventions.merge(weather_counties, left_on='FIPS', right_on='fips')
del weather_counties_mobility['fips']
del weather_counties_mobility['county']
del weather_counties_mobility['state']
weather_counties_mobility.AREA_NAME = weather_counties_mobility.AREA_NAME.str.upper() 
weather_counties_mobility = weather_counties_mobility.merge(google, left_on='AREA_NAME', right_on='sub_region_2')
del weather_counties_mobility['sub_region_2']
weather_counties_mobility = weather_counties_mobility.rename(columns={"AREA_NAME": "county"})


# In[18]:


train, test = train_test_split(weather_counties_mobility, test_size = 0.33, random_state = 42)


# In[19]:


cont_vars = ['stay at home', 'mean_temp', 'precipitation'] 
cat_vars = ['date', 'FIPS', 'STATE', 'county', 'stay at home', 'stay_at_home_announced', 'stay_at_home_effective',
       'lat', 'lon', ]
train = pipeline.impute(train)
test = pipeline.impute(test)


# In[21]:


train[cont_vars], scaler = pipeline.normalize(train[cont_vars])
test[cont_vars], scaler = pipeline.normalize(test[cont_vars], scaler)

