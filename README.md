# covid19-ml-project

### Directory Description
- scripts: Contains '.py' files that run the model
- data_raw:  Contains files directly pulled from online sources
- data_intermediate:  Contains cleaned files ready to be integrated into main data frame
- output:  Contains folders that describe results from the gridsearch
  - data: dfs used for most recent gridsearches (train, validation, test)
  - plots: plots describing the output of the models
  - models_predictions_nopca:  predictions and model objects for various hyper-parameters 
  - models_predictions_pca:  predictions and model objects for various hyper-parameters
  
  
### Description of Files

#### scripts
- build_master_df.py:  assembles df to be used in model
- chart_results.py:  contains functions that build select charts from results
- create_intermediate_data.py:  contains function to populate intermediate data folder
- fit_models_with_pca.py:  functions to prepare and execute gridsearch with pca
- fit_models_without_pca.py:  functions to prepare and execute gridsearch without pca
- get_final_test_error.py:  XXXXXXXXX
- identify_key_features.py  XXXXXXXXX
- import_health.py:  creates intermediate data csv for cdc health characteristics
- load_cl_CDC.py:  creates intermediate data csv for cases and deaths
- load_cl_target.py:  creates intermediate data csv for target variables
- load_interventions.py:  creates intermediate data csv for county-level covid interventions
- pipeline.py:  XXXXXXXX
- pull_census.py:  creates intermediate data csv for ACS and NAICS data from census sources
- pull_noaa.py:  creates intermediate data csv for noaa weather data
- pull_votes.py:  creates intermediate data csv for MIT Election Data
- utils.py:  various utility functions
