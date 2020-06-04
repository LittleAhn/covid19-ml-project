import pandas as pd
from sklearn.preprocessing import StandardScaler


#Pre-Process Data
def impute(df):
    '''
    Returns impute dataframe by filling NaNs with median of column
    '''
    df.fillna(0, inplace=True)
    return df

def normalize(df, scaler=None):
    '''
    Normalizes dataframe features
    If scaler = None, normalize df and save it in scalar
    If scaler is not None, use given scalar's means and sds to normalize
    Returns normalized dataframe and the scaler
    '''
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(df)
    else:
        normalized_features = scaler.transform(df)  
    normalized_df = pd.DataFrame(normalized_features)                                        
    normalized_df.index=df.index
    normalized_df.columns=df.columns
    return normalized_df, scaler


