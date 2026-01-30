from libs import *

def clean_data(data):
    # Average duplicated values in the dataset
    data = data.groupby('date').mean().reset_index()
    # Interpolate missing values
    for column in data.columns:
        if column != 'date':
            data[column].interpolate(method='pchip', inplace=True)
    return data

def smooth_data(data, rolling_window):
    # Implement smoothing logic here
    data.iloc[:,1:] = data.iloc[:,1:].rolling(window=rolling_window).mean()
    data = data.iloc[rolling_window - 1:].reset_index(drop=True)
    return data

def remove_outliers(data, threshold=5):
    # Calculate z-scores and remove outliers
    data['z_scores'] = stats.zscore(data['gear_bearing_temp'], nan_policy='omit', axis=0)
    data['gear_bearing_temp'] = np.where(data['z_scores'].abs() > threshold, np.nan, data['gear_bearing_temp'])
    data = data.drop(columns=['z_scores'])
    data['gear_bearing_temp'].interpolate(method='pchip', inplace=True)
    return data

def feature_selection(data):
    if 'gear_oil_temp_inlet' in data.columns:
        data['gear_oil_temp'] = np.mean([data['gear_oil_temp'], data['gear_oil_temp_inlet']], axis=0)
        data.drop(columns=['gear_oil_temp_inlet'], inplace=True)
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:,1:])
    data.iloc[:,1:]=scaler.transform(data.iloc[:,1:])
    return data, scaler

def split_sequences(sequences, n_steps):
    # Convert DataFrame to NumPy array excluding the date column
    data = sequences.iloc[:, 1:].to_numpy()
    
    # Calculate the number of samples
    n_samples = len(sequences) - n_steps
    
    # Initialize arrays for X and y
    X = np.zeros((n_samples, n_steps, data.shape[1]))
    y = np.zeros(n_samples)
    
    # Create a view into the data array for sequences
    for i in range(n_samples):
        # Get the sequence
        X[i] = data[i:i+n_steps]
        # Get the target value
        y[i] = data[i+n_steps, 1]
    
    return X, y