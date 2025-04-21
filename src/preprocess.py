import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Drop non-relevant columns (if any)
    data = data.drop(['Time'], axis=1)
    
    # Separate features (X) and target (y)
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Return the preprocessed data
    return X_scaled, y

