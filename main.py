import pandas as pd
from src.preprocess import preprocess_data
from src.model import train_model

# Load the dataset
data = pd.read_csv("data/creditcard.csv")

# Preprocess the data (e.g., handling missing values, scaling, etc.)
preprocessed_data = preprocess_data(data)

# Train the fraud detection model
train_model(preprocessed_data)

