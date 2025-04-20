import pandas as pd
import joblib
from preprocess import preprocess_data
from feature_engineering import engineer_features

# Load the dataset into a DataFrame with the correct encoding
# df = pd.read_csv('IMDb.csv', encoding='ISO-8859-1')  # Try using ISO-8859-1 encoding

# # Preprocess the data
# df_processed = preprocess_data(df)  # Pass the DataFrame to preprocess_data

# # Perform feature engineering
# df_engineered, genre_columns = engineer_features(df_processed)  # Ensure you're passing processed data

# # Save genre_columns for use in predict.py
# joblib.dump(genre_columns, 'models/genre_columns.pkl')

# print("✅ genre_columns saved.")

import joblib

# Load genre_columns.pkl
genre_columns = joblib.load('models/genre_columns.pkl')

# Print the genre columns to verify
print("Genre Columns:", genre_columns)
