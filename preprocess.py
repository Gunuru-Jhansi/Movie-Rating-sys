# import pandas as pd
# from datetime import datetime
# import numpy as np

# def preprocess_data(filename):
#     """
#     Preprocess the IMDb dataset
#     """
#     try:
#         # Read the dataset with appropriate encoding
#         df = pd.read_csv(filename, encoding='ISO-8859-1')
        
#         # Handle missing values in Rating (our target variable)
#         df = df.dropna(subset=['Rating'])
        
#         # Handle categorical missing values
#         categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
#         for col in categorical_cols:
#             df[col] = df[col].fillna('Unknown')
        
#         # Clean Duration column
#         df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
#         df['Duration'] = df['Duration'].fillna(df['Duration'].median())
        
#         # Clean Year column and create movie_age
#         current_year = datetime.now().year
#         df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
#         df['movie_age'] = current_year - df['Year']
#         df['movie_age'] = df['movie_age'].fillna(df['movie_age'].median())
        
#         # Clean Votes column
#         df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
        
#         return df
        
#     except Exception as e:
#         print(f"Error in preprocessing: {str(e)}")
#         raise

import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(df):
    """
    Preprocess a single movie DataFrame for prediction.
    Assumes all necessary fields are present (from user input).
    """
    try:
        # Fill missing categorical fields with 'Unknown' if any
        for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        # Clean Duration (convert '120 min' → 120.0)
        if 'Duration' in df.columns:
            df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)
            df['Duration'] = df['Duration'].fillna(120.0)  # fallback default

        # Compute movie_age from Year
        if 'Year' in df.columns:
            current_year = datetime.now().year
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['movie_age'] = current_year - df['Year']
            df['movie_age'] = df['movie_age'].fillna(10)  # default age

        # Add dummy Votes column for compatibility with feature_engineering
        df['Votes'] = 10000.0  # use a constant, or allow this in your form if needed

        return df

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
