import joblib
import pandas as pd
import numpy as np
from preprocess import preprocess_data
from feature_engineering import engineer_features

def load_model():
    """Load the trained model, scaler, and genre columns"""
    try:
        model = joblib.load('models/movie_rating_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        genre_columns = joblib.load('models/genre_columns.pkl')  # Load genre columns
        return model, scaler, genre_columns
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict_rating(movie_data):
    """
    Predict rating for a new movie
    
    Parameters:
    -----------
    movie_data : dict
        Dictionary containing movie information with keys:
        - Title
        - Genre
        - Director
        - Actor 1
        - Actor 2
        - Actor 3
        - Year
        - Duration
    """
    try:
        # Convert single movie data to DataFrame
        df = pd.DataFrame([movie_data])
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Engineer features
        df_features, genre_columns = engineer_features(df_processed)
        
        # Load model, scaler, and genre columns
        model, scaler, saved_genre_columns = load_model()

        # Add missing genre columns to the DataFrame if any
        for genre_col in saved_genre_columns:
            if genre_col not in df_features.columns:
                df_features[genre_col] = 0  # Add missing genre columns with value 0

        # Ensure that all other columns are present (like Duration, actor ratings, etc.)
        required_columns = [
            'Duration', 'movie_age', 'director_avg_rating', 'actor1_avg_rating', 
            'actor2_avg_rating', 'actor3_avg_rating', 'director_movie_count', 
            'average_votes'
        ]

        for col in required_columns:
            if col not in df_features.columns:
                df_features[col] = 0  # Set missing columns to default value (0)

        # Reorder columns to match the training data
        df_features = df_features[required_columns + saved_genre_columns]
        
        # Scale features
        X_scaled = scaler.transform(df_features)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return round(prediction, 1)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Example movie data
    new_movie = {
        'Title': 'Example Movie',
        'Genre': 'Action',
        'Director': 'John Doe',
        'Actor 1': 'Actor A',
        'Actor 2': 'Actor B',
        'Actor 3': 'Actor C',
        'Year': 2023,
        'Duration': '120 min'
    }
    
    try:
        predicted_rating = predict_rating(new_movie)
        print(f"\nPredicted Rating for '{new_movie['Title']}': {predicted_rating}")
    except Exception as e:
        print(f"Failed to make prediction: {str(e)}")
