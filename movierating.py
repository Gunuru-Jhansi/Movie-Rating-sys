from preprocess import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model
from cross_validation import cross_validate
from visualize import visualize_predictions
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def main():
    try:
        print("Loading and preprocessing data...")
        data=pd.read_csv('IMDb.csv',encoding='ISO-8859-1')
        df = preprocess_data(data)
        
        print("Engineering features...")
        df, genre_columns = engineer_features(df)
        
        # Define features - make sure these columns exist in df
        base_features = [
            'Duration', 'movie_age', 'director_avg_rating',
            'actor1_avg_rating', 'actor2_avg_rating', 'actor3_avg_rating',
            'director_movie_count', 'average_votes'
        ]
        
        # Verify all features exist
        all_features = base_features + list(genre_columns)
        existing_features = [f for f in all_features if f in df.columns]
        
        print("Using features:", existing_features)
        
        # Prepare data
        X = df[existing_features]
        y = df['Rating']
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        model, scaler = train_model(X_train, y_train)
        
        # Transform test data
        X_test_scaled = scaler.transform(X_test)
        
        print("Making predictions...")
        y_pred = model.predict(X_test_scaled)
        
        print("Performing cross-validation...")
        cv_rmse = cross_validate(model, X, y)
        print(f"Cross-Validation RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
        
        print("Creating visualizations...")
        visualize_predictions(y_test, y_pred)
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': existing_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print("\nDataFrame columns available:", df.columns.tolist())
        raise

if __name__ == "__main__":
    main()
