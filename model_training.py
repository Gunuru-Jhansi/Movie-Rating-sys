from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

def train_model(X_train, y_train):
    """
    Train the Random Forest model, save it, and store feature names.
    """
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Initialize scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize and train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit the model
        model.fit(X_train_scaled, y_train)
        
        # Model Evaluation
        y_train_pred = model.predict(X_train_scaled)
        mse = mean_squared_error(y_train, y_train_pred)
        r2 = r2_score(y_train, y_train_pred)
        
        print(f"Model Evaluation: MSE = {mse}, R2 = {r2}")
        
        # Save the feature names        
        # Save model and scaler
        joblib.dump(model, 'models/movie_rating_model.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        
        print("Model and scaler saved in 'models' directory")
        return model, scaler
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise
