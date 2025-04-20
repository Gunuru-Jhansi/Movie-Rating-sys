# 🎬 Movie Rating Prediction System

This project predicts IMDb-style ratings for Indian movies using machine learning. It takes in various movie attributes like genre, director, cast, duration, and release year, then predicts the likely rating.

## 🚀 Features

- Predict movie ratings using a trained `RandomForestRegressor`.
- Feature engineering based on:
  - Movie genre (one-hot encoding)
  - Director and actor statistics
  - Movie duration and age
- Scalable preprocessing pipeline
- Interactive prediction support



## 🧠 Model

- **Algorithm**: Random Forest Regressor
- **Framework**: Scikit-learn
- **Metrics**: Mean Squared Error
- Implemented Flask

## 🛠️ Installation

1. **Clone the repo**:

```bash
git clone https://github.com/your-username/Movie-Rating-sys.git
cd Movie-Rating-sys
pip install pandas numpy scikit-learn joblib
```

The dataset i have used is - https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies

Step 1 : Run movierating.py 
The models will be saved in model folder that will be created after model creation.
Visualization graph comes.

Step 2 : Run predict.py with sample examples
Step 3 : Run Flask App - app.py


This is the web page : 

![Screenshot (12)](https://github.com/user-attachments/assets/1fcf7a31-0061-4359-b24e-9636865567bd)
