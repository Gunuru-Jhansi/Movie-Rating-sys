# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# def engineer_features(df):
#     """
#     Engineer features for the movie rating prediction
#     """
#     try:
#         # Create director features
#         director_stats = df.groupby('Director').agg({
#             'Rating': ['mean', 'count']
#         }).reset_index()
#         director_stats.columns = ['Director', 'director_avg_rating', 'director_movie_count']
#         df = df.merge(director_stats, on='Director', how='left')
        
#         # Create actor features
#         for i in range(1, 4):
#             actor_col = f'Actor {i}'
#             actor_stats = df.groupby(actor_col).agg({
#                 'Rating': ['mean', 'count']
#             }).reset_index()
#             actor_stats.columns = [actor_col, f'actor{i}_avg_rating', f'actor{i}_movie_count']
#             df = df.merge(actor_stats, on=actor_col, how='left')
        
#         # Handle Genre
#         df['Genre'] = df['Genre'].str.split(',').str[0]
#         genre_dummies = pd.get_dummies(df['Genre'], prefix='genre')
#         df = pd.concat([df, genre_dummies], axis=1)
        
#         # Normalize votes
#         df['average_votes'] = df['Votes']
#         df['average_votes'] = (df['average_votes'] - df['average_votes'].mean()) / df['average_votes'].std()
        
#         # Fill any remaining NaN values with 0
#         df = df.fillna(0)
        
#         return df, list(genre_dummies.columns)
        
#     except Exception as e:
#         print(f"Error in feature engineering: {str(e)}")
#         raise

import pandas as pd
import numpy as np

# def engineer_features(df):
#     """
#     Engineer features for movie rating prediction.
#     Handles both training and prediction modes based on 'Rating' column presence.
#     """
#     try:
#         # Training Mode
#         if 'Rating' in df.columns:
#             # Director stats
#             director_stats = df.groupby('Director').agg({
#                 'Rating': ['mean', 'count']
#             }).reset_index()
#             director_stats.columns = ['Director', 'director_avg_rating', 'director_movie_count']
#             df = df.merge(director_stats, on='Director', how='left')

#             # Actor stats
#             for i in range(1, 4):
#                 actor_col = f'Actor {i}'
#                 actor_stats = df.groupby(actor_col).agg({
#                     'Rating': ['mean', 'count']
#                 }).reset_index()
#                 actor_stats.columns = [actor_col, f'actor{i}_avg_rating', f'actor{i}_movie_count']
#                 df = df.merge(actor_stats, on=actor_col, how='left')
#         else:
#             # Prediction Mode – use default averages
#             df['director_avg_rating'] = 5.0
#             df['director_movie_count'] = 10
#             for i in range(1, 4):
#                 df[f'actor{i}_avg_rating'] = 5.0
#                 df[f'actor{i}_movie_count'] = 10

#         # Genre processing (first genre only)
#         df['Genre'] = df['Genre'].str.split(',').str[0].str.strip()
#         genre_dummies = pd.get_dummies(df['Genre'], prefix='genre')
#         df = pd.concat([df, genre_dummies], axis=1)

#         # Normalize votes (simple z-score)
#         df['average_votes'] = df['Votes']
#         df['average_votes'] = (df['average_votes'] - df['average_votes'].mean()) / df['average_votes'].std()

#         # Fill remaining NaNs
#         df = df.fillna(0)

#         return df, list(genre_dummies.columns)

#     except Exception as e:
#         print(f"Error in feature engineering: {str(e)}")
#         raise

def engineer_features(df):
    """
    Engineer features for movie rating prediction.
    Handles both training and prediction modes based on 'Rating' column presence.
    """
    try:
        # Convert 'Genre' to string and fill NaN values with 'Unknown'
        df['Genre'] = df['Genre'].astype(str).fillna('Unknown')

        # Training Mode
        if 'Rating' in df.columns:
            # Director stats
            director_stats = df.groupby('Director').agg({
                'Rating': ['mean', 'count']
            }).reset_index()
            director_stats.columns = ['Director', 'director_avg_rating', 'director_movie_count']
            df = df.merge(director_stats, on='Director', how='left')

            # Actor stats
            for i in range(1, 4):
                actor_col = f'Actor {i}'
                actor_stats = df.groupby(actor_col).agg({
                    'Rating': ['mean', 'count']
                }).reset_index()
                actor_stats.columns = [actor_col, f'actor{i}_avg_rating', f'actor{i}_movie_count']
                df = df.merge(actor_stats, on=actor_col, how='left')
        else:
            # Prediction Mode – use default averages
            df['director_avg_rating'] = 5.0
            df['director_movie_count'] = 10
            for i in range(1, 4):
                df[f'actor{i}_avg_rating'] = 5.0
                df[f'actor{i}_movie_count'] = 10

        # Genre processing (first genre only)
        df['Genre'] = df['Genre'].str.split(',').str[0].str.strip()  # Split and clean genre
        genre_dummies = pd.get_dummies(df['Genre'], prefix='genre')
        df = pd.concat([df, genre_dummies], axis=1)

        # Normalize votes (simple z-score)
        df['average_votes'] = df['Votes']
        df['average_votes'] = (df['average_votes'] - df['average_votes'].mean()) / df['average_votes'].std()

        # Fill remaining NaNs
        df = df.fillna(0)

        return df, list(genre_dummies.columns)

    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        raise
