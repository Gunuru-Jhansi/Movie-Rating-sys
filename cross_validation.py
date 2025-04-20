from sklearn.model_selection import cross_val_score
import numpy as np

def cross_validate(model, X, y):
    """
    Perform cross-validation
    """
    try:
        cv_scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        return np.sqrt(-cv_scores)
    except Exception as e:
        print(f"Error in cross-validation: {str(e)}")
        raise
