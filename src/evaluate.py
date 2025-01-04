import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)



def evaluate_best_pipeline(random_lgbm_search, X_test, y_test):
    """
    Retrieve the best pipeline from a RandomizedSearchCV (or similar),
    use it to predict on the test set, and print evaluation metrics.
    """

    best_pipeline = random_lgbm_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    return best_pipeline, mse, r2

