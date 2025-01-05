from sklearn.pipeline import make_pipeline
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from evaluate import mean_std_cross_val_scores
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV

def make_dummy_pipeline(X_train, y_train, preprocessor, results, scoring_metric):
    """
    Construct an DummyRegressor pipeline and evaluate it using cross-validation.
    """
    dummy = DummyRegressor()


    results["Dummy"] = mean_std_cross_val_scores(
        dummy, X_train, y_train, return_train_score=True, scoring=scoring_metric
    )

    return dummy

def make_ridge_pipeline(X_train, y_train, preprocessor, results, alpha_vals, scoring_metric):
    """
    Construct a Ridge pipeline and evaluate it using cross-validation.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training features.
        y_train (pd.Series or np.ndarray): The training target.
        preprocessor (sklearn.pipeline.Pipeline or similar): The pre-processing pipeline
            (e.g., for feature transformations).
        results (dict): A dictionary to store cross-validation results.
        scoring_metric (str or callable): The metric used for evaluating model performance.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline object consisting of the preprocessor
        and the model.
    """
    pipe_ridge = make_pipeline(preprocessor,
        RidgeCV(alphas = alpha_vals, cv=10)
    )

    # Evaluate the pipeline via cross-validation and store results
    results['RidgeCV'] = mean_std_cross_val_scores(
        pipe_ridge, X_train, y_train, return_train_score=True, scoring=scoring_metric
    )

    return pipe_ridge


def make_lgbm_pipeline(X_train, y_train, preprocessor, results, scoring_metric):
    """
    Construct an LGBM pipeline and evaluate it using cross-validation.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training features.
        y_train (pd.Series or np.ndarray): The training target.
        preprocessor (sklearn.pipeline.Pipeline or similar): The pre-processing pipeline
            (e.g., for feature transformations).
        results (dict): A dictionary to store cross-validation results.
        scoring_metric (str or callable): The metric used for evaluating model performance.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline object consisting of the preprocessor
        and the LGBM model.
    """
    pipe_lgbm = make_pipeline(
        preprocessor,
        LGBMRegressor(n_jobs=-1, verbose=-1, random_state=573)
    )

    # Evaluate the pipeline via cross-validation and store results
    results['lgbm'] = mean_std_cross_val_scores(
        pipe_lgbm, X_train, y_train, return_train_score=True, scoring=scoring_metric
    )

    return pipe_lgbm


def make_rf_pipeline(X_train, y_train, preprocessor, results, scoring_metric):
    """
    Construct a Random Forest pipeline and evaluate it using cross-validation.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training features.
        y_train (pd.Series or np.ndarray): The training target.
        preprocessor (sklearn.pipeline.Pipeline or similar): The pre-processing pipeline
            (e.g., for feature transformations).
        results (dict): A dictionary to store cross-validation results.
        scoring_metric (str or callable): The metric used for evaluating model performance.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline object consisting of the preprocessor
        and the RandomForestRegressor model.
    """
    pipe_rf = make_pipeline(
        preprocessor,
        RandomForestRegressor(n_jobs=-1, random_state=573)
    )

    # Evaluate the pipeline via cross-validation and store results
    results["random_forests"] = mean_std_cross_val_scores(
        pipe_rf, X_train, y_train, return_train_score=True, scoring=scoring_metric
    )

    return pipe_rf


def make_elastic_pipeline(X_train, y_train, preprocessor, results, scoring_metric):
    """
    Construct an ElasticNet pipeline and evaluate it using cross-validation.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training features.
        y_train (pd.Series or np.ndarray): The training target.
        preprocessor (sklearn.pipeline.Pipeline or similar): The pre-processing pipeline
            (e.g., for feature transformations).
        results (dict): A dictionary to store cross-validation results.
        scoring_metric (str or callable): The metric used for evaluating model performance.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline object consisting of the preprocessor
        and the ElasticNetCV model.
    """
    elastic_pipe = make_pipeline(
        preprocessor,
        ElasticNetCV(max_iter=20_000, tol=0.01, cv=10)
    )

    # Evaluate the pipeline via cross-validation and store results
    results['elastic_net'] = mean_std_cross_val_scores(
        elastic_pipe, X_train, y_train, return_train_score=True, scoring=scoring_metric, cv=10
    )

    return elastic_pipe


