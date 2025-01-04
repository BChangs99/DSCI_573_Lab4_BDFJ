
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV


def run_ridge_random_search(X_train, y_train, preprocessor,
    param_dist_ridge, scoring_metric='r2', n_iter=1,
    refit_metric='R2', random_state=573, opti_results=None):
    """
    Create a RidgeCV pipeline and run a RandomizedSearchCV to find the best parameters.
    Then, store evaluation metrics in the 'opti_results' dictionary (if provided).

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        preprocessor (sklearn.pipeline.Pipeline): Preprocessing steps.
        param_dist_ridge (dict): Parameter distributions for the randomized search.
            - The keys should follow the format "ridgecv__<param>" to match the 
              RidgeCV step in the pipeline.
        scoring_metric (str, optional): The main metric used for evaluating each parameter
            setting (e.g., 'r2' or a custom scorer). Defaults to 'r2'.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 1.
        refit_metric (str, optional): The metric used to refit the best model. Defaults to 'R2'.
        random_state (int, optional): Seed for random number generators. Defaults to 573.
        opti_results (dict, optional): Dictionary for storing results. If provided,
            the function will add or update the "ridge" key with training and validation R² scores.

    Returns:
        RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model 
        and search results.
    """
    random_ridge_pipe = make_pipeline(
        preprocessor,
        RidgeCV()
    )

    random_ridge_search = RandomizedSearchCV(
        estimator=random_ridge_pipe,
        param_distributions=param_dist_ridge, 
        n_iter=n_iter,
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        scoring=scoring_metric, 
        refit=refit_metric,
        verbose=False
    )

    random_ridge_search.fit(X_train, y_train)

    if opti_results is not None:
        opti_results['ridge'] = {
            "Optimized Train Score (R2)": random_ridge_search.best_estimator_.score(X_train, y_train),
            "Optimized Validation Score (R2)": random_ridge_search.best_score_
        }

    return random_ridge_search



def run_lgbm_random_search(X_train, y_train, preprocessor, param_grid_lgbm, 
                           scoring_metric='r2', n_iter=1, refit_metric='R2', 
                           random_state=573, opti_results=None):
    """
    Create an LGBM pipeline and run a RandomizedSearchCV to find the best parameters.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        preprocessor (sklearn.pipeline.Pipeline): Preprocessing steps.
        param_grid_lgbm (dict): Parameter distributions for the randomized search.
        scoring_metric (str, optional): The metric used for evaluating each parameter setting.
            Defaults to 'r2'.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 1.
        refit_metric (str, optional): Metric used to refit the best model. Defaults to 'R2'.
        random_state (int, optional): Seed for random number generators. Defaults to 573.

    Returns:
        RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model and
                            search results.
    """

    pipe_lgbm_reduced = make_pipeline(
        preprocessor,
        LGBMRegressor(n_jobs=-1, random_state=random_state)
    )

    random_lgbm_search = RandomizedSearchCV(
        pipe_lgbm_reduced,
        param_distributions=param_grid_lgbm,
        n_iter=n_iter,
        n_jobs=-1,
        return_train_score=True,
        scoring=scoring_metric,
        refit=refit_metric,
        verbose=False,
        random_state=random_state
    )

    random_lgbm_search.fit(X_train, y_train)
    if opti_results is not None:
        opti_results['lgbm'] = {
            "Optimized Train Score (R2)": random_lgbm_search.best_estimator_.score(X_train, y_train),
            "Optimized Validation Score (R2)": random_lgbm_search.best_score_
        }

    return random_lgbm_search



def run_rf_random_search(X_train, y_train, preprocessor, param_grid_rf, 
    scoring_metric='r2', n_iter=1, refit_metric='R2', 
    random_state=573, opti_results=None):
    """
    Create a Random Forest pipeline and run a RandomizedSearchCV to find the best parameters.
    Then, store evaluation metrics in the 'opti_results' dictionary.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        preprocessor (sklearn.pipeline.Pipeline): Preprocessing steps.
        param_grid_rf (dict): Parameter distributions for the randomized search.
        scoring_metric (str, optional): The main metric used for evaluating each parameter
            setting (e.g., 'r2' or a custom scorer). Defaults to 'r2'.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 1.
        refit_metric (str, optional): The metric used to refit the best model. Defaults to 'R2'.
        random_state (int, optional): Seed for random number generators. Defaults to 573.
        opti_results (dict, optional): Dictionary for storing results. If provided,
            the function will add or update the "rf" key with training and validation R² scores.

    Returns:
        RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model 
        and search results.
    """

    pipe_rf_reduced = make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_jobs=-1,
            random_state=random_state
        )
    )
    
    random_rf_search = RandomizedSearchCV(
        estimator=pipe_rf_reduced,
        param_distributions=param_grid_rf,
        n_iter=n_iter,
        random_state=random_state,
        return_train_score=True,
        scoring=scoring_metric, 
        refit=refit_metric,
        n_jobs=-1
    )
    
    random_rf_search.fit(X_train, y_train)
    
    if opti_results is not None:
        opti_results['rf'] = {
            "Optimized Train Score (R2)": random_rf_search.best_estimator_.score(X_train, y_train),
            "Optimized Validation Score (R2)": random_rf_search.best_score_
        }
    
    return random_rf_search



def run_elastic_net_random_search(X_train, y_train, preprocessor, param_grid_elastic,
    scoring_metric='r2', n_iter=1, refit_metric='R2',
    random_state=573, cv=5, opti_results=None):
    """
    Create an Elastic Net pipeline and run a RandomizedSearchCV to find the best parameters.
    Then, store evaluation metrics in the 'opti_results' dictionary (if provided).

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        preprocessor (sklearn.pipeline.Pipeline): Preprocessing steps.
        param_grid_elastic (dict): Parameter distributions for the randomized search.
        scoring_metric (str, optional): The main metric used for evaluating each parameter 
            setting (e.g., 'r2' or a custom scorer). Defaults to 'r2'.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 1.
        refit_metric (str, optional): The metric used to refit the best model. Defaults to 'R2'.
        random_state (int, optional): Seed for random number generators. Defaults to 573.
        cv (int, optional): Number of folds for ElasticNetCV. Defaults to 5.
        opti_results (dict, optional): Dictionary for storing results. If provided,
            the function will add or update the "elastic_net" key with training and 
            validation R² scores.

    Returns:
        RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model 
        and search results.
    """
    pipe_elasticnet_reduced = make_pipeline(
        preprocessor,
        ElasticNetCV(max_iter=20_000, tol=0.01, cv=cv)
    )

    random_en_search = RandomizedSearchCV(
        estimator=pipe_elasticnet_reduced, 
        param_distributions=param_grid_elastic,
        n_iter=n_iter,
        verbose=1,
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        scoring=scoring_metric,
        refit=refit_metric
    )

    random_en_search.fit(X_train, y_train)

    if opti_results is not None:
        opti_results['elastic_net'] = {
            "Optimized Train Score (R2)": random_en_search.best_estimator_.score(X_train, y_train),
            "Optimized Validation Score (R2)": random_en_search.best_score_
        }

    return random_en_search