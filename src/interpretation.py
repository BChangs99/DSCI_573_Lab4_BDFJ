import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

def get_permutation_importance(pipe):
    X_train_transformed = pipe[:-1].transform(X_train)
    feature_names = pipe[:-1].get_feature_names_out()
    
    result = permutation_importance(
        pipe[-1],  # Final model in the pipeline
        X_train_transformed,
        y_train,
        n_repeats=10,
        random_state=123,
        n_jobs=-1
    )
    
    perm_sorted_idx = result.importances_mean.argsort()
    
    # Plot permutation importance
    plt.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=[feature_names[i] for i in perm_sorted_idx],  # Use transformed feature names
    )
    plt.xlabel('Permutation feature importance')
    plt.show()


def get_lgbm_explanation(random_lgbm_search, X_test, y_test):
    """
    This function:
      1. Extracts the best pipeline from a search result (excluding the last model step).
      2. Transforms the test set (X_test).
      3. Creates a DataFrame with transformed features and simplified column names.
      4. Resets the index of y_test.
      5. Uses SHAP to create a TreeExplainer for the LightGBM regressor and generates
         explanations for X_test_enc.

    Args:
        random_lgbm_search: The fitted search object (e.g., RandomizedSearchCV) 
            that contains the best LightGBM pipeline.
        X_test (pd.DataFrame): The original test features.
        y_test (pd.Series): The original test targets.

    Returns:
        tuple:
            - y_test_reset (pd.Series): y_test with its index reset.
            - lgbm_explanation (shap._explanation.Explanation): The SHAP values/explanation
              for the LightGBM regressor on the transformed test data.
    """

    preprocessing_steps = random_lgbm_search.best_estimator_[:-1]

    # Transform the test features
    X_test_transformed = preprocessing_steps.transform(X_test)
    feature_names = preprocessing_steps.get_feature_names_out()

    X_test_enc = pd.DataFrame(
        data=X_test_transformed,
        columns=feature_names,
        index=X_test.index
    )

    X_test_enc.columns = X_test_enc.columns.str.split('__').str[1]

    y_test_reset = y_test.reset_index(drop=True)

    lgbm_explainer = shap.TreeExplainer(
        random_lgbm_search.best_estimator_.named_steps['lgbmregressor']
    )

    lgbm_explanation = lgbm_explainer(X_test_enc)

    return y_test_reset, lgbm_explanation

