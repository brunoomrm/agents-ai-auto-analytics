import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import statsmodels.api as sm

def adjusted_r2(r2, n, k):
    '''
    Computes adjusted R² from ordinary R², sample size (n), and number of predictors (k).
    
    Inputs
    ----------
    r2 : float
        Ordinary R-squared value.
    n : int
        Number of samples.
    k : int
        Number of predictors (features) used in the model.
    
    Output
    -------
    float
        Adjusted R-squared value.
    '''
    denominator = n - k - 1 ### if we have more features than samples, adjusted r squared is not meaninigul
    if denominator <= 0:
        return np.nan
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

def run_ols_regression(df, target='price', test_size=0.2, random_state=42, print_output=True):
    '''
    Performs OLS linear regression with automatic handling of categorical variables and missing data.
    Converts all categorical columns to dummy variables, ensures numerics, splits into train/test, fits the model and returns performance metrics and model summary.
    
    Inputs:

    df : pandas.DataFrame
        Input dataframe containing all features and the target column.
    target : str, default='price'
        Name of the column to predict (target variable).
    test_size : float, default=0.2
        Proportion of the dataset to be used as test set.
    random_state : int, default=42
        Random seed for train/test splitting to ensure reproducibility.
    print_output : bool, default=True
        Whether to print model summary and performance metrics.
    
    Output:

    dict
        Dictionary with:
            - 'rmse' : float, root mean squared error on test set
            - 'mae' : float, mean absolute error on test set
            - 'r2' : float, test R² score
            - 'adj_r2' : float, test adjusted R² score
            - 'model' : statsmodels.regression.linear_model.RegressionResultsWrapper, fitted model object
            - 'importances' : pandas.Series, absolute t-values of each feature (as proxy for "importance")
            - 'X_test' : pandas.DataFrame, test set features (with intercept column)
            - 'y_test' : pandas.Series, test set targets
            - 'y_pred' : numpy.ndarray, model predictions on test set
    '''
    features = [col for col in df.columns if col != target]
    df_ols = df.dropna(subset=features + [target]).copy()
    
    # Get dummies for all categoricals, drop_first for multicollinearity    
    X = pd.get_dummies(df_ols[features], drop_first=True)
    
    # --> Force standard float (numpy type, not extension type) <--
    X = X.astype(np.float64)
    
    X = X.dropna()
    y = df_ols.loc[X.index, target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train_sm).fit()
    y_pred = model.predict(X_test_sm)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2_res = adjusted_r2(r2, X_test_sm.shape[0], X_test_sm.shape[1] - 1)
    importances = pd.Series(np.abs(model.tvalues[1:]), index=X_train.columns).sort_values(ascending=False)
    if print_output:
        print(model.summary())
        print(f"\nTest MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")
        print(f"Adjusted R²: {adj_r2_res:.4f}")
        print("\nTop Feature Importances (|t-value|):")
        print(importances.head(10))
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2_res,
        'model': model,
        'importances': importances,
        'X_test': X_test_sm,
        'y_test': y_test,
        'y_pred': y_pred
    }

def run_xgboost_cv(df, features, remove_nan, target='price',
    exclude_cols=None,param_grid=None,test_size=0.2,random_state=42,print_output=True):
    '''
    Runs XGBoost regression with cross-validated hyperparameter tuning.
    Handles encoding of categorical features and NaN removal, splits data, runs GridSearchCV,evaluates performance, prints summary, and returns results as a dictionary.

    Inputs:
    df : pandas.DataFrame
        Dataframe containing both features and target.
    features : list of str
        List of feature column names to use in modeling.
    remove_nan : bool
        If True, drop all rows with missing data in features or target before modeling.
    target : str, default='price'
        Name of the column to predict.
    exclude_cols : list of str or None, default=None
        Which columns to exclude from features.
    param_grid : dict or None, default=None
        Grid of XGBoost hyperparameters for GridSearchCV. If None, a default grid is used.
    test_size : float, default=0.2
        The proportion of the data used for testing (the rest for training).
    random_state : int, default=42
        Random seed for train and test split.
    print_output : bool, default=True
        If True, print results and top features.

    Outputs
    Dictionary with:
            'model': best trained XGBRegressor,
            'best_params': best parameters from grid search,
            'cv_rmse': best grid search RMSE (negated score),
            'rmse': RMSE on test set,
            'mae': MAE on test set,
            'r2': R² on test set,
            'adj_r2': Adjusted R² on test set,
            'X_test': Test features,
            'importances': Feature importance as a pandas Series,
            'y_test': True test values,
            'y_pred': Predicted test values
        }
    '''
    if exclude_cols is None:
        exclude_cols = [target]
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 250, 400],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.8, 1],
        }
    df_xgboost = df.copy()
    median_price = df_xgboost[target].median()
    df_xgboost[target] = df_xgboost[target].fillna(median_price)

    # Remove missing values in features (optional!!)
    if remove_nan:
        df_xgboost = df_xgboost.dropna(subset=features + [target]).copy()
        
    X = df_xgboost[features].copy()
    y = df_xgboost[target].copy()

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    xgb = XGBRegressor(tree_method='hist', random_state=random_state, enable_categorical=True)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        cv=5,
        verbose=2
    )
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2_res = adjusted_r2(r2, X_test.shape[0], X_test.shape[1])

    importances = pd.Series(grid.best_estimator_.feature_importances_, index=features).sort_values(ascending=False)

    if print_output:
        print("Best parameters:", grid.best_params_)
        print("Best CV RMSE:", -grid.best_score_)
        print(f"Test RMSE: {rmse:.2f}")
        print(f"Test MAE: {mae:.2f}")
        print(f"Test R²: {r2:.3f}")
        print(f"Adjusted R²: {adj_r2_res:.3f}")
        print("\nTop Feature Importances:")
        print(importances.head(10))

    return {
        'model': grid.best_estimator_,
        'best_params': grid.best_params_,
        'cv_rmse': -grid.best_score_,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2_res,
        'X_test': X_test,
        'importances': importances,
        'y_test': y_test,
        'y_pred': y_pred
    }