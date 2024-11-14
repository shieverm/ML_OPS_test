"""
ML Ops Test Script

Instructions:
- Build this script from scratch.
- Ensure all necessary steps are included as per the task requirements.
- Document your code clearly.
- Include your name and the date at the top of this file.

Applicant Name: [Shiena Verma]
Date: [11.11.2024]

"""

import matplotlib.pyplot as plt
import seaborn as sns

"""
ML Ops Test Script - Model Evaluation

Instructions:
- This script evaluates the performance of trained models.
- It includes residual analysis, regression metrics, and optional hyperparameter tuning.
- Documented clearly as per requirements.

Applicant Name: [Your Name]
Date: [11.11.2024]
"""

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso

def plot_residuals(y_test, y_pred):
    """
    Plot residual distribution and residuals vs predicted values scatter plot.
    Args:
        y_test (array-like): True values for the test set.
        y_pred (array-like): Predicted values from the model.
    """
    try:
        residuals = y_test - y_pred

        # Residual histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Residual Distribution')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.show()

        # Residual vs Predicted scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.show()

    except Exception as e:
        print(f"Error during residual plotting: {e}")

def evaluate_model(y_test, y_pred):
    """
    Evaluate the model using common regression metrics.
    Args:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted values.
    """
    try:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Model Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

    except Exception as e:
        print(f"Error during model evaluation: {e}")

def perform_hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for Ridge and Lasso regression using GridSearchCV.
    Args:
        X_train (DataFrame): Training feature matrix.
        y_train (array-like): Training target variable.
    """
    try:
        # Grid search for Ridge Regression
        ridge_param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
        grid_search_ridge = GridSearchCV(Ridge(), ridge_param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search_ridge.fit(X_train, y_train)
        print(f"Best alpha for Ridge: {grid_search_ridge.best_params_}")

        # Grid search for Lasso Regression
        lasso_param_grid = {'alpha': [0.01, 0.1, 1, 10]}
        grid_search_lasso = GridSearchCV(Lasso(), lasso_param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search_lasso.fit(X_train, y_train)
        print(f"Best alpha for Lasso: {grid_search_lasso.best_params_}")

    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")

def main_model_evalaution(X_test, y_test, X_train, y_train):
    """
    Main function to load and evaluate the trained model.
    Args:
        X_test (DataFrame): Test feature matrix.
        y_test (array-like): True target values.
        X_train, y_train (DataFrame, array-like): Training data for hyperparameter tuning.
    """
    try:
        # Load the trained model
        model = joblib.load('linear_model.pkl')
        print("Model loaded successfully.")

        # Generate predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        evaluate_model(y_test, y_pred)

        # Plot residuals
        plot_residuals(y_test, y_pred)

        # Optional: Perform hyperparameter tuning for Ridge and Lasso
        perform_hyperparameter_tuning(X_train, y_train)

    except Exception as e:
        print(f"An error occurred in the main function: {e}")


    # Call main function with placeholders (replace with actual data)
    
