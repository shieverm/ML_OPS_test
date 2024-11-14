"""
ML Ops Test Script

Instructions:
- Build this script from scratch.
- Ensure all necessary steps are included as per the task requirements.
- Document your code clearly.
- Include your name and the date at the top of this file.

Applicant Name: [Shiena Verma]
Date: [14/Nov]

"""

# Your code starts here
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from preprocessing import load_and_preprocess_data  # Assuming preprocessing.py is in the same folder

def train_linear_model(X_train, y_train):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error while training Linear Regression model: {e}")
        return None

def train_ridge_model(X_train, y_train, alpha=1.0):
    try:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train, y_train)
        return ridge_model
    except Exception as e:
        print(f"Error while training Ridge Regression model: {e}")
        return None

def train_lasso_model(X_train, y_train, alpha=0.1):
    try:
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        return lasso_model
    except Exception as e:
        print(f"Error while training Lasso Regression model: {e}")
        return None

def evaluate_model_with_cross_validation(model, X, y):
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_mse = -cv_scores.mean()
        print(f"Cross-Validation Mean Squared Error (MSE): {mean_cv_mse:.4f}")
        return mean_cv_mse
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        return None

def main():
    try:
        # Load and preprocess the data
        X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()

        # Train the Linear Regression model
        linear_model = train_linear_model(X_train, y_train)
        if linear_model:
            joblib.dump(linear_model, 'linear_model.pkl')
            print("Linear Regression model saved as 'linear_model.pkl'")
            # Evaluate the model using cross-validation
            evaluate_model_with_cross_validation(linear_model, X, y)

        # Train and evaluate Ridge Regression model
        ridge_model = train_ridge_model(X_train, y_train, alpha=1.0)
        if ridge_model:
            joblib.dump(ridge_model, 'ridge_model.pkl')
            print("Ridge Regression model saved as 'ridge_model.pkl'")
            evaluate_model_with_cross_validation(ridge_model, X, y)

        # Train and evaluate Lasso Regression model
        lasso_model = train_lasso_model(X_train, y_train, alpha=0.1)
        if lasso_model:
            joblib.dump(lasso_model, 'lasso_model.pkl')
            print("Lasso Regression model saved as 'lasso_model.pkl'")
            evaluate_model_with_cross_validation(lasso_model, X, y)

    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
