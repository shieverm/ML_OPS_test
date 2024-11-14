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
from sklearn.linear_model import LinearRegression
import joblib

def train_linear_model(X_train, y_train):
    """
    Train a Linear Regression model.
    Args:
        X_train (DataFrame): Training feature matrix.
        y_train (array-like): Training target variable.
    Returns:
        model (LinearRegression): Trained Linear Regression model.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Linear Regression model trained successfully.")
        return model
    except Exception as e:
        print(f"Error during Linear Regression model training: {e}")
        return None

def model_training_main(X_train, X_test, y_train):
    """
    Main function to train the model and save it.
    Args:
        X_train, X_test, y_train (DataFrame, array-like): Train-test data.
    """
    try:
        # Train the model
        model = train_linear_model(X_train, y_train)
        if model:
            # Save the trained model
            joblib.dump(model, 'linear_model.pkl')
            print("Model saved as 'linear_model.pkl'")
            # Generate and save predictions (optional, for later evaluation)
            y_pred = model.predict(X_test)
            return model, y_pred
        else:
            print("Model training failed.")
            return None, None
    except Exception as e:
        print(f"An error occurred in the main function: {e}")
        return None, None

