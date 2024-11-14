"""
ML Ops Test Script

Instructions:
- Build this script from scratch.
- Ensure all necessary steps are included as per the task requirements.
- Document your code clearly.
- Include your name and the date at the top of this file.

Applicant Name: [SHIENA VERMA]
Date: [14/11/2024]

"""

from preprocessing import (
    get_file_path, load_data, preprocess_data,
    fill_missing_values, feature_engineering, scale_and_split_data
)
from model_training import *
from model_evaluation import *
import numpy as np

def main():
    try:
        # Load data files
        energy_file = get_file_path('energy_data.csv')
        sensor_data_file = get_file_path('sensor_data.csv')
        weather_data_file = get_file_path('weather_data.csv')

        energy = load_data(energy_file)
        sensor_data = load_data(sensor_data_file)
        weather = load_data(weather_data_file)

        if energy is None or sensor_data is None or weather is None:
            print("Data loading failed. Exiting.")
            return

        # Preprocess data
        merged_data = preprocess_data(energy, sensor_data, weather)
        if merged_data is None:
            print("Data preprocessing failed. Exiting.")
            return

        # Handle missing values using the provided functions
        fill_missing_values(merged_data, 'building_energy_consumption_kwh', 'RelativeHumidity')
        fill_missing_values(merged_data, 'building_energy_consumption_kwh', 'Temperature')

        # Feature engineering
        merged_data = feature_engineering(merged_data)
        if merged_data is None:
            print("Feature engineering failed. Exiting.")
            return

        # Scale and split the data
        X_train, X_test, y_train, y_test,X,y = scale_and_split_data(merged_data)
        if X_train is None:
            print("Data scaling and splitting failed. Exiting.")
            return
        #Train the model
        model, y_pred = model_training_main(X_train, X_test, y_train)
        # Evaluate the model during training
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Linear Regression Performance:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R^2 Score: {r2:.2f}")

        # Initialize the Linear Regression model
        linear_model = LinearRegression()

        # Perform cross-validation with 5 folds (you can change cv as needed)
        cv_mse_linear = cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_squared_error')

        # Convert the negative MSE to positive for interpretability
        cv_mse_linear_mean = np.mean(cv_mse_linear)
        print(f"Cross-Validation Mean Squared Error (MSE) for Linear Regression: {cv_mse_linear_mean:.4f}")
                

        ###Model Evaluation 
        # Call model_evalaution.py
        main_model_evalaution(X_test, y_test, X_train, y_train)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

