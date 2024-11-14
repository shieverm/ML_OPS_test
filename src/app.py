"""
ML Ops Test Script

Instructions:
- Build this script from scratch.
- Ensure all necessary steps are included as per the task requirements.
- Document your code clearly.
- Include your name and the date at the top of this file.

Applicant Name: [Your Name]
Date: [Current Date]

"""

from preprocessing import (
    get_file_path, load_data, preprocess_data,
    fill_missing_values, feature_engineering, scale_and_split_data
)
# from model_training import train_linear_model  # Import any other training functions you need
# # from src.model_evaluation import evaluate_model  # Import evaluation functions if applicable
# import joblib  # For saving/loading models

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
        X_train, X_test, y_train, y_test = scale_and_split_data(merged_data)
        if X_train is None:
            print("Data scaling and splitting failed. Exiting.")
            return

        # Train the model (example: Linear Regression)
        model = train_linear_model(X_train, y_train)
        if model:
            print("Model trained successfully.")
            joblib.dump(model, 'linear_model.pkl')  # Save the model for later use
        else:
            print("Model training failed.")
            return

        # Optional: Evaluate the model
        if model:
            evaluate_model(model, X_test, y_test)  # If you have an evaluation function in your src module

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

