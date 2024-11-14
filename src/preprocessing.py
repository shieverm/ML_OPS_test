# import os

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# print(os.path.abspath('../data/energy_data.csv'))
# # Get the directory where the current script is located
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to the data file dynamically
# energy_data_file_path = os.path.join(current_dir, '..', 'data', 'energy_data.csv')
# sensor_data_file_path = os.path.join(current_dir, '..', 'data', 'sensor_data.csv')
# weather_data_file_path = os.path.join(current_dir, '..', 'data', 'weather_data.csv')


# def load_and_preprocess_data():
#     # Load datasets
#     energy = pd.read_csv(energy_data_file_path)
#     sensor_data = pd.read_csv(sensor_data_file_path)
#     weather = pd.read_csv(weather_data_file_path)

#    # Convert timestamps to datetime format
#     energy['timestamp'] = pd.to_datetime(energy['timestamp']).dt.tz_localize('UTC')
#     sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp']).dt.tz_convert('UTC')
#     weather['timestamp'] = pd.to_datetime(weather['timestamp']).dt.tz_localize('UTC')

#     print(energy['timestamp'].unique())
#     print(sensor_data['timestamp'].unique())
#     print(weather['timestamp'].unique())


#     # Merge datasets on timestamp using outer join
    
#     merged_data = pd.merge(energy, sensor_data, on='timestamp', how='outer')
#     merged_data = pd.merge(merged_data, weather, on='timestamp', how='outer')


#     # Handle missing values (example: forward fill)
#     merged_data.fillna(method='ffill', inplace=True)

#    # Handling missing values for Relative Humidity
#     reg = LinearRegression()

#     # Training data: where Relative Humidity is not missing
#     not_missing_mask = merged_data['RelativeHumidity'].notna()
#     X_train = merged_data.loc[not_missing_mask, ['building_energy_consumption_kwh']].values.reshape(-1, 1)
#     y_train = merged_data.loc[not_missing_mask, 'RelativeHumidity'].values

#     # Fit regression model
#     reg.fit(X_train, y_train)

#     # Predict missing Relative Humidity values
#     missing_mask = merged_data['RelativeHumidity'].isna()
#     X_pred = merged_data.loc[missing_mask, ['building_energy_consumption_kwh']].values.reshape(-1, 1)
#     merged_data.loc[missing_mask, 'RelativeHumidity'] = reg.predict(X_pred)

#     # Repeat the process for Temperature
#     reg = LinearRegression()

#     # Training data: where Temperature is not missing
#     not_missing_mask = merged_data['Temperature'].notna()
#     X_train = merged_data.loc[not_missing_mask, ['building_energy_consumption_kwh']].values.reshape(-1, 1)
#     y_train = merged_data.loc[not_missing_mask, 'Temperature'].values

#     # Fit regression model
#     reg.fit(X_train, y_train)

#     # Predict missing Temperature values
#     missing_mask = merged_data['Temperature'].isna()
#     X_pred = merged_data.loc[missing_mask, ['building_energy_consumption_kwh']].values.reshape(-1, 1)
#     merged_data.loc[missing_mask, 'Temperature'] = reg.predict(X_pred)\
    
#         # Handle both missing values and empty brackets
#     merged_data['site_id'] = merged_data['site_id'].replace('[]', '["Unknown Area 2"]').fillna('["Unknown Area 1"]')
#     merged_data['tags_name'] = merged_data['tags_name'].fillna('UnknownTag')

#     # Distribution plot for energy consumption
#     plt.figure(figsize=(10, 6))
#     sns.histplot(merged_data['building_energy_consumption_kwh'], bins=30, kde=True)
#     plt.title('Distribution of Energy Consumption')
#     plt.show()



#     # Boxplot for energy consumption
#     sns.boxplot(data=merged_data, x='building_energy_consumption_kwh')
#     plt.title('Boxplot of Building Energy Consumption')
#     plt.show()

#     # Scatterplot to observe relationships
#     sns.scatterplot(data=merged_data, x='Temperature', y='building_energy_consumption_kwh')
#     plt.title('Energy Consumption vs Temperature')
#     plt.show()


#     Q1 = merged_data['building_energy_consumption_kwh'].quantile(0.25)
#     Q3 = merged_data['building_energy_consumption_kwh'].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # Identify outliers
#     outliers = merged_data[(merged_data['building_energy_consumption_kwh'] < lower_bound) | 
#                         (merged_data['building_energy_consumption_kwh'] > upper_bound)]

#     print(f"Number of outliers: {outliers.shape[0]}")

#     # Extracting useful time-based features
#     merged_data['hour'] = merged_data['timestamp'].dt.hour
#     merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek  # 0 = Monday, 6 = Sunday
#     merged_data['month'] = merged_data['timestamp'].dt.month
#     merged_data['week_of_year'] = merged_data['timestamp'].dt.isocalendar().week
#     merged_data['is_weekend'] = merged_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Binary flag for weekends

#         # Interaction features
#     merged_data['temp_energy_interaction'] = merged_data['Temperature'] * merged_data['building_energy_consumption_kwh']
#     merged_data['humidity_energy_interaction'] = merged_data['RelativeHumidity'] * merged_data['building_energy_consumption_kwh']
    
#     # One-hot encoding for categorical features
#     merged_data = pd.get_dummies(merged_data, columns=['site_id', 'tags_name'], drop_first=True)

#     # Autocorrelation plot
#     pd.plotting.autocorrelation_plot(merged_data['building_energy_consumption_kwh'])
#     plt.title('Autocorrelation of Energy Consumption')
#     plt.show()

#     # Create lag features for the first few lags
#     merged_data['lag_1'] = merged_data['building_energy_consumption_kwh'].shift(1)
#     merged_data['lag_2'] = merged_data['building_energy_consumption_kwh'].shift(2)
#     merged_data['lag_3'] = merged_data['building_energy_consumption_kwh'].shift(3)
        
#     # Calculate rolling statistics (window size can be adjusted)
#     merged_data['rolling_mean_3'] = merged_data['building_energy_consumption_kwh'].rolling(window=3).mean()
#     merged_data['rolling_std_3'] = merged_data['building_energy_consumption_kwh'].rolling(window=3).std()
        
    
#     scaler = MinMaxScaler()
#     numerical_cols = ['building_energy_consumption_kwh', 'RelativeHumidity', 'Temperature']
#     merged_data[numerical_cols] = scaler.fit_transform(merged_data[numerical_cols])

#     merged_data.dropna(inplace=True)
#     # Define features (X) and target (y)
#     X = merged_data.drop(['building_energy_consumption_kwh', 'timestamp'], axis=1)  # Drop target and any irrelevant columns
#     y = merged_data['building_energy_consumption_kwh']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
#     return X_train, X_test, y_train, y_test, X, y

# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()
#     # Print the shapes of the resulting datasets
#     print(f"X_train shape: {X_train.shape}")
#     print(f"X_test shape: {X_test.shape}")
#     print(f"y_train shape: {y_train.shape}")
#     print(f"y_test shape: {y_test.shape}")

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score



def get_file_path(file_name):
    try:
        # Get the directory where the current script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the data file dynamically
        return os.path.join(current_dir, '..', 'data', file_name)
    except Exception as e:
        print(f"Error constructing file path for {file_name}: {e}")
        return None

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def preprocess_data(energy, sensor_data, weather):
    try:
        # Convert timestamps to datetime format
        energy['timestamp'] = pd.to_datetime(energy['timestamp']).dt.tz_localize('UTC')
        sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp']).dt.tz_convert('UTC')
        weather['timestamp'] = pd.to_datetime(weather['timestamp']).dt.tz_localize('UTC')

        # Merge datasets on timestamp using outer join
        merged_data = pd.merge(energy, sensor_data, on='timestamp', how='outer')
        merged_data = pd.merge(merged_data, weather, on='timestamp', how='outer')

        # Handle missing values (example: forward fill)
        merged_data.fillna(method='ffill', inplace=True)

        return merged_data
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None

def fill_missing_values(merged_data, column, target_col):
    try:
        reg = LinearRegression()

        # Training data: where target_col is not missing
        not_missing_mask = merged_data[target_col].notna()
        X_train = merged_data.loc[not_missing_mask, [column]].values.reshape(-1, 1)
        y_train = merged_data.loc[not_missing_mask, target_col].values

        # Fit regression model
        reg.fit(X_train, y_train)

        # Predict missing target_col values
        missing_mask = merged_data[target_col].isna()
        X_pred = merged_data.loc[missing_mask, [column]].values.reshape(-1, 1)
        merged_data.loc[missing_mask, target_col] = reg.predict(X_pred)

    except Exception as e:
        print(f"Error filling missing values for {target_col}: {e}")

def create_plots(merged_data):
    try:
        # Distribution plot for energy consumption
        plt.figure(figsize=(10, 6))
        sns.histplot(merged_data['building_energy_consumption_kwh'], bins=30, kde=True)
        plt.title('Distribution of Energy Consumption')
        plt.show()

        # Boxplot for energy consumption
        sns.boxplot(data=merged_data, x='building_energy_consumption_kwh')
        plt.title('Boxplot of Building Energy Consumption')
        plt.show()

        # Scatterplot to observe relationships
        sns.scatterplot(data=merged_data, x='Temperature', y='building_energy_consumption_kwh')
        plt.title('Energy Consumption vs Temperature')
        plt.show()

        # Autocorrelation plot
        pd.plotting.autocorrelation_plot(merged_data['building_energy_consumption_kwh'])
        plt.title('Autocorrelation of Energy Consumption')
        plt.show()
    except Exception as e:
        print(f"Error creating plots: {e}")

def feature_engineering(merged_data):
    try:
        # Extracting useful time-based features
        merged_data['hour'] = merged_data['timestamp'].dt.hour
        merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
        merged_data['month'] = merged_data['timestamp'].dt.month
        merged_data['week_of_year'] = merged_data['timestamp'].dt.isocalendar().week
        merged_data['is_weekend'] = merged_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Interaction features
        merged_data['temp_energy_interaction'] = merged_data['Temperature'] * merged_data['building_energy_consumption_kwh']
        merged_data['humidity_energy_interaction'] = merged_data['RelativeHumidity'] * merged_data['building_energy_consumption_kwh']

        # One-hot encoding for categorical features
        merged_data = pd.get_dummies(merged_data, columns=['site_id', 'tags_name'], drop_first=True)

        # Create lag features
        merged_data['lag_1'] = merged_data['building_energy_consumption_kwh'].shift(1)
        merged_data['lag_2'] = merged_data['building_energy_consumption_kwh'].shift(2)
        merged_data['lag_3'] = merged_data['building_energy_consumption_kwh'].shift(3)

        # Calculate rolling statistics
        merged_data['rolling_mean_3'] = merged_data['building_energy_consumption_kwh'].rolling(window=3).mean()
        merged_data['rolling_std_3'] = merged_data['building_energy_consumption_kwh'].rolling(window=3).std()

        return merged_data
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return None

def scale_and_split_data(merged_data):
    try:
        scaler = MinMaxScaler()
        numerical_cols = ['building_energy_consumption_kwh', 'RelativeHumidity', 'Temperature']
        merged_data[numerical_cols] = scaler.fit_transform(merged_data[numerical_cols])
        merged_data.dropna(inplace=True)

        # Define features (X) and target (y)
        X = merged_data.drop(['building_energy_consumption_kwh', 'timestamp'], axis=1)
        y = merged_data['building_energy_consumption_kwh']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test ,X, y
    except Exception as e:
        print(f"Error during scaling and data splitting: {e}")
        return None, None, None, None
    
#########################MODEL TRAINING CODE#############################

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
        energy_file = get_file_path('energy_data.csv')
        sensor_data_file = get_file_path('sensor_data.csv')
        weather_data_file = get_file_path('weather_data.csv')

        energy = load_data(energy_file)
        sensor_data = load_data(sensor_data_file)
        weather = load_data(weather_data_file)

        if energy is None or sensor_data is None or weather is None:
            print("Data loading failed. Exiting.")
            return

        merged_data = preprocess_data(energy, sensor_data, weather)
        if merged_data is None:
            print("Data preprocessing failed. Exiting.")
            return

        fill_missing_values(merged_data, 'building_energy_consumption_kwh', 'RelativeHumidity')
        fill_missing_values(merged_data, 'building_energy_consumption_kwh', 'Temperature')

        # create_plots(merged_data)

        merged_data = feature_engineering(merged_data)
        if merged_data is None:
            print("Feature engineering failed. Exiting.")
            return

        X_train, X_test, y_train, y_test,X,y = scale_and_split_data(merged_data)
        if X_train is None:
            print("Data scaling and splitting failed. Exiting.")
            return

        # Print the shapes of the resulting datasets
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

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
