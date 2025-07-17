import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Loading and reading the dataset
data_file_path = 'global_air_pollution_dataset.csv'
aq_data = pd.read_csv(data_file_path)
aq_data = aq_data.dropna(axis = 0)
# Selecting features and the target variable
aq_features = ['CO AQI Value', 'Ozone AQI Value', 'PM2.5 AQI Value', 'NO2 AQI Value']
X = aq_data[aq_features]
y = aq_data['AQI Value']

# Splitting the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state = 0)

aq_forest_model = RandomForestRegressor(random_state= 1)
aq_forest_model.fit(train_X, train_y)

# Make predictions for the first 5 rows
print("\n--- Predicting AQI for first 5 rows ---")
print("Input features:")
print(X.head())
print("\nPredicted AQI Values:")
print(aq_forest_model.predict(X.head()))

# Predict on validation set
val_predictions = aq_forest_model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)

# Show evaluation results
print("\n--- Validation Results ---")
print(f"Mean Absolute Error: {mae:.2f}")
print("\nFirst 5 validation predictions:")
print(val_predictions[:5])
print("\nActual AQI values:")
print(val_y[:5])



    
