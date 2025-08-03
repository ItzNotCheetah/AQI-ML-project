import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
train_X, val_X, train_y, val_y = train_test_split(X,y, test_size = 0.2, random_state = 0)

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

# Visualizing the feature importance correlation using a heatmap
corr_matrix = aq_data.select_dtypes(include='number').corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            linewidths=0.5, linecolor='gray')

plt.title("Correlation Heatmap of Air Quality Features", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualizing the predictions vs actual values using residuals
residuals = val_y - val_predictions

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor="black", color="salmon")

plt.title("Residuals Distribution")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
