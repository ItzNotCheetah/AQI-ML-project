🌍 Air Quality Index Prediction Using Random Forest

This project uses a Random Forest Regressor to predict the overall Air Quality Index (AQI) based on concentrations of key air pollutants: Carbon Monoxide (CO), Ozone (O₃), PM2.5, and Nitrogen Dioxide (NO₂). 
The dataset is sourced from a global air pollution database containing real-world measurements collected from cities worldwide.

📊 Project Workflow
  Data Cleaning: Removed rows with missing values for a clean, consistent dataset.
  Feature Selection: Selected pollutant values (CO, Ozone, PM2.5, NO2) as predictors.
  Model Training: Trained a Random Forest Regressor on the processed data using an 80/20 train-test split.
  Evaluation: Assessed model performance using Mean Absolute Error (MAE).
  
📈 Visualizations
  🔧 Feature Importance Bar Plot
      This graph shows how much each pollutant contributes to predicting the AQI. A higher importance score indicates that the feature has more influence on the model’s predictions.
      Example: If PM2.5 has the highest bar, it's the most significant pollutant affecting AQI.
  📉 Residuals Histogram
      The residuals (difference between actual and predicted AQI) are plotted to visualize the model's error distribution.
      A histogram centered around zero with a narrow spread indicates accurate predictions with low bias.
  📐 Correlation Heatmap
    This heatmap displays the pairwise correlations between all numerical features. Strong correlations (positive or negative) help understand relationships between pollutants and AQI.
    For instance, a high positive correlation between PM2.5 and AQI suggests PM2.5 levels greatly impact overall air quality.
    
📌 Key Results
  Model Used: Random Forest Regressor
  MAE (Mean Absolute Error): ~0.15
  Top Feature: (Usually) PM2.5 AQI Value
