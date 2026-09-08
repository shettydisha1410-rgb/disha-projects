# Weather-Based AQI Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("cleaned_dataset.csv")
df["season"] = df["season"].astype("category").cat.codes
df["day_of_week"] = df["day_of_week"].astype("category").cat.codes
df["aqi_category"] = df["aqi_category"].astype("category").cat.codes
features = ["temperature", "humidity", "wind_speed", "visibility", "month", "hour", "season", "day_of_week", "is_weekend"]
X, y = df[features], df["aqi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mse = mean_squared_error(y_test, prediction)
print(f"MAE: {mean_absolute_error(y_test, prediction):.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}")
print(f"R2 Score: {r2_score(y_test, prediction):.4f}")
pd.DataFrame({"Actual AQI": y_test.values[:15], "Predicted AQI": prediction[:15]}).to_csv("weather_prediction_results.csv", index=False)
