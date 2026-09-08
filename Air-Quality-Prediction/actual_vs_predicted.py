# Actual vs Predicted AQI
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("cleaned_dataset.csv")
df["season"] = df["season"].astype("category").cat.codes
df["day_of_week"] = df["day_of_week"].astype("category").cat.codes
features = ["temperature", "humidity", "wind_speed", "visibility", "month", "hour", "season", "day_of_week", "is_weekend"]
X, y = df[features], df["aqi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, prediction, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("14_Actual_vs_Predicted.png")
plt.show()
