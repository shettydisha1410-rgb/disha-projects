# AQI Research Project - Feature Importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("cleaned_dataset.csv")
features = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed", "visibility"]
X, y = df[features], df["aqi"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
print(importance)
plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Feature Importance using Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("13_Feature_Importance.png")
plt.show()
