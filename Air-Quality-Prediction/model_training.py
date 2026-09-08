# AQI Research Project - Model Training and Evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("cleaned_dataset.csv")
features = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed", "visibility"]
X, y = df[features], df["aqi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate(model, Xtrain, Xtest, name):
    model.fit(Xtrain, y_train)
    prediction = model.predict(Xtest)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.6f}")
    return [name, mae, mse, rmse, r2]

results = [
    evaluate(LinearSVR(C=1.0, epsilon=0.1, random_state=42, max_iter=10000), X_train_scaled, X_test_scaled, "Support Vector Regression (LinearSVR)"),
    evaluate(DecisionTreeRegressor(random_state=42), X_train, X_test, "Decision Tree Regression"),
    evaluate(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), X_train, X_test, "Random Forest Regression")
]

pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"]).to_csv("model_results.csv", index=False)
print("Results saved as model_results.csv")
