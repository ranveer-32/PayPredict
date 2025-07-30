# src/train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print(" Starting model training...")

X = np.load("E:\\payPredict\\data\\X.py.npy")
y = np.load("E:\\payPredict\\data\\y.py.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Model trained! MSE: {mse:.2f}, R²: {r2:.2f}")

joblib.dump(model, "E:\\payPredict\\models\\salary_model.pkl")
