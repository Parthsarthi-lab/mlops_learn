import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("D:\\BOOKS FOR TEACHING\\MLOPS_2026_Upgrade\\mlops_learn\\data\\data.csv")
df.head()

X = df.drop(columns=["Sales"])
y = df["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create model
model = LinearRegression()
# Train model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)
# Evaluate
rmse = root_mean_squared_error(y_test, predictions)
print("RMSE:", rmse)


# Save model
joblib.dump(model, "models/linear_reg_model.pkl")
print("Model saved successfully.")