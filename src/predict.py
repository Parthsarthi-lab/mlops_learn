import joblib
import pandas as pd
# Load saved model
model = joblib.load(
"models/linear_reg_model.pkl"
)
# New observation
new_data = pd.DataFrame({
"TV": [35],
"Radio": [50000],
"Newspaper": [8]
})
# Prediction
prediction = model.predict(new_data)
print("Prediction:", prediction[0])
