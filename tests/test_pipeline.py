import os
import pandas as pd
import mlflow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_data_schema():
    data_path = os.path.join(BASE_DIR, "data", "data.csv")
    assert os.path.exists(data_path), "Data file data.csv not found!"
    df = pd.read_csv(data_path)
    expected_cols = {"TV", "Radio", "Newspaper", "Sales"}
    assert expected_cols.issubset(set(df.columns)), f"Missing required columns in dataset"
    assert df[["TV", "Radio", "Newspaper"]].isnull().sum().sum() == 0, "Null values found in features"

def test_champion_model_loading_and_prediction():
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    # Verify champion alias exists and can predict
    model_uri = "models:/Sales_Prediction_Model@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    
    sample_input = pd.DataFrame([[100.0, 25.0, 10.0]], columns=["TV", "Radio", "Newspaper"])
    prediction = model.predict(sample_input)
    
    assert len(prediction) == 1, "Prediction output shape invalid"
    assert float(prediction[0]) > 0, "Prediction value is unexpected or non-positive"