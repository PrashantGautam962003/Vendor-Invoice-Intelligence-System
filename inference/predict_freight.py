import os
import joblib
import pandas as pd

# 1. Dynamically find the folder where this python script lives (e.g., .../inference)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root, then down into your models folder
MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "..",
    "models",
    "predict_freight_model.pkl"
)

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data: dict):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        The input features for the model in dictionary format.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the input data along with the predicted freight cost.
    """
    # 1. Load the trained model
    model = load_model()

    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df)
    return input_df

if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)

