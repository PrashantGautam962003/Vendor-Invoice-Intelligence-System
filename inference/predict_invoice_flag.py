import os
import joblib
import pandas as pd

# 1. Dynamically find the folder where this python script lives (e.g., .../inference)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root, then down into the local models folder structure
# This translates perfectly to both Windows (C:\...) and Streamlit (Linux /mount/src/...)
MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "..",
    "Invoice_Flagging",
    "models",
    "predict_flagged_invoice.pkl"
)

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model
def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    input_data = {}