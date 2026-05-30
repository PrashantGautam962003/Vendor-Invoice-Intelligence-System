import sqlite3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_invoice_data():
    # Establish connection to the SQLite database
    conn = sqlite3.connect(
        r'C:\Users\prash\machine_learning_project\data\inventory.db'
    )

    # SQL query utilizing a Common Table Expression (CTE) to aggregate purchase order records
    query = """
    WITH purchase_agg AS (
        SELECT 
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT 
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa 
        ON vi.PONumber = pa.PONumber
    """

    # Execute query and load results into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    return df


def create_invoice_risk_label(row):
    # Rule 1: Check for invoice total mismatch with item-level total
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1

    # Rule 2: Check for abnormally high receiving delay
    if row["avg_receiving_delay"] > 10:
        return 1

    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def split_data(df, features, target):
    x = df[features]
    y = df[target]

    return train_test_split(x, y, test_size=0.2, random_state=42)


def scale_features(x_train, x_test, scaler_path="models/scaler.pkl"):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Save the fitted scaler to the specified path using joblib
    joblib.dump(scaler, 'models/scaler.pkl')

    return x_train_scaled, x_test_scaled

