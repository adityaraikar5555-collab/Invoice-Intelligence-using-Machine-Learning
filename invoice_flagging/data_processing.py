import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_invoice_data():
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
        COALESCE(pa.total_brands, 0) AS total_brands,
        COALESCE(pa.total_item_quantity, 0) AS total_item_quantity,
        COALESCE(pa.total_item_dollars, 0) AS total_item_dollars,
        COALESCE(pa.avg_receiving_delay, 0) AS avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    db_path = os.path.join(os.path.dirname(__file__), "..", "Data_Project", "inventory.db")
    db_path = os.path.abspath(db_path)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def apply_labels(df):
    df = df.copy()

    df["flag_invoice"] = (
        ((df["invoice_dollars"] - df["total_item_dollars"]).abs() > 5) |
        (df["avg_receiving_delay"] > 10)
    ).astype(int)

    return df


def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled


def preprocess_pipeline(features, target):
    df = load_invoice_data()
    df = apply_labels(df)

    X_train, X_test, y_train, y_test = split_data(df, features, target)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    return df, X_train_scaled, X_test_scaled, y_train, y_test