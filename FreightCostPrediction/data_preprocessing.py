import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """Load vendor invoice data from SQLite database."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")

    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM vendor_invoice"
        df = pd.read_sql_query(query, conn)

    return df


def prepare_features(df: pd.DataFrame):
    """Prepare input features and target."""
    df = df.copy()

    # Fill missing values
    df["Dollars"] = df["Dollars"].fillna(0)
    df["Freight"] = df["Freight"].fillna(0)

    # Features
    X = df[["Dollars"]]
    y = df["Freight"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )