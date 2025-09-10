import pandas as pd
import numpy as np

# The canonical column order used by the training pipeline:
FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET = "target"

def read_heart_csv(path: str) -> pd.DataFrame:
    """
    Reads the heart dataset CSV and normalizes columns.
    Expects at least FEATURES + TARGET to be present.
    - If 'num' exists (common in UCI versions) it is renamed to 'target'.
    - Columns are coerced to numeric; non-numeric values become NaN (imputed later).
    """
    df = pd.read_csv(path)
    # Attempt to standardize lowercase names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map 'num' -> 'target' if needed
    if "num" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"num": "target"})

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Coerce to numeric (errors='coerce' will create NaNs we later impute)
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop exact duplicate rows
    - Clip numeric outliers to 0.5/99.5 percentiles
    - Median-impute numeric NaNs
    """
    df = df.drop_duplicates()

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            lo, hi = np.nanpercentile(df[c], [0.5, 99.5])
            df[c] = df[c].clip(lo, hi)

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].median()
            df[c] = df[c].fillna(med)

    return df

def train_val_test_split(df: pd.DataFrame, val_size=0.2, test_size=0.1, random_state=42):
    """Stratified split into train, val, test."""
    from sklearn.model_selection import train_test_split
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size+test_size), stratify=y, random_state=random_state
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
