"""
Data Preprocessing Module
-------------------------

Handles:
1. Loading the Online Retail dataset
2. Cleaning transaction data
3. Creating TotalAmount
4. Creating customer-level RFM and behavioral features
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data(path=RAW_DATA_PATH):
    """
    Load the raw Online Retail dataset.

    Parameters
    ----------
    path : Path or str
        Location of the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw transaction dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="latin1"
    )

    return df


# ============================================================
# CLEAN TRANSACTIONS
# ============================================================

def clean_transactions(df):
    """
    Clean retail transaction data.

    Operations:
    - Remove missing CustomerID records
    - Remove cancellation invoices
    - Remove invalid quantities
    - Remove invalid prices
    - Remove invalid transaction records
    - Create TotalAmount
    - Remove duplicate rows
    """

    data = df.copy()

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    data.columns = (
        data.columns
        .str.strip()
    )

    # --------------------------------------------------------
    # Convert data types
    # --------------------------------------------------------

    if "InvoiceDate" in data.columns:

        data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    
        )

    if "Quantity" in data.columns:

        data["Quantity"] = pd.to_numeric(
            data["Quantity"],
            errors="coerce"
        )

    if "UnitPrice" in data.columns:

        data["UnitPrice"] = pd.to_numeric(
            data["UnitPrice"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove missing CustomerID
    # --------------------------------------------------------

    if "CustomerID" in data.columns:

        data = data[
            data["CustomerID"].notna()
        ].copy()

    # --------------------------------------------------------
    # Remove cancellation invoices
    # --------------------------------------------------------

    if "InvoiceNo" in data.columns:

        data["InvoiceNo"] = (
            data["InvoiceNo"]
            .astype(str)
            .str.strip()
        )

        data = data[
            ~data["InvoiceNo"]
            .str.upper()
            .str.startswith("C")
        ].copy()

    # --------------------------------------------------------
    # Remove invalid quantities
    # --------------------------------------------------------

    if "Quantity" in data.columns:

        data = data[
            data["Quantity"] > 0
        ].copy()

    # --------------------------------------------------------
    # Remove invalid prices
    # --------------------------------------------------------

    if "UnitPrice" in data.columns:

        data = data[
            data["UnitPrice"] > 0
        ].copy()

    # --------------------------------------------------------
    # Remove invalid dates
    # --------------------------------------------------------

    if "InvoiceDate" in data.columns:

        data = data[
            data["InvoiceDate"].notna()
        ].copy()

    # --------------------------------------------------------
    # Calculate TotalAmount
    # --------------------------------------------------------

    data["TotalAmount"] = (
        data["Quantity"] *
        data["UnitPrice"]
    )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    data = data.drop_duplicates()

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    data = data.reset_index(drop=True)

    return data


# ============================================================
# BUILD CUSTOMER FEATURES
# ============================================================

def build_customer_features(df):
    """
    Create customer-level behavioral features.

    Features:
    - Recency
    - Frequency
    - Monetary
    - TotalQuantity
    - UniqueProducts
    - AOV
    """

    data = df.copy()

    # --------------------------------------------------------
    # Reference date
    # --------------------------------------------------------

    reference_date = (
        data["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------------
    # Customer-level aggregation
    # --------------------------------------------------------

    customer_features = (
        data
        .groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (
                    reference_date - x.max()
                ).days
            ),

            Frequency=(
                "InvoiceNo",
                "nunique"
            ),

            Monetary=(
                "TotalAmount",
                "sum"
            ),

            TotalQuantity=(
                "Quantity",
                "sum"
            ),

            UniqueProducts=(
                "StockCode",
                "nunique"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Average Order Value
    # --------------------------------------------------------

    customer_features["AOV"] = (
        customer_features["Monetary"]
        /
        customer_features["Frequency"]
    )

    return customer_features


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing(
    input_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH
):
    """
    Run the complete preprocessing pipeline.

    Returns
    -------
    cleaned_data : pandas.DataFrame
    customer_features : pandas.DataFrame
    """

    print("=" * 70)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # Load
    raw_data = load_raw_data(input_path)

    print(
        f"Raw dataset shape: {raw_data.shape}"
    )

    # Clean
    cleaned_data = clean_transactions(
        raw_data
    )

    print(
        f"Cleaned dataset shape: {cleaned_data.shape}"
    )

    # Customer features
    customer_features = build_customer_features(
        cleaned_data
    )

    print(
        f"Customer feature shape: "
        f"{customer_features.shape}"
    )

    # Save cleaned data
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned_data.to_csv(
        output_path,
        index=False
    )

    print(
        f"Cleaned data saved to: "
        f"{output_path}"
    )

    print("=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    return (
        cleaned_data,
        customer_features
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    cleaned_data, customer_features = (
        run_preprocessing()
    )

    print()
    print("First 5 cleaned transactions:")
    print(
        cleaned_data.head()
    )

    print()
    print("First 5 customer features:")
    print(
        customer_features.head()
    )