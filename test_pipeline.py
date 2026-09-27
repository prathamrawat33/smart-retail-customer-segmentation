# ============================================================
# SMART RETAIL PROJECT
# FINAL PIPELINE TESTS
# ============================================================

import os
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


# ============================================================
# FILE PATHS
# ============================================================

CLEANED_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "cleaned_data.csv"
)

CUSTOMER_CLUSTERS_PATH = os.path.join(
    PROCESSED_DIR,
    "customer_clusters.csv"
)

RULES_PATH = os.path.join(
    PROCESSED_DIR,
    "cluster_specific_association_rules.csv"
)

RECOMMENDATIONS_PATH = os.path.join(
    PROCESSED_DIR,
    "customer_recommendations.csv"
)


# ============================================================
# TEST 1
# CLEANED DATASET
# ============================================================

def test_cleaned_dataset_exists():

    assert os.path.exists(
        CLEANED_DATA_PATH
    ), "Cleaned dataset does not exist."


def test_cleaned_dataset_has_rows():

    df = pd.read_csv(
        CLEANED_DATA_PATH
    )

    assert len(df) > 0


def test_cleaned_dataset_has_required_columns():

    df = pd.read_csv(
        CLEANED_DATA_PATH
    )

    required_columns = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "TotalAmount"
    }

    assert required_columns.issubset(
        set(df.columns)
    )


# ============================================================
# TEST 2
# CUSTOMER CLUSTERS
# ============================================================

def test_customer_clusters_exists():

    assert os.path.exists(
        CUSTOMER_CLUSTERS_PATH
    )


def test_all_four_clusters_exist():

    df = pd.read_csv(
        CUSTOMER_CLUSTERS_PATH
    )

    actual_clusters = set(
        df["Cluster"]
        .dropna()
        .astype(int)
        .unique()
    )

    expected_clusters = {
        0,
        1,
        2,
        3
    }

    assert expected_clusters.issubset(
        actual_clusters
    )


def test_customer_ids_are_unique():

    df = pd.read_csv(
        CUSTOMER_CLUSTERS_PATH
    )

    assert (
        df["CustomerID"].is_unique
    )


# ============================================================
# TEST 3
# ASSOCIATION RULES
# ============================================================

def test_association_rules_exists():

    assert os.path.exists(
        RULES_PATH
    )


def test_association_rules_have_rows():

    df = pd.read_csv(
        RULES_PATH
    )

    assert len(df) > 0


def test_association_rules_columns():

    df = pd.read_csv(
        RULES_PATH
    )

    required_columns = {
        "Cluster",
        "Segment_Label",
        "antecedents_names",
        "consequents_names",
        "support",
        "confidence",
        "lift"
    }

    assert required_columns.issubset(
        set(df.columns)
    )


def test_association_rule_thresholds():

    df = pd.read_csv(
        RULES_PATH
    )

    assert (
        df["confidence"] >= 0.50
    ).all()

    assert (
        df["lift"] >= 1.20
    ).all()


# ============================================================
# TEST 4
# RECOMMENDATIONS
# ============================================================

def test_recommendations_exist():

    assert os.path.exists(
        RECOMMENDATIONS_PATH
    )


def test_recommendations_have_rows():

    df = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    assert len(df) > 0


def test_recommendation_columns():

    df = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    required_columns = {
        "CustomerID",
        "Cluster",
        "Segment_Label",
        "Rank",
        "Product",
        "Recommendation_Source"
    }

    assert required_columns.issubset(
        set(df.columns)
    )


# ============================================================
# TEST 5
# VALID CUSTOMER IDS
# ============================================================

def test_recommendation_customer_ids_are_valid():

    clusters = pd.read_csv(
        CUSTOMER_CLUSTERS_PATH
    )

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    valid_ids = set(
        clusters[
            "CustomerID"
        ]
    )

    recommended_ids = set(
        recommendations[
            "CustomerID"
        ]
    )

    assert recommended_ids.issubset(
        valid_ids
    )


# ============================================================
# TEST 6
# VALID RECOMMENDATION SOURCES
# ============================================================

def test_recommendation_sources():

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    valid_sources = {
        "Association Rule",
        "Segment Bestseller Fallback"
    }

    actual_sources = set(
        recommendations[
            "Recommendation_Source"
        ]
        .dropna()
        .unique()
    )

    assert actual_sources.issubset(
        valid_sources
    )


# ============================================================
# TEST 7
# NO DUPLICATE RECOMMENDATIONS
# ============================================================

def test_no_duplicate_customer_products():

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    duplicates = recommendations.duplicated(
        subset=[
            "CustomerID",
            "Product"
        ]
    )

    assert not duplicates.any()


# ============================================================
# TEST 8
# RECOMMENDATION RANKS
# ============================================================

def test_recommendation_ranks():

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    assert (
        recommendations["Rank"]
        .min()
        >= 1
    )

    assert (
        recommendations["Rank"]
        .max()
        <= 5
    )


# ============================================================
# TEST 9
# CLUSTER CONSISTENCY
# ============================================================

def test_cluster_values_are_valid():

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    valid_clusters = {
        0,
        1,
        2,
        3
    }

    actual_clusters = set(
        recommendations[
            "Cluster"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    assert actual_clusters.issubset(
        valid_clusters
    )


# ============================================================
# TEST 10
# RECOMMENDATION COUNT
# ============================================================

def test_recommendation_count():

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    assert len(
        recommendations
    ) > 0


# ============================================================
# TEST SUMMARY
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SMART RETAIL PIPELINE TESTS")
    print("=" * 70)

    print(
        "\nRun the tests using:"
    )

    print(
        "python -m pytest tests -v"
    )