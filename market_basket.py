# ============================================================
# CLUSTER-SPECIFIC MARKET BASKET ANALYSIS
# Smart Retail Customer Segmentation
# ============================================================

import os

import pandas as pd

from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules
)


# ============================================================
# PATH CONFIGURATION
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

CLEANED_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "cleaned_data.csv"
)

CUSTOMER_CLUSTERS_PATH = os.path.join(
    PROCESSED_DIR,
    "customer_clusters.csv"
)

RULES_OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "cluster_specific_association_rules.csv"
)

SUMMARY_OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "segment_rule_summary.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.50
MIN_LIFT = 1.20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading cleaned transactions...")

    transactions = pd.read_csv(
        CLEANED_DATA_PATH
    )

    print(
        "Transactions shape:",
        transactions.shape
    )

    print(
        "\nLoading customer clusters..."
    )

    customer_clusters = pd.read_csv(
        CUSTOMER_CLUSTERS_PATH
    )

    print(
        "Customer cluster shape:",
        customer_clusters.shape
    )

    return (
        transactions,
        customer_clusters
    )


# ============================================================
# PREPARE TRANSACTIONS
# ============================================================

def prepare_transactions(
    transactions,
    customer_clusters
):

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "CustomerID",
        "Quantity"
    ]

    data = transactions[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Clean product descriptions
    # --------------------------------------------------------

    data["Description"] = (
        data["Description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )

    # --------------------------------------------------------
    # Remove empty products
    # --------------------------------------------------------

    data = data[
        data["Description"] != ""
    ].copy()

    # --------------------------------------------------------
    # Merge customer clusters
    # --------------------------------------------------------

    cluster_columns = [
        "CustomerID",
        "Cluster",
        "Segment_Label"
    ]

    data = data.merge(
        customer_clusters[
            cluster_columns
        ],
        on="CustomerID",
        how="inner"
    )

    # --------------------------------------------------------
    # Remove invalid cluster rows
    # --------------------------------------------------------

    data = data[
        data["Cluster"].notna()
    ].copy()

    data["Cluster"] = (
        data["Cluster"]
        .astype(int)
    )

    print(
        "\nTransactions after cluster mapping:",
        data.shape
    )

    return data


# ============================================================
# CREATE BASKET MATRIX
# ============================================================

def create_basket(
    cluster_data
):

    # --------------------------------------------------------
    # Invoice × Product matrix
    # --------------------------------------------------------

    basket = (
        cluster_data
        .groupby(
            [
                "InvoiceNo",
                "Description"
            ]
        )["Quantity"]
        .sum()
        .unstack(
            fill_value=0
        )
    )

    # --------------------------------------------------------
    # Convert quantities into binary presence
    # --------------------------------------------------------

    basket = (
        basket
        .gt(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Remove single-product baskets
    # --------------------------------------------------------

    basket = basket[
        basket.sum(axis=1) >= 2
    ]

    return basket


# ============================================================
# FP-GROWTH
# ============================================================

def mine_frequent_itemsets(
    basket
):

    if basket.empty:

        return pd.DataFrame(
            columns=[
                "support",
                "itemsets"
            ]
        )

    frequent_itemsets = fpgrowth(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    return frequent_itemsets


# ============================================================
# ASSOCIATION RULES
# ============================================================

def generate_rules(
    frequent_itemsets
):

    if frequent_itemsets.empty:

        return pd.DataFrame()

    if len(frequent_itemsets) < 2:

        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if rules.empty:

        return rules

    rules = rules[
        rules["lift"] >= MIN_LIFT
    ].copy()

    return rules


# ============================================================
# CONVERT SETS TO PRODUCT NAMES
# ============================================================

def convert_rule_sets(
    rules
):

    if rules.empty:

        return rules

    rules = rules.copy()

    rules[
        "antecedents_names"
    ] = rules[
        "antecedents"
    ].apply(
        lambda x:
        " , ".join(
            sorted(x)
        )
    )

    rules[
        "consequents_names"
    ] = rules[
        "consequents"
    ].apply(
        lambda x:
        " , ".join(
            sorted(x)
        )
    )

    return rules


# ============================================================
# MINE RULES FOR ONE CLUSTER
# ============================================================

def mine_cluster_rules(
    cluster_data,
    cluster_id,
    segment_label
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"CLUSTER {cluster_id} "
        f"- {segment_label}"
    )

    print(
        "-" * 70
    )

    basket = create_basket(
        cluster_data
    )

    print(
        "Basket shape:",
        basket.shape
    )

    if basket.empty:

        print(
            "No usable baskets."
        )

        return pd.DataFrame()

    frequent_itemsets = (
        mine_frequent_itemsets(
            basket
        )
    )

    print(
        "Frequent itemsets:",
        len(frequent_itemsets)
    )

    if frequent_itemsets.empty:

        return pd.DataFrame()

    rules = generate_rules(
        frequent_itemsets
    )

    print(
        "Association rules:",
        len(rules)
    )

    if rules.empty:

        return pd.DataFrame()

    rules = convert_rule_sets(
        rules
    )

    # --------------------------------------------------------
    # Add cluster information
    # --------------------------------------------------------

    rules[
        "Cluster"
    ] = cluster_id

    rules[
        "Segment_Label"
    ] = segment_label

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    rules = rules[
        [
            "Cluster",
            "Segment_Label",
            "antecedents_names",
            "consequents_names",
            "support",
            "confidence",
            "lift"
        ]
    ].copy()

    rules = rules.sort_values(
        [
            "lift",
            "confidence"
        ],
        ascending=False
    )

    return rules


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)

    print(
        "CLUSTER-SPECIFIC MARKET BASKET ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        transactions,
        customer_clusters
    ) = load_data()

    # --------------------------------------------------------
    # Prepare transactions
    # --------------------------------------------------------

    transactions = prepare_transactions(
        transactions,
        customer_clusters
    )

    # --------------------------------------------------------
    # Mine rules for each cluster
    # --------------------------------------------------------

    all_rules = []

    cluster_info = (
        transactions[
            [
                "Cluster",
                "Segment_Label"
            ]
        ]
        .drop_duplicates()
        .sort_values("Cluster")
    )

    for _, row in cluster_info.iterrows():

        cluster_id = int(
            row["Cluster"]
        )

        segment_label = (
            row["Segment_Label"]
        )

        cluster_data = transactions[
            transactions["Cluster"]
            == cluster_id
        ]

        rules = mine_cluster_rules(
            cluster_data,
            cluster_id,
            segment_label
        )

        if not rules.empty:

            all_rules.append(
                rules
            )

    # --------------------------------------------------------
    # Combine rules
    # --------------------------------------------------------

    if all_rules:

        final_rules = pd.concat(
            all_rules,
            ignore_index=True
        )

    else:

        final_rules = pd.DataFrame(
            columns=[
                "Cluster",
                "Segment_Label",
                "antecedents_names",
                "consequents_names",
                "support",
                "confidence",
                "lift"
            ]
        )

    # --------------------------------------------------------
    # Save rules
    # --------------------------------------------------------

    final_rules.to_csv(
        RULES_OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = (
        final_rules
        .groupby(
            [
                "Cluster",
                "Segment_Label"
            ]
        )
        .agg(

            Rule_Count=(
                "lift",
                "count"
            ),

            Average_Support=(
                "support",
                "mean"
            ),

            Average_Confidence=(
                "confidence",
                "mean"
            ),

            Average_Lift=(
                "lift",
                "mean"
            ),

            Maximum_Lift=(
                "lift",
                "max"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "CLUSTER RULE SUMMARY"
    )

    print("=" * 70)

    if summary.empty:

        print(
            "No association rules generated."
        )

    else:

        print(
            summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Top rules
    # --------------------------------------------------------

    if not final_rules.empty:

        print("\n" + "=" * 70)

        print(
            "TOP 20 ASSOCIATION RULES"
        )

        print("=" * 70)

        display_columns = [
            "Cluster",
            "Segment_Label",
            "antecedents_names",
            "consequents_names",
            "support",
            "confidence",
            "lift"
        ]

        print(
            final_rules[
                display_columns
            ]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "MARKET BASKET PIPELINE COMPLETED"
    )

    print("=" * 70)

    print(
        "\nRules saved to:"
    )

    print(
        RULES_OUTPUT_PATH
    )

    print(
        "\nSummary saved to:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()