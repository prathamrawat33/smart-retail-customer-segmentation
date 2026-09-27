# ============================================================
# CUSTOMER RECOMMENDATION ENGINE
# Smart Retail Customer Segmentation
# ============================================================

import os
import pandas as pd


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

TRANSACTIONS_PATH = os.path.join(
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
# RECOMMENDATION PARAMETERS
# ============================================================

MAX_RECOMMENDATIONS = 5

MIN_CONFIDENCE = 0.50

MIN_LIFT = 1.20


# ============================================================
# PRODUCT NAME NORMALIZATION
# ============================================================

def clean_product_name(value):

    if pd.isna(value):

        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace("  ", " ")
    )


# ============================================================
# PARSE ASSOCIATION RULE PRODUCTS
# ============================================================

def parse_product_set(value):

    if pd.isna(value):

        return set()

    value = str(value).strip()

    if not value:

        return set()

    if " , " in value:

        products = value.split(" , ")

    elif "," in value:

        products = value.split(",")

    else:

        products = [value]

    return {
        clean_product_name(product)
        for product in products
        if clean_product_name(product)
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CUSTOMER RECOMMENDATION ENGINE")
    print("=" * 70)

    print("\nLoading transactions...")

    transactions = pd.read_csv(
        TRANSACTIONS_PATH
    )

    print(
        "Transactions:",
        transactions.shape
    )

    print("\nLoading customer clusters...")

    customer_clusters = pd.read_csv(
        CUSTOMER_CLUSTERS_PATH
    )

    print(
        "Customer clusters:",
        customer_clusters.shape
    )

    print("\nLoading association rules...")

    rules = pd.read_csv(
        RULES_PATH
    )

    print(
        "Association rules:",
        rules.shape
    )

    return (
        transactions,
        customer_clusters,
        rules
    )


# ============================================================
# PREPARE TRANSACTIONS
# ============================================================

def prepare_transactions(
    transactions
):

    transactions = transactions.copy()

    transactions[
        "Description_Clean"
    ] = (
        transactions["Description"]
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

    transactions = transactions[
        transactions[
            "Description_Clean"
        ] != ""
    ].copy()

    return transactions


# ============================================================
# PREPARE RULES
# ============================================================

def prepare_rules(rules):

    rules = rules.copy()

    rules[
        "Antecedent_Set"
    ] = rules[
        "antecedents_names"
    ].apply(
        parse_product_set
    )

    rules[
        "Consequent_Set"
    ] = rules[
        "consequents_names"
    ].apply(
        parse_product_set
    )

    rules = rules[
        (rules["confidence"] >= MIN_CONFIDENCE)
        &
        (rules["lift"] >= MIN_LIFT)
    ].copy()

    return rules


# ============================================================
# CUSTOMER PROFILE
# ============================================================

def get_customer_profile(
    customer_id,
    customer_clusters
):

    profile = customer_clusters[
        customer_clusters[
            "CustomerID"
        ] == customer_id
    ]

    if profile.empty:

        return None

    return profile.iloc[0]


# ============================================================
# CUSTOMER PURCHASE HISTORY
# ============================================================

def get_customer_purchase_history(
    customer_id,
    transactions
):

    history = transactions[
        transactions[
            "CustomerID"
        ] == customer_id
    ].copy()

    return history


# ============================================================
# FIND MATCHING RULES
# ============================================================

def find_matching_rules(
    customer_products,
    cluster,
    rules
):

    cluster_rules = rules[
        rules["Cluster"] == cluster
    ].copy()

    if cluster_rules.empty:

        return pd.DataFrame()

    matching_rules = []

    for _, rule in cluster_rules.iterrows():

        antecedent = rule[
            "Antecedent_Set"
        ]

        if antecedent.issubset(
            customer_products
        ):

            matching_rules.append(
                rule
            )

    if not matching_rules:

        return pd.DataFrame()

    return pd.DataFrame(
        matching_rules
    )


# ============================================================
# GENERATE RULE RECOMMENDATIONS
# ============================================================

def generate_rule_recommendations(
    customer_products,
    matching_rules
):

    recommendations = []

    if matching_rules.empty:

        return pd.DataFrame()

    for _, rule in matching_rules.iterrows():

        consequents = rule[
            "Consequent_Set"
        ]

        for product in consequents:

            if product not in customer_products:

                recommendations.append({

                    "Product": product,

                    "Support": rule[
                        "support"
                    ],

                    "Confidence": rule[
                        "confidence"
                    ],

                    "Lift": rule[
                        "lift"
                    ],

                    "Recommendation_Source":
                        "Association Rule"
                })

    if not recommendations:

        return pd.DataFrame()

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# RANK RECOMMENDATIONS
# ============================================================

def rank_recommendations(
    recommendations,
    n=5
):

    if recommendations.empty:

        return pd.DataFrame()

    ranked = (
        recommendations
        .sort_values(
            by=[
                "Confidence",
                "Lift",
                "Support"
            ],
            ascending=False
        )
        .drop_duplicates(
            subset=["Product"]
        )
        .head(n)
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "Rank",
        range(
            1,
            len(ranked) + 1
        )
    )

    return ranked


# ============================================================
# SEGMENT BESTSELLERS
# ============================================================

def create_segment_bestsellers(
    transactions,
    customer_clusters
):

    data = transactions.merge(
        customer_clusters[
            [
                "CustomerID",
                "Cluster"
            ]
        ],
        on="CustomerID",
        how="inner"
    )

    data[
        "Description_Clean"
    ] = (
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

    data = data[
        data[
            "Description_Clean"
        ] != ""
    ].copy()

    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    )

    data = data[
        data["Quantity"].notna()
    ]

    segment_bestsellers = {}

    for cluster in sorted(
        data["Cluster"].unique()
    ):

        cluster_data = data[
            data["Cluster"] == cluster
        ]

        bestseller_table = (
            cluster_data
            .groupby(
                "Description_Clean"
            )["Quantity"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        segment_bestsellers[
            int(cluster)
        ] = (
            bestseller_table
            .head(20)
            .index
            .tolist()
        )

    return segment_bestsellers


# ============================================================
# FALLBACK RECOMMENDATIONS
# ============================================================

def generate_fallback_recommendations(
    customer_products,
    cluster,
    segment_bestsellers,
    n=5
):

    products = (
        segment_bestsellers
        .get(cluster, [])
    )

    recommendations = []

    for product in products:

        if product not in customer_products:

            recommendations.append({

                "Product": product,

                "Support": None,

                "Confidence": None,

                "Lift": None,

                "Recommendation_Source":
                    "Segment Bestseller Fallback"
            })

        if len(
            recommendations
        ) >= n:

            break

    if not recommendations:

        return pd.DataFrame()

    result = pd.DataFrame(
        recommendations
    )

    result.insert(
        0,
        "Rank",
        range(
            1,
            len(result) + 1
        )
    )

    return result


# ============================================================
# COMPLETE CUSTOMER RECOMMENDATION
# ============================================================

def generate_customer_recommendations(
    customer_id,
    transactions,
    customer_clusters,
    rules,
    segment_bestsellers,
    n=5
):

    # --------------------------------------------------------
    # Customer profile
    # --------------------------------------------------------

    profile = get_customer_profile(
        customer_id,
        customer_clusters
    )

    if profile is None:

        return pd.DataFrame()

    cluster = int(
        profile["Cluster"]
    )

    segment_label = (
        profile["Segment_Label"]
    )

    # --------------------------------------------------------
    # Purchase history
    # --------------------------------------------------------

    history = (
        get_customer_purchase_history(
            customer_id,
            transactions
        )
    )

    customer_products = set(
        history[
            "Description_Clean"
        ]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Association rules
    # --------------------------------------------------------

    matching_rules = (
        find_matching_rules(
            customer_products,
            cluster,
            rules
        )
    )

    recommendations = (
        generate_rule_recommendations(
            customer_products,
            matching_rules
        )
    )

    recommendations = (
        rank_recommendations(
            recommendations,
            n=n
        )
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if recommendations.empty:

        recommendations = (
            generate_fallback_recommendations(
                customer_products,
                cluster,
                segment_bestsellers,
                n=n
            )
        )

    if recommendations.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Add customer information
    # --------------------------------------------------------

    recommendations.insert(
        0,
        "CustomerID",
        customer_id
    )

    recommendations.insert(
        1,
        "Cluster",
        cluster
    )

    recommendations.insert(
        2,
        "Segment_Label",
        segment_label
    )

    return recommendations


# ============================================================
# BATCH RECOMMENDATIONS
# ============================================================

def generate_batch_recommendations(
    transactions,
    customer_clusters,
    rules,
    segment_bestsellers,
    customer_ids=None,
    n=5
):

    if customer_ids is None:

        customer_ids = (
            customer_clusters[
                "CustomerID"
            ]
            .dropna()
            .unique()
        )

    all_recommendations = []

    total = len(
        customer_ids
    )

    for index, customer_id in enumerate(
        customer_ids,
        start=1
    ):

        recommendations = (
            generate_customer_recommendations(
                customer_id,
                transactions,
                customer_clusters,
                rules,
                segment_bestsellers,
                n=n
            )
        )

        if not recommendations.empty:

            all_recommendations.append(
                recommendations
            )

        if index % 100 == 0:

            print(
                f"Processed {index}/{total} customers..."
            )

    if not all_recommendations:

        return pd.DataFrame()

    return pd.concat(
        all_recommendations,
        ignore_index=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        transactions,
        customer_clusters,
        rules
    ) = load_data()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    transactions = (
        prepare_transactions(
            transactions
        )
    )

    rules = prepare_rules(
        rules
    )

    print(
        "\nFiltered association rules:",
        len(rules)
    )

    # --------------------------------------------------------
    # Create segment bestseller engine
    # --------------------------------------------------------

    segment_bestsellers = (
        create_segment_bestsellers(
            transactions,
            customer_clusters
        )
    )

    print(
        "\nSegment bestseller engine created."
    )

    # --------------------------------------------------------
    # Test one customer from each cluster
    # --------------------------------------------------------

    test_customers = (
        customer_clusters
        .groupby("Cluster")
        ["CustomerID"]
        .first()
        .tolist()
    )

    print(
        "\nTesting recommendation engine..."
    )

    for customer_id in test_customers:

        result = (
            generate_customer_recommendations(
                customer_id,
                transactions,
                customer_clusters,
                rules,
                segment_bestsellers,
                n=5
            )
        )

        print(
            "\n" + "-" * 70
        )

        print(
            "Customer:",
            customer_id
        )

        if result.empty:

            print(
                "No recommendations."
            )

        else:

            print(
                result[
                    [
                        "CustomerID",
                        "Cluster",
                        "Segment_Label",
                        "Rank",
                        "Product",
                        "Confidence",
                        "Lift",
                        "Recommendation_Source"
                    ]
                ].to_string(
                    index=False
                )
            )

    # --------------------------------------------------------
    # Batch evaluation
    # --------------------------------------------------------

    print(
        "\nGenerating recommendations for 100 customers..."
    )

    sample_customers = (
        customer_clusters[
            "CustomerID"
        ]
        .dropna()
        .drop_duplicates()
        .head(100)
        .tolist()
    )

    recommendation_df = (
        generate_batch_recommendations(
            transactions,
            customer_clusters,
            rules,
            segment_bestsellers,
            customer_ids=sample_customers,
            n=5
        )
    )

    # --------------------------------------------------------
    # Save recommendations
    # --------------------------------------------------------

    recommendation_df.to_csv(
        RECOMMENDATIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RECOMMENDATION ENGINE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Customers evaluated:",
        len(sample_customers)
    )

    print(
        "Customers receiving recommendations:",
        recommendation_df[
            "CustomerID"
        ].nunique()
    )

    print(
        "Total recommendations:",
        len(recommendation_df)
    )

    print(
        "Average recommendations/customer:",
        round(
            len(recommendation_df)
            /
            len(sample_customers),
            2
        )
    )

    print(
        "\nRecommendation sources:"
    )

    print(
        recommendation_df[
            "Recommendation_Source"
        ]
        .value_counts()
    )

    print(
        "\nRecommendations saved to:"
    )

    print(
        RECOMMENDATIONS_PATH
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RECOMMENDATION ENGINE COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()