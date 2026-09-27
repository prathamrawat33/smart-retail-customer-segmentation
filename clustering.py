# ============================================================
# CUSTOMER CLUSTERING PIPELINE
# Smart Retail Customer Segmentation
# ============================================================

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

CLEANED_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "cleaned_data.csv"
)

CUSTOMER_FEATURES_PATH = os.path.join(
    PROCESSED_DIR,
    "customer_features_engineered.csv"
)

SCALED_FEATURES_PATH = os.path.join(
    PROCESSED_DIR,
    "customer_clustering_features_scaled.csv"
)

CUSTOMER_CLUSTERS_PATH = os.path.join(
    PROCESSED_DIR,
    "customer_clusters.csv"
)

SEGMENT_SUMMARY_PATH = os.path.join(
    PROCESSED_DIR,
    "segment_summary.csv"
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned_data():

    if not os.path.exists(CLEANED_DATA_PATH):

        raise FileNotFoundError(
            f"Cleaned dataset not found:\n"
            f"{CLEANED_DATA_PATH}"
        )

    df = pd.read_csv(
        CLEANED_DATA_PATH,
        parse_dates=["InvoiceDate"]
    )

    return df


# ============================================================
# CREATE CUSTOMER FEATURES
# ============================================================

def create_customer_features(df):

    reference_date = (
        df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    customer_features = df.groupby(
        "CustomerID"
    ).agg(

        Recency=(
            "InvoiceDate",
            lambda x:
            (reference_date - x.max()).days
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

    customer_features["AOV"] = (
        customer_features["Monetary"]
        /
        customer_features["Frequency"]
    )

    customer_features = (
        customer_features
        .reset_index()
    )

    return customer_features


# ============================================================
# LOG TRANSFORMATION
# ============================================================

def transform_features(customer_features):

    skewed_features = [
        "Frequency",
        "Monetary",
        "AOV",
        "TotalQuantity",
        "UniqueProducts"
    ]

    for feature in skewed_features:

        customer_features[
            f"Log{feature}"
        ] = np.log1p(
            customer_features[feature]
        )

    return customer_features


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(customer_features):

    clustering_features = [

        "Recency",

        "LogFrequency",

        "LogMonetary",

        "LogAOV",

        "LogTotalQuantity",

        "LogUniqueProducts"
    ]

    X = customer_features[
        clustering_features
    ].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=clustering_features
    )

    return scaled_df, X_scaled, clustering_features


# ============================================================
# EVALUATE K-MEANS
# ============================================================

def evaluate_kmeans(X_scaled):

    results = []

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X_scaled
        )

        inertia = model.inertia_

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        results.append({

            "K": k,

            "Inertia": inertia,

            "Silhouette": silhouette
        })

    return pd.DataFrame(results)


# ============================================================
# FINAL K-MEANS
# ============================================================

def perform_final_clustering(
    customer_features,
    X_scaled
):

    optimal_k = 4

    model = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=20
    )

    customer_features["Cluster"] = (
        model.fit_predict(X_scaled)
    )

    return customer_features, model


# ============================================================
# PCA
# ============================================================

def perform_pca(X_scaled):

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    return pca, X_pca


# ============================================================
# CLUSTER PROFILE
# ============================================================

def create_cluster_profile(
    customer_features
):

    profile = (
        customer_features
        .groupby("Cluster")
        .agg(

            Customers=(
                "CustomerID",
                "count"
            ),

            Recency=(
                "Recency",
                "mean"
            ),

            Frequency=(
                "Frequency",
                "mean"
            ),

            Monetary=(
                "Monetary",
                "mean"
            ),

            AOV=(
                "AOV",
                "mean"
            ),

            TotalQuantity=(
                "TotalQuantity",
                "mean"
            ),

            UniqueProducts=(
                "UniqueProducts",
                "mean"
            )
        )
    )

    profile[
        "Customer_Percentage"
    ] = (
        profile["Customers"]
        /
        len(customer_features)
        *
        100
    )

    return profile.reset_index()


# ============================================================
# ASSIGN BUSINESS SEGMENT LABELS
# ============================================================

def assign_segment_labels(
    customer_features,
    profile
):

    labels = {}

    for _, row in profile.iterrows():

        cluster = int(
            row["Cluster"]
        )

        frequency = row["Frequency"]
        monetary = row["Monetary"]
        recency = row["Recency"]

        if monetary > 5000 and frequency > 8:

            labels[cluster] = (
                "High-Value Champions"
            )

        elif recency > 200 and frequency <= 2:

            labels[cluster] = (
                "At-Risk / Inactive Customers"
            )

        elif frequency >= 2 and monetary > 700:

            labels[cluster] = (
                "Loyal Regulars"
            )

        else:

            labels[cluster] = (
                "Occasional / Discount Buyers"
            )

    customer_features[
        "Segment_Label"
    ] = customer_features[
        "Cluster"
    ].map(labels)

    return customer_features


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER CLUSTERING PIPELINE")
    print("=" * 70)

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading cleaned data...")

    df = load_cleaned_data()

    print(
        "Cleaned dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Customer features
    # --------------------------------------------------------

    print(
        "\nCreating customer-level features..."
    )

    customer_features = (
        create_customer_features(df)
    )

    print(
        "Customer feature shape:",
        customer_features.shape
    )

    # --------------------------------------------------------
    # Log transformation
    # --------------------------------------------------------

    customer_features = (
        transform_features(
            customer_features
        )
    )

    # --------------------------------------------------------
    # Save engineered features
    # --------------------------------------------------------

    customer_features.to_csv(
        CUSTOMER_FEATURES_PATH,
        index=False
    )

    print(
        "\nCustomer features saved to:"
    )

    print(
        CUSTOMER_FEATURES_PATH
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print(
        "\nScaling clustering features..."
    )

    (
        scaled_df,
        X_scaled,
        clustering_features
    ) = scale_features(
        customer_features
    )

    scaled_output = pd.concat(
        [
            customer_features[
                ["CustomerID"]
            ].reset_index(drop=True),

            scaled_df.reset_index(drop=True)
        ],
        axis=1
    )

    scaled_output.to_csv(
        SCALED_FEATURES_PATH,
        index=False
    )

    print(
        "Scaled features saved to:"
    )

    print(
        SCALED_FEATURES_PATH
    )

    # --------------------------------------------------------
    # K-Means evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating K-Means for K=2 to K=8..."
    )

    evaluation = evaluate_kmeans(
        X_scaled
    )

    print("\nK-Means Evaluation:")
    print(evaluation.to_string(
        index=False
    ))

    # --------------------------------------------------------
    # Final K-Means
    # --------------------------------------------------------

    print(
        "\nTraining final K-Means model..."
    )

    (
        customer_features,
        kmeans_model
    ) = perform_final_clustering(
        customer_features,
        X_scaled
    )

    print(
        "Final number of clusters:",
        4
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print(
        "\nPerforming PCA..."
    )

    pca, X_pca = perform_pca(
        X_scaled
    )

    customer_features["PCA1"] = (
        X_pca[:, 0]
    )

    customer_features["PCA2"] = (
        X_pca[:, 1]
    )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    print(
        "PCA explained variance:",
        explained_variance
    )

    # --------------------------------------------------------
    # Cluster profile
    # --------------------------------------------------------

    print(
        "\nCreating cluster profiles..."
    )

    profile = create_cluster_profile(
        customer_features
    )

    # --------------------------------------------------------
    # Segment labels
    # --------------------------------------------------------

    customer_features = (
        assign_segment_labels(
            customer_features,
            profile
        )
    )

    # Recreate profile after labels
    profile = (
        customer_features
        .groupby(
            ["Cluster", "Segment_Label"]
        )
        .agg(

            Customers=(
                "CustomerID",
                "count"
            ),

            Recency=(
                "Recency",
                "mean"
            ),

            Frequency=(
                "Frequency",
                "mean"
            ),

            Monetary=(
                "Monetary",
                "mean"
            ),

            AOV=(
                "AOV",
                "mean"
            ),

            TotalQuantity=(
                "TotalQuantity",
                "mean"
            ),

            UniqueProducts=(
                "UniqueProducts",
                "mean"
            )
        )
        .reset_index()
    )

    profile[
        "Customer_Percentage"
    ] = (
        profile["Customers"]
        /
        len(customer_features)
        *
        100
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    customer_features.to_csv(
        CUSTOMER_CLUSTERS_PATH,
        index=False
    )

    profile.to_csv(
        SEGMENT_SUMMARY_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\nCustomer clusters saved to:"
    )

    print(
        CUSTOMER_CLUSTERS_PATH
    )

    print(
        "\nSegment summary saved to:"
    )

    print(
        SEGMENT_SUMMARY_PATH
    )

    print("\n" + "=" * 70)
    print("CLUSTER PROFILE")
    print("=" * 70)

    print(
        profile.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("CLUSTERING PIPELINE COMPLETED")
    print("=" * 70)


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":

    main()