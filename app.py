import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Retail AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    cleaned = pd.read_csv(
        DATA_DIR / "cleaned_data.csv"
    )

    clusters = pd.read_csv(
        DATA_DIR / "customer_clusters.csv"
    )

    rules = pd.read_csv(
        DATA_DIR / "cluster_specific_association_rules.csv"
    )

    recommendations = pd.read_csv(
        DATA_DIR / "customer_recommendations.csv"
    )

    cleaned["InvoiceDate"] = pd.to_datetime(
        cleaned["InvoiceDate"],
        errors="coerce"
    )

    return (
        cleaned,
        clusters,
        rules,
        recommendations
    )


try:

    cleaned, clusters, rules, recommendations = load_data()

except Exception as e:

    st.error("Could not load project data.")
    st.code(str(e))
    st.stop()


# ============================================================
# SEGMENTS
# ============================================================

SEGMENTS = {
    0: "Occasional / Discount Buyers",
    1: "High-Value Champions",
    2: "At-Risk / Inactive Customers",
    3: "Loyal Regulars"
}

ICONS = {
    0: "🛍️",
    1: "💎",
    2: "⚠️",
    3: "⭐"
}


clusters["Cluster"] = clusters["Cluster"].astype(int)
rules["Cluster"] = rules["Cluster"].astype(int)
recommendations["Cluster"] = recommendations["Cluster"].astype(int)

clusters["Segment"] = clusters["Cluster"].map(SEGMENTS)
rules["Segment"] = rules["Cluster"].map(SEGMENTS)
recommendations["Segment"] = recommendations["Cluster"].map(SEGMENTS)


# ============================================================
# HELPER
# ============================================================

def clean_rule(value):

    if pd.isna(value):
        return ""

    text = str(value)

    try:

        parsed = ast.literal_eval(text)

        if isinstance(
            parsed,
            (set, frozenset, list, tuple)
        ):

            return " + ".join(
                sorted(
                    str(x)
                    for x in parsed
                )
            )

    except Exception:
        pass

    return (
        text
        .replace("frozenset({", "")
        .replace("})", "")
        .replace("{", "")
        .replace("}", "")
        .replace("'", "")
        .replace('"', "")
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛍️ Smart Retail AI")

    st.caption(
        "Customer Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "👥 Customer Intelligence",
            "🛒 Market Basket AI",
            "🎯 Recommendation Engine"
        ]
    )

    st.divider()

    st.caption(
        "K-Means • FP-Growth • Recommendations"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🛍️ Smart Retail AI")

st.caption(
    "Customer Segmentation → Market Basket Analysis → "
    "Personalized Recommendations"
)

st.divider()


# ============================================================
# PAGE 1
# EXECUTIVE OVERVIEW
# ============================================================

if page == "📊 Executive Overview":

    st.header("Executive Overview")

    st.caption(
        "A compact view of the retail business and customer segments."
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    customers = clusters["CustomerID"].nunique()

    orders = cleaned["InvoiceNo"].nunique()

    products = cleaned["StockCode"].nunique()

    revenue = cleaned["TotalAmount"].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{customers:,}"
    )

    c2.metric(
        "Orders",
        f"{orders:,}"
    )

    c3.metric(
        "Products",
        f"{products:,}"
    )

    c4.metric(
        "Revenue",
        f"£{revenue:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER MIX
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Customer Segment Distribution"
        )

        segment_counts = (
            clusters
            .groupby("Segment")
            .size()
            .reset_index(
                name="Customers"
            )
        )

        fig = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            hole=0.55
        )

        fig.update_layout(
            height=430
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # VALUE
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Average Customer Value"
        )

        value = (
            clusters
            .groupby("Segment")["Monetary"]
            .mean()
            .sort_values()
            .reset_index()
        )

        fig = px.bar(
            value,
            x="Monetary",
            y="Segment",
            orientation="h",
            text_auto=".0f"
        )

        fig.update_layout(
            height=430,
            xaxis_title="Average Revenue (£)",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # SEGMENT TABLE
    # --------------------------------------------------------

    st.subheader(
        "Segment Profile"
    )

    profile = (
        clusters
        .groupby(
            ["Cluster", "Segment"]
        )
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgAOV=("AOV", "mean")
        )
        .reset_index()
    )

    profile["Share"] = (
        profile["Customers"]
        / customers
    )

    st.dataframe(
        profile.style.format(
            {
                "Share": "{:.1%}",
                "AvgRecency": "{:.1f}",
                "AvgFrequency": "{:.2f}",
                "AvgMonetary": "£{:,.0f}",
                "AvgAOV": "£{:,.0f}"
            }
        ),
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

    st.subheader(
        "Top Products"
    )

    top_products = (
        cleaned
        .groupby("Description")["Quantity"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products.sort_values("Quantity"),
        x="Quantity",
        y="Description",
        orientation="h",
        text_auto=True
    )

    fig.update_layout(
        height=500,
        xaxis_title="Quantity Sold",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# PAGE 2
# CUSTOMER INTELLIGENCE
# ============================================================

elif page == "👥 Customer Intelligence":

    st.header(
        "👥 Customer Intelligence"
    )

    st.caption(
        "Explore the behavioral profile produced by K-Means."
    )

    # --------------------------------------------------------
    # SELECT SEGMENT
    # --------------------------------------------------------

    selected_cluster = st.selectbox(
        "Select Segment",
        sorted(SEGMENTS.keys()),
        format_func=lambda x:
            f"{ICONS[x]} {SEGMENTS[x]}"
    )

    data = clusters[
        clusters["Cluster"]
        == selected_cluster
    ]

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{len(data):,}"
    )

    c2.metric(
        "Avg Recency",
        f"{data['Recency'].mean():.1f} days"
    )

    c3.metric(
        "Avg Frequency",
        f"{data['Frequency'].mean():.2f}"
    )

    c4.metric(
        "Avg Monetary",
        f"£{data['Monetary'].mean():,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # BEHAVIORAL PROFILE
    # --------------------------------------------------------

    left, right = st.columns([1, 1.5])

    with left:

        st.subheader(
            "Behavioral Profile"
        )

        values = {
            "Recency": data["Recency"].mean(),
            "Frequency": data["Frequency"].mean(),
            "AOV": data["AOV"].mean(),
            "Unique Products": data["UniqueProducts"].mean()
        }

        values = pd.Series(values)

        minimum = values.min()
        maximum = values.max()

        normalized = (
            (values - minimum)
            / (maximum - minimum)
            if maximum != minimum
            else values * 0
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=normalized.values,
                theta=normalized.index,
                fill="toself",
                name=SEGMENTS[selected_cluster]
            )
        )

        fig.update_layout(
            height=430,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Customer Cluster Map"
        )

        if (
            "PC1" in clusters.columns
            and
            "PC2" in clusters.columns
        ):

            fig = px.scatter(
                clusters,
                x="PC1",
                y="PC2",
                color="Segment",
                hover_name="CustomerID",
                hover_data=[
                    "Recency",
                    "Frequency",
                    "Monetary"
                ],
                opacity=0.7
            )

            fig.update_layout(
                height=560,
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.warning(
                "PCA coordinates are not available "
                "in customer_clusters.csv."
            )

    # --------------------------------------------------------
    # CUSTOMER LIST
    # --------------------------------------------------------

    with st.expander(
        "View customers in this segment"
    ):

        st.dataframe(
            data[
                [
                    "CustomerID",
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "AOV",
                    "UniqueProducts"
                ]
            ]
            .sort_values(
                "Monetary",
                ascending=False
            )
            .head(100),
            width="stretch",
            hide_index=True
        )


# ============================================================
# PAGE 3
# MARKET BASKET
# ============================================================
    # ==============================
    # MODEL DIAGNOSTICS
    # ==============================

    st.divider()

    st.subheader("🔬 Model Diagnostics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Clusters", "4")

    with col2:
        st.metric("K=4 Silhouette", "0.259")

    with col3:
        st.metric("PCA Variance", "81.57%")

    st.markdown("### Silhouette Score by Number of Clusters")

    silhouette_df = pd.DataFrame({
        "K": [2, 3, 4, 5, 6, 7, 8],
        "Silhouette": [
            0.350888,
            0.272122,
            0.258979,
            0.240340,
            0.244674,
            0.230763,
            0.216564
        ]
    })

    fig_silhouette = px.line(
        silhouette_df,
        x="K",
        y="Silhouette",
        markers=True,
        title="Clustering Quality"
    )

    fig_silhouette.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score"
    )

    st.plotly_chart(
        fig_silhouette,
        width="stretch"
    )

    st.markdown("### PCA Explained Variance")

    pca_df = pd.DataFrame({
        "Component": ["PC1", "PC2", "Combined"],
        "Variance": [64.89, 16.68, 81.57]
    })

    fig_pca = px.bar(
        pca_df,
        x="Component",
        y="Variance",
        text="Variance",
        title="PCA Variance Explained (%)"
    )

    fig_pca.update_layout(
        yaxis_title="Variance Explained (%)"
    )

    st.plotly_chart(
        fig_pca,
        width="stretch"
    )



elif page == "🛒 Market Basket AI":


        # ==============================
    # MODEL DIAGNOSTICS
    # ==============================

    st.divider()

    st.subheader("🔬 Model Diagnostics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Clusters", "4")

    with col2:
        st.metric("K=4 Silhouette", "0.259")

    with col3:
        st.metric("PCA Variance", "81.57%")

    # Silhouette scores
    silhouette_df = pd.DataFrame({
        "K": [2, 3, 4, 5, 6, 7, 8],
        "Silhouette": [
            0.350888,
            0.272122,
            0.258979,
            0.240340,
            0.244674,
            0.230763,
            0.216564
        ]
    })

    st.markdown("### Silhouette Score by Number of Clusters")

    fig_silhouette = px.line(
        silhouette_df,
        x="K",
        y="Silhouette",
        markers=True
    )

    fig_silhouette.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score"
    )

    st.plotly_chart(
        fig_silhouette,
        width="stretch"
    )

    # PCA explained variance
    st.markdown("### PCA Explained Variance")

    pca_df = pd.DataFrame({
        "Component": ["PC1", "PC2", "Combined"],
        "Variance": [64.89, 16.68, 81.57]
    })

    fig_pca = px.bar(
        pca_df,
        x="Component",
        y="Variance",
        text="Variance"
    )

    fig_pca.update_layout(
        yaxis_title="Variance Explained (%)"
    )

    st.plotly_chart(
        fig_pca,
        width="stretch"
    )

    st.header(
        "🛒 Market Basket AI"
    )

    st.caption(
        "Segment-specific product associations discovered using FP-Growth."
    )

    selected_cluster = st.selectbox(
        "Select Segment",
        sorted(SEGMENTS.keys()),
        format_func=lambda x:
            f"{ICONS[x]} {SEGMENTS[x]}"
    )

    segment_rules = rules[
        rules["Cluster"]
        == selected_cluster
    ].copy()

    # --------------------------------------------------------
    # RULE METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rules",
        f"{len(segment_rules):,}"
    )

    c2.metric(
        "Avg Support",
        f"{segment_rules['support'].mean():.2%}"
    )

    c3.metric(
        "Avg Confidence",
        f"{segment_rules['confidence'].mean():.2%}"
    )

    c4.metric(
        "Avg Lift",
        f"{segment_rules['lift'].mean():.2f}x"
    )

    st.divider()

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    st.subheader(
        "Association Rule Explorer"
    )

    c1, c2 = st.columns(2)

    with c1:

        confidence = st.slider(
            "Minimum Confidence",
            0.0,
            1.0,
            0.30,
            0.05
        )

    with c2:

        maximum_lift = max(
            10.0,
            min(
                100.0,
                float(
                    segment_rules["lift"].max()
                )
            )
        )

        lift = st.slider(
            "Minimum Lift",
            1.0,
            maximum_lift,
            1.0,
            1.0
        )

    filtered = segment_rules[
        (
            segment_rules["confidence"]
            >= confidence
        )
        &
        (
            segment_rules["lift"]
            >= lift
        )
    ].copy()

    st.caption(
        f"{len(filtered):,} rules match the selected thresholds."
    )

    if len(filtered) > 0:

        filtered["From"] = (
            filtered["antecedents_names"]
            .apply(clean_rule)
        )

        filtered["To"] = (
        filtered["consequents_names"]
        .apply(clean_rule)
)
        

        table = (
            filtered[
                [
                    "From",
                    "To",
                    "support",
                    "confidence",
                    "lift"
                ]
            ]
            .sort_values(
                "lift",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            table.style.format(
                {
                    "support": "{:.2%}",
                    "confidence": "{:.2%}",
                    "lift": "{:.2f}x"
                }
            ),
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "Highest-Lift Associations"
        )

        chart = (
            filtered
            .sort_values(
                "lift",
                ascending=False
            )
            .head(10)
            .copy()
        )

        chart["Rule"] = (
            chart["From"]
            + " → "
            + chart["To"]
        )

        fig = px.bar(
            chart.sort_values("lift"),
            x="lift",
            y="Rule",
            orientation="h",
            text_auto=".1f"
        )

        fig.update_layout(
            height=500,
            xaxis_title="Lift",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "No rules match the selected thresholds."
        )


# ============================================================
# PAGE 4
# RECOMMENDATION ENGINE
# ============================================================

elif page == "🎯 Recommendation Engine":

    st.header(
        "🎯 Recommendation Engine"
    )

    st.caption(
        "Customer history → segment → association rules → recommendation"
    )

    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    customer_ids = sorted(
        recommendations[
            "CustomerID"
        ].unique()
    )

    selected_customer = st.selectbox(
        "Select Customer",
        customer_ids,
        format_func=lambda x:
            str(int(float(x)))
    )

    customer = clusters[
        clusters["CustomerID"]
        == selected_customer
    ]

    recs = recommendations[
        recommendations["CustomerID"]
        == selected_customer
    ].copy()

    if len(customer) == 0:

        st.warning(
            "Customer profile not found."
        )

        st.stop()

    customer = customer.iloc[0]

    cluster = int(
        customer["Cluster"]
    )

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    st.subheader(
        f"{ICONS[cluster]} {SEGMENTS[cluster]}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customer",
        str(int(float(selected_customer)))
    )

    c2.metric(
        "Recency",
        f"{customer['Recency']:.0f} days"
    )

    c3.metric(
        "Frequency",
        f"{customer['Frequency']:.0f}"
    )

    c4.metric(
        "Monetary",
        f"£{customer['Monetary']:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # EXPLAINABLE PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "How the recommendation is generated"
    )

    a, b, c, d = st.columns(4)

    with a:
        st.info(
            f"**1. Customer**\n\n"
            f"#{int(float(selected_customer))}"
        )

    with b:
        st.info(
            f"**2. Segment**\n\n"
            f"{ICONS[cluster]} "
            f"{SEGMENTS[cluster]}"
        )

    with c:
        st.info(
            "**3. Association Rules**\n\n"
            "Rules learned from this segment"
        )

    with d:
        st.success(
            "**4. Recommendation**\n\n"
            "Suggest products not yet purchased"
        )

    st.divider()

    # --------------------------------------------------------
    # HISTORY + RECOMMENDATIONS
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "🛒 Purchase History"
        )

        history = cleaned[
            cleaned["CustomerID"]
            == selected_customer
        ]

        history_data = (
            history
            .groupby(
                "Description"
            )["Quantity"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        if len(history_data) > 0:

            fig = px.bar(
                history_data.sort_values(
                    "Quantity"
                ),
                x="Quantity",
                y="Description",
                orientation="h",
                text_auto=True
            )

            fig.update_layout(
                height=500,
                xaxis_title="Quantity Purchased",
                yaxis_title=""
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

    with right:

        st.subheader(
            "✨ Recommended Products"
        )

        if len(recs) == 0:

            st.info(
                "No recommendations available."
            )

        else:

            recs = (
                recs
                .sort_values("Rank")
                .head(6)
            )

            for _, row in recs.iterrows():

                product = row.get(
                    "Product",
                    "Recommended Product"
                )

                source = row.get(
                    "Source",
                    "Recommendation"
                )

                confidence = row.get(
                    "Confidence",
                    None
                )

                lift = row.get(
                    "Lift",
                    None
                )

                st.container(
                    border=True
                )

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### ✨ {product}"
                    )

                    if pd.notna(confidence):

                        confidence_text = (
                            f"{float(confidence):.1%}"
                        )

                    else:

                        confidence_text = "—"

                    if pd.notna(lift):

                        lift_text = (
                            f"{float(lift):.2f}x"
                        )

                    else:

                        lift_text = "—"

                    st.caption(
                        f"{source}  •  "
                        f"Confidence: {confidence_text}  •  "
                        f"Lift: {lift_text}"
                    )

    # --------------------------------------------------------
    # CONFIDENCE CHART
    # --------------------------------------------------------

    if (
        len(recs) > 0
        and
        "Confidence" in recs.columns
    ):

        st.divider()

        st.subheader(
            "Recommendation Confidence"
        )

        chart = recs.copy()

        chart["Confidence"] = pd.to_numeric(
            chart["Confidence"],
            errors="coerce"
        )

        chart = chart.dropna(
            subset=["Confidence"]
        )

        if len(chart) > 0:

            fig = px.bar(
                chart.sort_values(
                    "Confidence"
                ),
                x="Confidence",
                y="Product",
                orientation="h",
                text_auto=".0%"
            )

            fig.update_layout(
                height=400,
                xaxis_title="Confidence",
                yaxis_title=""
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Retail AI • "
    "K-Means • FP-Growth • Personalized Recommendations"
)