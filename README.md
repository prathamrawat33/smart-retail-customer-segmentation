


# Smart Retail Customer Segmentation & Cluster-Specific Market Basket Analysis

## 📌 Project Overview

This project combines customer segmentation, market basket analysis, and personalized recommendations to understand retail customer behavior.

The system follows:

Customer Data
↓
Data Preprocessing
↓
RFM & Behavioral Features
↓
K-Means Customer Segmentation
↓
Cluster-Specific Market Basket Analysis
↓
Personalized Product Recommendations
↓
Streamlit Dashboard

---

## 🎯 Objectives

- Clean and preprocess retail transaction data
- Build customer-level behavioral features
- Segment customers using K-Means clustering
- Perform market basket analysis separately for each customer segment
- Generate personalized product recommendations
- Provide an interactive dashboard for business insights

---

## 📊 Dataset

The project uses an online retail transaction dataset.

After preprocessing:

- Transactions: 392,692
- Customers: 4,338
- Products: 3,665
- Invoices: 18,532
- Countries: 37

---

## ⚙️ Customer Features

The following features are used for customer segmentation:

- Recency
- Frequency
- Monetary
- Average Order Value
- Total Quantity
- Unique Products

Highly skewed features were transformed using `log1p`, followed by standardization using `StandardScaler`.

---

## 🤖 Customer Segmentation

K-Means clustering was evaluated for K = 2 to 8.

The project uses **K = 4** to provide four meaningful business segments.

### Customer Segments

| Cluster | Segment |
|---|---|
| 0 | Occasional / Discount Buyers |
| 1 | High-Value Champions |
| 2 | At-Risk / Inactive Customers |
| 3 | Loyal Regulars |

### Model Diagnostics

- Selected clusters: 4
- K=4 Silhouette Score: 0.259
- PCA PC1: 64.89%
- PCA PC2: 16.68%
- Combined PCA variance: 81.57%

---

## 🛒 Market Basket Analysis

FP-Growth is applied separately to each customer segment.

This allows the system to discover product associations that are specific to different customer groups.

### Association Rules Generated

| Cluster | Rules |
|---|---:|
| 0 | 223 |
| 1 | 16,188 |
| 2 | 2,062 |
| 3 | 3,010 |

Total association rules: **21,483**

Rules are evaluated using:

- Support
- Confidence
- Lift

---

## 🎯 Recommendation Engine

The recommendation engine follows:

Customer
↓
Customer Cluster
↓
Cluster-Specific Association Rules
↓
Customer Purchase History
↓
Recommended Products

If suitable association rules are unavailable, the system falls back to segment-level bestselling products.

---

## 🖥️ Dashboard

The Streamlit dashboard contains:

### Executive Overview
- Customer statistics
- Revenue
- Product statistics
- Segment distribution
- Segment profiles
- Top products

### Customer Intelligence
- Customer metrics
- Behavioral profile
- PCA visualization
- Customer information
- Model diagnostics

### Market Basket AI
- Segment-specific rules
- Support
- Confidence
- Lift
- Association analysis

### Recommendation Engine
- Customer selection
- Customer segment
- Purchase history
- Personalized recommendations
- Recommendation source
- Confidence and lift

---

## 🧪 Testing

The automated test suite validates:

- Cleaned dataset
- Required columns
- Customer clusters
- Association rules
- Recommendation output
- Valid customer IDs
- Recommendation sources
- Duplicate recommendations
- Recommendation ranking

Current result:

**19 tests passed**

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt