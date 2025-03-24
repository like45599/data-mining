# Customer Segmentation Analysis

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Intermediate</li>
      <li><strong>Type</strong>: Clustering Analysis</li>
      <!-- <li><strong>Estimated Time</strong>: 5-7 hours</li> -->
      <li><strong>Skills</strong>: Data Standardization, K-Means Clustering, Cluster Evaluation, Business Interpretation</li>
      <li><strong>Related Knowledge Module</strong>: <a href="/en/core/clustering/kmeans.html">Clustering Algorithms</a></li>
    </ul>
  </div>
</div>

## Project Background

Customer segmentation is an important strategy for businesses to understand customer behavior and preferences. By grouping similar customers together, companies can develop targeted marketing strategies, personalized product recommendations, and tailored service plans. Traditional segmentation is often based on demographic features (such as age, gender, and income), but modern analysis methods can combine multidimensional data like transaction history, browsing behavior, and interaction patterns to create more refined and meaningful customer groups.

In this project, we will use clustering algorithms to segment customer data from a retailer, identify the characteristics and behavior patterns of different groups, and support business decision-making.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>RFM analysis is a classic method for customer segmentation, based on three key indicators: Recency, Frequency, and Monetary. This simple yet effective approach is still widely used and can be combined with modern clustering algorithms to create more detailed customer profiles.</p>
  </div>
</div>

## Dataset Introduction

The dataset used in this project contains one year of transaction records from an online retailer, with approximately 50,000 transactions and the following fields:

- **CustomerID**: Unique customer identifier
- **InvoiceNo**: Invoice number
- **InvoiceDate**: Transaction date and time
- **StockCode**: Product code
- **Description**: Product description
- **Quantity**: Quantity purchased
- **UnitPrice**: Unit price
- **Country**: Customer’s country

We will extract customer-level features from these transaction records and then apply clustering algorithms for segmentation.

## Project Objectives

1. Extract customer-level features from transaction data  
2. Apply clustering algorithms such as K-Means to segment customers  
3. Evaluate clustering results and determine the optimal number of clusters  
4. Analyze the characteristics and behavior patterns of different customer groups  
5. Provide business recommendations based on segmentation results  

## Implementation Steps

### Step 1: Data Loading and Preprocessing

First, we load the data and perform the necessary preprocessing.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# View basic information about the data
print(df.info())
print(df.head())

# Data cleaning
# Remove rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove cancelled orders (Quantity < 0)
df = df[df['Quantity'] > 0]

# Remove records with UnitPrice <= 0
df = df[df['UnitPrice'] > 0]

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create a TotalAmount column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# View the cleaned data
print("Shape of cleaned data:", df.shape)
print(df.describe())

# View the number of customers per country
country_counts = df.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar')
plt.title('Number of Customers by Country')
plt.xlabel('Country')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 2: Feature Extraction - RFM Analysis

Next, we use the RFM method to extract customer-level features from the transaction data.

```python
# Set the analysis snapshot date (the day after the last date in the dataset)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Compute RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# View the RFM data
print(rfm.head())
print(rfm.describe())

# Visualize the RFM distributions
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(rfm['Recency'], bins=30, kde=True)
plt.title('Recency Distribution')

plt.subplot(1, 3, 2)
sns.histplot(rfm['Frequency'], bins=30, kde=True)
plt.title('Frequency Distribution')

plt.subplot(1, 3, 3)
sns.histplot(rfm['Monetary'], bins=30, kde=True)
plt.title('Monetary Distribution')
plt.tight_layout()
plt.show()

# View correlations among RFM metrics
plt.figure(figsize=(10, 8))
sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Among RFM Metrics')
plt.show()
```

### Step 3: Feature Processing and Standardization

Before applying clustering algorithms, we need to handle outliers and standardize the features.

```python
# Handle outliers
# Use the IQR method to identify outliers
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Apply outlier removal
rfm_clean = remove_outliers(rfm, ['Recency', 'Frequency', 'Monetary'])
print(f"Rows before outlier removal: {rfm.shape[0]}, after removal: {rfm_clean.shape[0]}")

# Log-transform to handle skewed distributions
rfm_log = rfm_clean.copy()
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

# Visualize the transformed distributions
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(rfm_log['Recency'], bins=30, kde=True)
plt.title('Recency Distribution')

plt.subplot(1, 3, 2)
sns.histplot(rfm_log['Frequency'], bins=30, kde=True)
plt.title('Log(Frequency) Distribution')

plt.subplot(1, 3, 3)
sns.histplot(rfm_log['Monetary'], bins=30, kde=True)
plt.title('Log(Monetary) Distribution')
plt.tight_layout()
plt.show()

# Prepare features for clustering
X = rfm_log.copy()
features = ['Recency', 'Frequency', 'Monetary']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for analysis
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print(X_scaled_df.head())
```

### Step 4: Determine the Optimal Number of Clusters

Use the elbow method and silhouette scores to determine the best number of clusters.

```python
# Elbow method
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualize the elbow method and silhouette scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.grid(True)
plt.tight_layout()
plt.show()

# Choose the optimal number of clusters based on the silhouette score
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters (based on Silhouette Score): {best_k}")
```

### Step 5: Apply K-Means Clustering

Apply K-Means clustering using the determined optimal number of clusters.

```python
# Apply K-Means clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# View the size of each cluster
cluster_sizes = rfm['Cluster'].value_counts().sort_index()
print("Cluster Sizes:")
print(cluster_sizes)

# Calculate cluster centers (in original scale)
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                              columns=features)
cluster_centers['Cluster'] = range(best_k)
print("\nCluster Centers:")
print(cluster_centers)

# Visualize clustering results using PCA to reduce dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for i in range(best_k):
    plt.scatter(X_pca[rfm['Cluster'] == i, 0], X_pca[rfm['Cluster'] == i, 1], label=f'Cluster {i}')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], 
            s=300, c='red', marker='X', label='Centroids')
plt.title('Customer Segmentation (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
```

### Step 6: Analyze Customer Group Characteristics

Analyze the characteristics of each customer group and provide business recommendations.

```python
# Analyze the RFM characteristics for each cluster
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

# Visualize the characteristics for each cluster
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.barplot(x='Cluster', y='Recency', data=cluster_analysis)
plt.title('Average Recency per Cluster')

plt.subplot(1, 3, 2)
sns.barplot(x='Cluster', y='Frequency', data=cluster_analysis)
plt.title('Average Frequency per Cluster')

plt.subplot(1, 3, 3)
sns.barplot(x='Cluster', y='Monetary', data=cluster_analysis)
plt.title('Average Monetary Value per Cluster')
plt.tight_layout()
plt.show()

# Radar chart visualization for cluster characteristics
from math import pi

# Normalize data for radar chart
radar_df = cluster_analysis.copy()
for feature in features:
    radar_df[feature] = (radar_df[feature] - radar_df[feature].min()) / (radar_df[feature].max() - radar_df[feature].min())
    # For Recency, lower is better so we invert the values
    if feature == 'Recency':
        radar_df[feature] = 1 - radar_df[feature]

# Set up radar chart
categories = features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(12, 10))
ax = plt.subplot(111, polar=True)

for i, cluster in enumerate(radar_df['Cluster']):
    values = radar_df.loc[radar_df['Cluster'] == cluster, features].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], categories)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Radar Chart of Customer Group Characteristics')
plt.show()

# Naming the customer segments and providing business recommendations
cluster_names = {
    0: "High-Value Loyal Customers",
    1: "Potential Customers",
    2: "Dormant Customers",
    3: "New Customers"
}

business_recommendations = {
    0: "Offer VIP services, send personalized discounts, and encourage referrals.",
    1: "Provide purchase incentives, increase purchase frequency, and recommend related products.",
    2: "Send reactivation emails, offer special discounts, and understand reasons for churn.",
    3: "Provide a good first experience, encourage a second purchase, and collect feedback."
}

# Create the final customer segmentation report
cluster_report = pd.DataFrame({
    'Cluster': range(best_k),
    'Name': [cluster_names.get(i, f"Cluster {i}") for i in range(best_k)],
    'Size': cluster_sizes.values,
    'Recency': cluster_analysis['Recency'],
    'Frequency': cluster_analysis['Frequency'],
    'Monetary': cluster_analysis['Monetary'],
    'Business Recommendations': [business_recommendations.get(i, "") for i in range(best_k)]
})

print("Customer Segmentation Report:")
print(cluster_report)
```

## Results Analysis

Through the K-Means clustering algorithm, we successfully segmented customers into distinct groups, each with different purchasing behavior characteristics:

1. **High-Value Loyal Customers**: These customers have made recent purchases, exhibit high purchase frequency, and spend a large amount. They are the core customer group contributing most to revenue.
2. **Potential Customers**: These customers show moderate purchase frequency and spending, and have the potential to become high-value customers.
3. **Dormant Customers**: These customers have not purchased for a long time despite previously good purchasing records. They are targets for reactivation campaigns.
4. **New Customers**: These customers have only recently started purchasing, with low frequency and spending. Special attention is needed to improve retention.

This customer segmentation helps businesses formulate targeted marketing strategies, improve customer satisfaction and loyalty, and ultimately enhance sales performance and customer lifetime value.

## Advanced Challenges

If you have completed the basic tasks, you can try the following advanced challenges:

1. **Expand the Feature Set**: In addition to RFM indicators, try incorporating features such as product category preferences and purchasing time patterns.
2. **Try Other Clustering Algorithms**: Compare the results of hierarchical clustering, DBSCAN, and other methods with K-Means.
3. **Time Series Analysis**: Analyze how customer segments change over time to identify migration patterns.
4. **Predictive Models**: Build churn prediction or next-purchase prediction models based on the segmentation results.
5. **Recommendation Systems**: Develop tailored product recommendation strategies for each customer segment.

## Summary and Reflection

Through this project, we learned how to use clustering algorithms to segment customers and extract valuable business insights from data. Customer segmentation is a powerful tool for understanding customer behavior and preferences, helping businesses optimize marketing strategies, improve customer satisfaction and loyalty. In practice, segmentation should be an ongoing process—updated regularly as new data accumulates and the business environment changes. Moreover, interpreting and applying segmentation results must be aligned with specific business contexts and objectives to maximize value.

### Questions for Reflection

1. Besides RFM indicators, what other customer features might be valuable for segmentation? How can these features be obtained and integrated?
2. How might customer segmentation differ across industries? For example, which indicators should be prioritized for segmentation in e-commerce, financial services, and content subscription services?
3. How can the business value of customer segmentation be evaluated? What metrics can be used to measure the effectiveness of segmentation strategies?

<div class="practice-link">
  <a href="/en/projects/clustering/image-segmentation.html" class="button">Next Project: Image Color Segmentation</a>
</div>
