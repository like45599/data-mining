# K-Means Clustering Algorithm

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the basic principles and workflow of the K-Means algorithm</li>
      <li>Master methods for selecting K and evaluation metrics</li>
      <li>Learn about K-Means optimization variants and limitations</li>
      <li>Practice applying K-Means in scenarios such as customer segmentation</li>
    </ul>
  </div>
</div>

## Basic Principles of K-Means

K-Means is a classic unsupervised learning algorithm used to partition data into K distinct clusters. The core idea of the algorithm is that each sample should belong to the cluster whose center (centroid) is closest.

### Algorithm Workflow

The basic steps of the K-Means algorithm are as follows:

1. **Initialization**: Randomly select K points as the initial centroids
2. **Assignment**: Assign each sample to the cluster represented by the closest centroid
3. **Update**: Recalculate the centroids of each cluster (the mean of all points in the cluster)
4. **Iteration**: Repeat steps 2 and 3 until the centroids no longer change significantly or the maximum number of iterations is reached

<div class="visualization-container">
  <div class="visualization-title">K-Means Clustering Process</div>
  <div class="visualization-content">
    <img src="/images/kmeans_process.svg" alt="K-Means Clustering Process">
  </div>
  <div class="visualization-caption">
    Figure: The K-Means clustering process. Starting with randomly initialized centroids, the algorithm iteratively assigns and updates the centroids, eventually converging to a stable cluster division.
  </div>
</div>

### Mathematical Expression

The objective of K-Means is to minimize the sum of squared distances from each sample to the centroid of its assigned cluster, that is, minimize the following objective function:

$$J = \sum_{j=1}^{k} \sum_{i=1}^{n_j} ||x_i^{(j)} - c_j||^2$$

Where:
- $k$ is the number of clusters
- $n_j$ is the number of samples in the $j$-th cluster
- $x_i^{(j)}$ is the $i$-th sample in the $j$-th cluster
- $c_j$ is the centroid of the $j$-th cluster

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>The K-Means algorithm was first proposed by Stuart Lloyd in 1957 (as a technique for pulse code modulation), but it was officially published in 1982. Despite its simplicity, it remains widely used in various fields and is considered the benchmark for clustering analysis. K-Means is a greedy algorithm that guarantees convergence to a local optimum, but not necessarily the global optimum.</p>
  </div>
</div>

## K-Means Implementation and Application

### Basic Implementation

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate example data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and train the K-Means model
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Get the cluster centers
centers = kmeans.cluster_centers_

# Visualize the clustering result
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

  </div>
</div>

### Determining the Optimal K Value

Choosing the correct K value is crucial in K-Means. Common methods include the elbow method and silhouette coefficient:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Calculate SSE (sum of squared errors) for different K values
sse = []
silhouette_scores = []
range_k = range(2, 11)

for k in range_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    
    # Calculate silhouette coefficient
    if k > 1:  # Silhouette score requires at least 2 clusters
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))

# Plot elbow chart
plt.figure(figsize=(12, 5))

# SSE chart
plt.subplot(1, 2, 1)
plt.plot(range_k, sse, 'bo-')
plt.xlabel('K Value')
plt.ylabel('SSE')
plt.title('Elbow Method')

# Silhouette score chart
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('K Value')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Coefficient Method')

plt.tight_layout()
plt.show()
```

  </div>
</div>

### Evaluating Clustering Quality

In addition to the elbow method and silhouette coefficient, other metrics can also be used to assess clustering quality:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Model with the best K value
best_k = 4  # Assume best K value determined by elbow method
kmeans = KMeans(n_clusters=best_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)

# Calculate Calinski-Harabasz index (the higher the better)
ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {ch_score:.2f}")

# Calculate Davies-Bouldin index (the lower the better)
db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_score:.2f}")

# Calculate silhouette coefficient (the closer to 1, the better)
silhouette = silhouette_score(X, labels)
print(f"Silhouette Coefficient: {silhouette:.2f}")
```

  </div>
</div>

## K-Means Application Case

### Customer Segmentation

K-Means has wide applications in market segmentation and customer clustering:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer data (example)
# In a real application, use actual customer data
# Here we create a sample dataset
np.random.seed(42)
n_customers = 200

# Create features: Age, Income, and Purchase Frequency
age = np.random.normal(35, 10, n_customers)
income = np.random.normal(50000, 15000, n_customers)
frequency = np.random.normal(10, 5, n_customers)

# Create DataFrame
customer_data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Frequency': frequency
})

# Feature scaling
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Analyze average characteristics of each cluster
cluster_summary = customer_data.groupby('Cluster').mean()
print("Average characteristics of each customer group:")
print(cluster_summary)

# Visualize the results
plt.figure(figsize=(15, 5))

# Age vs Income
plt.subplot(1, 3, 1)
sns.scatterplot(x='Age', y='Income', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Age vs Income')

# Age vs Purchase Frequency
plt.subplot(1, 3, 2)
sns.scatterplot(x='Age', y='Frequency', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Age vs Purchase Frequency')

# Income vs Purchase Frequency
plt.subplot(1, 3, 3)
sns.scatterplot(x='Income', y='Frequency', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Income vs Purchase Frequency')

plt.tight_layout()
plt.show()

# Create customer profiles for each cluster
for cluster in range(3):
    print(f"\nCustomer Group {cluster} Profile:")
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    print(f"Number of customers: {len(cluster_data)}")
    print(f"Average age: {cluster_data['Age'].mean():.1f} years")
    print(f"Average income: ${cluster_data['Income'].mean():.2f}")
    print(f"Average purchase frequency: {cluster_data['Frequency'].mean():.1f} times/month")
```

  </div>
</div>

## K-Means Variants and Optimization

### K-Means++

K-Means++ improves the method of selecting initial centroids, making the clustering results more stable:

1. Randomly select the first centroid
2. For each subsequent centroid, choose points that are farther from the existing centroids
3. This method reduces the sensitivity of K-Means to initial values

### Mini-Batch K-Means

For large-scale datasets, Mini-Batch K-Means trains using small batches of data, improving computational efficiency:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Generate large-scale data
X_large, _ = make_blobs(n_samples=10000, centers=5, cluster_std=0.6, random_state=0)

# Compare the performance of K-Means and Mini-Batch K-Means
start_time = time.time()
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, random_state=0)
kmeans.fit(X_large)
kmeans_time = time.time() - start_time
print(f"K-Means runtime: {kmeans_time:.4f} seconds")

start_time = time.time()
mbk = MiniBatchKMeans(n_clusters=5, init='k-means++', batch_size=100, max_iter=100, random_state=0)
mbk.fit(X_large)
mbk_time = time.time() - start_time
print(f"Mini-Batch K-Means runtime: {mbk_time:.4f} seconds")
print(f"Speedup: {kmeans_time/mbk_time:.2f}x")

# Compare clustering results
kmeans_labels = kmeans.labels_
mbk_labels = mbk.labels_

# Calculate silhouette scores for both methods
kmeans_silhouette = silhouette_score(X_large, kmeans_labels)
mbk_silhouette = silhouette_score(X_large, mbk_labels)

print(f"K-Means silhouette score: {kmeans_silhouette:.4f}")
print(f"Mini-Batch K-Means silhouette score: {mbk_silhouette:.4f}")
```

  </div>
</div>

### Other Variants

- **K-Medoids**: Uses actual data points as cluster centers, more robust to outliers
- **K-Means++**: Optimizes the initial centroid selection
- **Fuzzy K-Means**: Allows samples to belong to multiple clusters with different degrees of membership
- **Spherical K-Means**: Suitable for high-dimensional sparse data, such as text

## Limitations of K-Means

Although K-Means is simple and efficient, it has some inherent limitations:

1. **Requires pre-specification of K**: Choosing the correct K can be difficult
2. **Sensitive to initial centroids**: Different initializations may lead to different results
3. **Assumes clusters are convex and similar in size**: Not suitable for identifying complex-shaped clusters
4. **Sensitive to outliers**: Outliers can significantly affect the centroid position
5. **Only suitable for numerical features**: Categorical features require special handling

<div class="visualization-container">
  <div class="visualization-title">Limitations of K-Means</div>
  <div class="visualization-content">
    <img src="/images/kmeans_limitations.svg" alt="Limitations of K-Means">
  </div>
  <div class="visualization-caption">
    Figure: Limitations of K-Means on non-convex clusters.
  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>Common Pitfalls
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Neglecting Feature Scaling</strong>: Failure to standardize features leads to high-variance features dominating the clustering result</li>
      <li><strong>Blindly Choosing K</strong>: Failing to validate the K value using methods like the elbow method</li>
      <li><strong>Over-interpreting Clustering Results</strong>: Treating clustering results as absolute truths rather than tools for data exploration</li>
      <li><strong>Neglecting Data Preprocessing</strong>: Ignoring outliers and missing values, which affect clustering quality</li>
    </ul>
  </div>
</div>

## Summary and Reflection

K-Means is a simple yet powerful clustering algorithm. Despite its limitations, it is still highly effective in many real-world applications.

### Key Takeaways

- K-Means divides data into K clusters through iterative optimization
- The algorithm's objective is to minimize the sum of squared distances from samples to centroids
- Selecting the appropriate K value is crucial for good clustering results
- Variants like K-Means++ improve initial centroid selection
- Mini-Batch K-Means provides computational efficiency for large-scale data

### Reflection Questions

1. In which cases might K-Means not be the best choice?
2. How can we address K-Means' sensitivity to outliers?
3. Apart from the elbow method and silhouette coefficient, what other methods can be used to determine the optimal K?

<BackToPath />

<div class="practice-link">
  <a href="/projects/clustering.html" class="button">Go to Practice Project</a>
</div>
