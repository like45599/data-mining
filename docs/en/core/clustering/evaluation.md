# Clustering Evaluation Metrics

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the importance and challenges of evaluating clustering results</li>
      <li>Master the calculation methods and applicable scenarios for internal evaluation metrics</li>
      <li>Learn the conditions and limitations of external evaluation metrics</li>
      <li>Understand how to choose appropriate evaluation metrics for clustering validation</li>
    </ul>
  </div>
</div>

## Challenges in Clustering Evaluation

Clustering is an unsupervised learning method with no standard answers, which makes evaluating the quality of clustering results challenging. Clustering evaluation typically considers the following aspects:

1. **Intra-cluster Similarity**: Samples within the same cluster should be as similar as possible
2. **Inter-cluster Discrepancy**: Samples between different clusters should be as different as possible
3. **Number of Clusters**: The appropriate number of clusters should reflect the natural structure of the data
4. **Shape of Clusters**: Whether the algorithm can identify non-convex clusters in the data

## Internal Evaluation Metrics

Internal evaluation metrics assess clustering quality based on the data's intrinsic characteristics without needing external labels.

### Silhouette Coefficient

The silhouette coefficient measures the similarity of a sample to its own cluster compared to other clusters. The range is [-1, 1], and the higher the value, the better the clustering result.

$$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$  

Where:  
- $S(i)$ is the silhouette coefficient of sample $i$.  
- $a(i)$ is the average distance from sample $i$ to other samples in the same cluster.  
- $b(i)$ is the average distance from sample $i$ to the nearest other cluster.  

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate example data
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# Calculate silhouette scores for different K values
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K = {k}, Silhouette Coefficient = {score:.3f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for Different K Values')
plt.grid(True)
plt.show()
```

  </div>
</div>

### Davies-Bouldin Index

The Davies-Bouldin index measures the ratio of the dispersion within clusters to the distance between clusters. A smaller value indicates better clustering.

$$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

Where:
- $k$ is the number of clusters
- $\sigma_i$ is the average distance from samples in cluster $i$ to the cluster center
- $d(c_i, c_j)$ is the distance between the centers of clusters $i$ and $j$

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin index for different K values
db_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    db_scores.append(score)
    print(f"K = {k}, Davies-Bouldin Index = {score:.3f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, db_scores, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index for Different K Values')
plt.grid(True)
plt.show()
```

  </div>
</div>

### Elbow Method

The Elbow method determines the optimal number of clusters by calculating the within-cluster sum of squares (WCSS) for different K values. The optimal K is typically the value where increasing K no longer significantly reduces WCSS.

$$WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $C_i$ is the $i$-th cluster
- $\mu_i$ is the center of the $i$-th cluster
- $||x - \mu_i||^2$ is the squared Euclidean distance from sample $x$ to the center $\mu_i$

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Calculate WCSS for different K values
wcss = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(f"K = {k}, WCSS = {kmeans.inertia_:.3f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()
```

  </div>
</div>

### Calinski-Harabasz Index

Also known as the Variance Ratio Criterion (VRC), the Calinski-Harabasz index computes the ratio of between-cluster dispersion to within-cluster dispersion. A larger value indicates better clustering.

$$CH = \frac{SS_B}{SS_W} \times \frac{N-k}{k-1}$$

Where:
- $SS_B$ is the between-cluster sum of squares
- $SS_W$ is the within-cluster sum of squares
- $N$ is the total number of samples
- $k$ is the number of clusters

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import calinski_harabasz_score

# Calculate Calinski-Harabasz index for different K values
ch_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = calinski_harabasz_score(X, labels)
    ch_scores.append(score)
    print(f"K = {k}, Calinski-Harabasz Index = {score:.3f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, ch_scores, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index for Different K Values')
plt.grid(True)
plt.show()
```

  </div>
</div>

## External Evaluation Metrics

External evaluation metrics require known true labels to assess the clustering results, and are typically used in research or benchmarking tests.

### Adjusted Rand Index (ARI)

The Adjusted Rand Index measures the similarity between two clustering results. The range is [-1, 1], and the higher the value, the closer the clustering result is to the true labels.

$$ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}{\frac{1}{2}[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}] - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}$$

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs

# Generate data with true labels
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)

# Use K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Calculate Adjusted Rand Index
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari:.3f}")
```

  </div>
</div>

### Adjusted Mutual Information (AMI)

The Adjusted Mutual Information measures the mutual information between the clustering result and the true labels. The range is [0, 1], and the higher the value, the better the clustering result.

$$AMI = \frac{MI(U, V) - E[MI(U, V)]}{\max(H(U), H(V)) - E[MI(U, V)]}$$

Where:
- $MI(U, V)$ is the mutual information
- $H(U)$ and $H(V)$ are the entropy
- $E[MI(U, V)]$ is the expected mutual information

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import adjusted_mutual_info_score

# Calculate Adjusted Mutual Information
ami = adjusted_mutual_info_score(y_true, y_pred)
print(f"Adjusted Mutual Information: {ami:.3f}")
```

  </div>
</div>

### Homogeneity, Completeness, and V-measure

- **Homogeneity**: Each cluster contains only samples of a single class
- **Completeness**: All samples of the same class are in the same cluster
- **V-measure**: The harmonic mean of homogeneity and completeness

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# Calculate homogeneity, completeness, and V-measure
homogeneity = homogeneity_score(y_true, y_pred)
completeness = completeness_score(y_true, y_pred)
v_measure = v_measure_score(y_true, y_pred)

print(f"Homogeneity: {homogeneity:.3f}")
print(f"Completeness: {completeness:.3f}")
print(f"V-measure: {v_measure:.3f}")
```

  </div>
</div>

## Choosing Evaluation Metrics

Choosing the appropriate evaluation metric involves considering the following factors:

1. **Whether True Labels Are Available**: If labels are available, external metrics can be used; otherwise, only internal metrics can be applied.
2. **Data Characteristics**: Clusters of different shapes, densities, and sizes may require different metrics.
3. **Clustering Algorithm**: Some metrics may favor certain types of clustering algorithms.
4. **Computational Complexity**: Some metrics may be very time-consuming for large datasets.

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Evaluation Metric</th>
        <th>Type</th>
        <th>Range</th>
        <th>Optimal Value</th>
        <th>Applicable Scenarios</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Silhouette Coefficient</td>
        <td>Internal</td>
        <td>[-1, 1]</td>
        <td>Close to 1</td>
        <td>Convex clusters, large inter-cluster distance</td>
      </tr>
      <tr>
        <td>Davies-Bouldin Index</td>
        <td>Internal</td>
        <td>[0, ∞)</td>
        <td>Close to 0</td>
        <td>Convex clusters, assessing different K values</td>
      </tr>
      <tr>
        <td>Elbow Method (WCSS)</td>
        <td>Internal</td>
        <td>[0, ∞)</td>
        <td>Elbow point</td>
        <td>Determining K value, K-means clustering</td>
      </tr>
      <tr>
        <td>Calinski-Harabasz Index</td>
        <td>Internal</td>
        <td>[0, ∞)</td>
        <td>The larger the better</td>
        <td>Convex clusters, large inter-cluster distance</td>
      </tr>
      <tr>
        <td>Adjusted Rand Index</td>
        <td>External</td>
        <td>[-1, 1]</td>
        <td>Close to 1</td>
        <td>With true labels, assessing clustering quality</td>
      </tr>
      <tr>
        <td>Adjusted Mutual Information</td>
        <td>External</td>
        <td>[0, 1]</td>
        <td>Close to 1</td>
        <td>With true labels, assessing information retention</td>
      </tr>
      <tr>
        <td>V-measure</td>
        <td>External</td>
        <td>[0, 1]</td>
        <td>Close to 1</td>
        <td>With true labels, balancing homogeneity and completeness</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>Common Pitfalls
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Relying solely on one metric</strong>: Different metrics reflect different aspects of clustering quality</li>
      <li><strong>Ignoring data characteristics</strong>: Some metrics may perform poorly on clusters of specific shapes</li>
      <li><strong>Over-interpreting evaluation results</strong>: Evaluation metrics are auxiliary tools, not absolute standards</li>
      <li><strong>Ignoring domain knowledge</strong>: The actual meaning of clustering results is more important than mathematical metrics</li>
    </ul>
  </div>
</div>

## Summary and Reflection

Clustering evaluation is a crucial step in clustering analysis that helps us select the appropriate clustering algorithm and parameters.

### Key Takeaways

- Clustering evaluation can be divided into internal and external metrics
- Internal metrics rely on the data's characteristics and do not require true labels
- External metrics require true labels and are typically used in research or benchmarking tests
- Different evaluation metrics apply to different data characteristics and clustering algorithms
- Using multiple evaluation metrics can provide a more comprehensive assessment of clustering quality

### Reflection Questions

1. How can the quality of clustering results be determined without true labels?
2. Different evaluation metrics may give different optimal K values. How should such situations be handled?
3. How can domain knowledge be incorporated into the clustering evaluation process?

<BackToPath />

<div class="practice-link">
  <a href="/projects/clustering.html" class="button">Go to Practice Project</a>
</div>
