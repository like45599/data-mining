# Clustering Practical Application Cases

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Key Takeaways
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the practical applications of clustering analysis in different fields</li>
      <li>Master the transformation method from business problems to clustering solutions</li>
      <li>Learn how to interpret clustering results and extract business value</li>
      <li>Understand the challenges and solutions of clustering analysis in real-world applications</li>
    </ul>
  </div>
</div>

## Customer Segmentation Case

Customer segmentation is one of the most common applications of clustering analysis. By dividing customers into different groups, businesses can develop targeted marketing strategies.

### Business Background

An e-commerce platform wants to analyze user behavior data and segment customers into different groups for the purpose of developing differentiated marketing strategies and personalized recommendations.

### Data Preparation

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('customer_data.csv')

# View data
print(df.head())
print(df.info())

# Feature selection
features = ['recency', 'frequency', 'monetary', 'tenure', 'age']
X = df[features]

# Handle missing values
X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# View feature correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

  </div>
</div>

### Determine the Optimal Number of Clusters

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import silhouette_score

# Use the elbow method to determine the optimal K value
wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Compute silhouette score
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Visualize elbow method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.grid(True)

# Visualize silhouette score
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.grid(True)

plt.tight_layout()
plt.show()

# Choose the best K value
best_k = 4  # Based on the above analysis
```

  </div>
</div>

### Clustering Analysis

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Perform clustering with the best K value
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize clustering results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.8)
plt.title('Customer Segmentation Result (PCA Reduction)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

# Analyze the cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print("Cluster Centers:")
print(cluster_centers)

# Statistical description of each cluster
for i in range(best_k):
    print(f"\nStatistics for Cluster {i}:")
    print(df[df['cluster'] == i][features].describe())
```

  </div>
</div>

### Business Interpretation and Application

Based on the clustering results, we can categorize customers into the following groups:

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Customer Group</th>
        <th>Feature Description</th>
        <th>Marketing Strategy</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>High Value Loyal Customers</td>
        <td>
          - High purchase frequency<br>
          - High spending<br>
          - Recently made a purchase<br>
          - Long customer tenure
        </td>
        <td>
          - VIP membership program<br>
          - Exclusive discounts<br>
          - High-end product recommendations<br>
          - Loyalty rewards
        </td>
      </tr>
      <tr>
        <td>Potential Customers</td>
        <td>
          - Medium purchase frequency<br>
          - Medium spending<br>
          - Recently made a purchase<br>
          - Short customer tenure
        </td>
        <td>
          - Membership upgrade incentives<br>
          - Cross-selling<br>
          - Personalized recommendations<br>
          - Time-limited discounts
        </td>
      </tr>
      <tr>
        <td>Dormant Customers</td>
        <td>
          - Low purchase frequency<br>
          - Medium spending<br>
          - No recent purchases<br>
          - Long customer tenure
        </td>
        <td>
          - Re-engagement campaigns<br>
          - Special discounts<br>
          - New product notifications<br>
          - Feedback surveys
        </td>
      </tr>
      <tr>
        <td>New Customers</td>
        <td>
          - Low purchase frequency<br>
          - Low spending<br>
          - Recently made a purchase<br>
          - Short customer tenure
        </td>
        <td>
          - Welcome gifts<br>
          - Entry-level product recommendations<br>
          - Automated marketing
        </td>
      </tr>
    </tbody>
  </table>
</div>

## Anomaly Detection Case

Clustering analysis can be used to detect anomalies in data, which is especially useful in fraud detection and cybersecurity.

### Business Background

A bank needs to identify potentially fraudulent transactions from a large set of transaction data.

### Data Preparation and Clustering

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Load transaction data
df = pd.read_csv('transactions.csv')

# Feature selection
features = ['amount', 'time_since_last_transaction', 'distance_from_home', 'foreign_transaction', 'high_risk_merchant']
X = df[features]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Add clustering results to the original data
df['cluster'] = clusters

# Identify anomalies (points labeled -1)
outliers = df[df['cluster'] == -1]
print(f"Detected {len(outliers)} anomalous transactions")

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.scatter(X_pca[clusters == -1, 0], X_pca[clusters == -1, 1], c='red', s=100, alpha=0.8, marker='X')
plt.title('Transaction Anomaly Detection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Analyze features of anomalous transactions
print("Feature statistics for anomalous transactions:")
print(outliers[features].describe())
print("\nFeature statistics for normal transactions:")
print(df[df['cluster'] != -1][features].describe())
```

  </div>
</div>

### Business Suggestions

Based on the anomaly detection results, the following recommendations can be made:

1. **Real-time monitoring system**: Integrate the clustering model into a real-time transaction monitoring system.
2. **Risk scoring**: Calculate an anomaly score for each transaction and trigger manual review when a threshold is exceeded.
3. **Layered defense**: Combine rule-based systems and machine learning models to build a multi-layered fraud prevention system.
4. **Continuous updates**: Periodically retrain the model with new data to adapt to changes in fraud patterns.

## Document Clustering Case

Clustering analysis can be used to organize and classify large volumes of textual documents, helping with information retrieval and topic discovery.

### Business Background

A news website needs to automatically categorize a large number of news articles to better organize content and recommend related articles.

### Text Preprocessing and Feature Extraction

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load news data
df = pd.read_csv('news_articles.csv')

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Recombine into text
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['content'].apply(preprocess_text)

# Use TF-IDF for feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text'])

# Dimensionality reduction for visualization
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Determine the best number of clusters
wcss = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualize elbow method
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# Choose the best K value
best_k = 5  # Based on the above analysis
```

  </div>
</div>

### Clustering Analysis and Topic Extraction

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Perform clustering with the best K value
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize clustering results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.8)
plt.title('News Article Clustering Result')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

# Extract keywords for each cluster
feature_names = vectorizer.get_feature_names_out()
centroids = kmeans.cluster_centers_

for i in range(best_k):
    # Get the top 10 keywords for the cluster center
    top_indices = centroids[i].argsort()[-10:][::-1]
    top_keywords = [feature_names[idx] for idx in top_indices]
    
    print(f"Cluster {i} Keywords: {', '.join(top_keywords)}")
    
    # Display sample article titles from the cluster
    print(f"Sample articles from Cluster {i}:")
    for title in df[df['cluster'] == i]['title'].head(3):
        print(f"- {title}")
    print()
```

  </div>
</div>

### Business Applications

Based on the document clustering results, the following applications can be implemented:

1. **Automatic content classification**: Automatically assign new articles to relevant categories.
2. **Related article recommendation**: Recommend other articles of the same category to users based on the current article.
3. **Topic discovery**: Identify popular topics and emerging trends.
4. **Content organization**: Optimize website navigation and content structure.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span> Common Pitfalls
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Ignoring business context</strong>: Clustering results must be interpreted in conjunction with business knowledge to make sense.</li>
      <li><strong>Over-relying on automation</strong>: Clustering is a tool for assistance, not a complete substitute for human judgment.</li>
      <li><strong>Ignoring data quality</strong>: Garbage in, garbage out—data quality is crucial for clustering results.</li>
      <li><strong>Ignoring model updates</strong>: Customer behavior and market environments change, so clustering models need regular updates.</li>
    </ul>
  </div>
</div>

## Summary and Reflection

Clustering analysis has broad applications in customer segmentation, anomaly detection, and document organization. By dividing data into meaningful groups, businesses can gain valuable insights.

### Key Takeaways

- Clustering analysis helps businesses discover natural groupings in data.
- From business problems to clustering solutions requires proper feature selection and preprocessing.
- The interpretation of clustering results must involve domain knowledge.
- Clustering analysis can support personalized marketing, risk management, and more.

### Reflection Questions

1. How can clustering results be translated into actionable business strategies?
2. How do you evaluate the business value of a clustering solution in practice?
3. How can clustering analysis be integrated with other data mining techniques?

<BackToPath />

<div class="practice-link">
  <a href="/projects/clustering.html" class="button">Go to Practice Projects</a>
</div>
