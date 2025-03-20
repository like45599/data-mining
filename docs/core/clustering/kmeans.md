# K-Means聚类算法

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解K-Means算法的基本原理和工作流程</li>
      <li>掌握K值选择的方法和评估指标</li>
      <li>学习K-Means的优化变体和局限性</li>
      <li>实践K-Means在客户分群等场景中的应用</li>
    </ul>
  </div>
</div>

## K-Means基本原理

K-Means是一种经典的无监督学习算法，用于将数据划分为K个不同的簇。算法的核心思想是：每个样本应该属于距离最近的簇中心（质心）所代表的簇。

### 算法流程

K-Means算法的基本步骤如下：

1. **初始化**：随机选择K个点作为初始质心
2. **分配**：将每个样本分配到距离最近的质心所代表的簇
3. **更新**：重新计算每个簇的质心（簇中所有点的均值）
4. **迭代**：重复步骤2和3，直到质心基本不再变化或达到最大迭代次数

<div class="visualization-container">
  <div class="visualization-title">K-Means聚类过程</div>
  <div class="visualization-content">
    <img src="/images/kmeans_process.svg" alt="K-Means聚类过程">
  </div>
  <div class="visualization-caption">
    图: K-Means聚类过程。从随机初始化质心开始，通过迭代分配和更新步骤，最终收敛到稳定的簇划分。
  </div>
</div>

### 数学表达

K-Means算法的目标是最小化所有样本到其所属簇质心的距离平方和，即最小化以下目标函数：

$$J = \sum_{j=1}^{k} \sum_{i=1}^{n_j} ||x_i^{(j)} - c_j||^2$$

其中：
- $k$ 是簇的数量
- $n_j$ 是第j个簇中的样本数
- $x_i^{(j)}$ 是第j个簇中的第i个样本
- $c_j$ 是第j个簇的质心

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>K-Means算法最早由Stuart Lloyd在1957年提出（作为脉冲编码调制的一种技术），但直到1982年才正式发表。尽管算法简单，但它在各个领域仍然广泛应用，是聚类分析的基准算法。K-Means是一种贪心算法，它保证收敛到局部最优解，但不一定是全局最优解。</p>
  </div>
</div>

## K-Means实现与应用

### 基本实现

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建并训练K-Means模型
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 获取簇中心
centers = kmeans.cluster_centers_

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.show()
```

  </div>
</div>

### 确定最佳K值

选择合适的K值是K-Means算法的关键。常用的方法包括肘部法则和轮廓系数：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 计算不同K值的SSE（误差平方和）
sse = []
silhouette_scores = []
range_k = range(2, 11)

for k in range_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    
    # 计算轮廓系数
    if k > 1:  # 轮廓系数需要至少2个簇
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))

# 绘制肘部图
plt.figure(figsize=(12, 5))

# SSE图
plt.subplot(1, 2, 1)
plt.plot(range_k, sse, 'bo-')
plt.xlabel('K值')
plt.ylabel('SSE')
plt.title('肘部法则')

# 轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('K值')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法')

plt.tight_layout()
plt.show()
```

  </div>
</div>

### 评估聚类质量

除了肘部法则和轮廓系数外，还可以使用其他指标评估聚类质量：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# 选择最佳K值后的模型
best_k = 4  # 假设通过肘部法则确定的最佳K值
kmeans = KMeans(n_clusters=best_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)

# 计算Calinski-Harabasz指数（越高越好）
ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz指数: {ch_score:.2f}")

# 计算Davies-Bouldin指数（越低越好）
db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin指数: {db_score:.2f}")

# 计算轮廓系数（越接近1越好）
silhouette = silhouette_score(X, labels)
print(f"轮廓系数: {silhouette:.2f}")
```

  </div>
</div>

## K-Means的应用案例

### 客户分群

K-Means在市场细分和客户分群中有广泛应用：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载客户数据（示例）
# 实际应用中，可以使用真实的客户数据
# 这里我们创建一个示例数据集
np.random.seed(42)
n_customers = 200

# 创建特征：年龄、收入和消费频率
age = np.random.normal(35, 10, n_customers)
income = np.random.normal(50000, 15000, n_customers)
frequency = np.random.normal(10, 5, n_customers)

# 创建DataFrame
customer_data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Frequency': frequency
})

# 特征缩放
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# 应用K-Means聚类
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# 分析各簇的特征
cluster_summary = customer_data.groupby('Cluster').mean()
print("各客户群体特征平均值:")
print(cluster_summary)

# 可视化结果
plt.figure(figsize=(15, 5))

# 年龄vs收入
plt.subplot(1, 3, 1)
sns.scatterplot(x='Age', y='Income', hue='Cluster', data=customer_data, palette='viridis')
plt.title('年龄 vs 收入')

# 年龄vs消费频率
plt.subplot(1, 3, 2)
sns.scatterplot(x='Age', y='Frequency', hue='Cluster', data=customer_data, palette='viridis')
plt.title('年龄 vs 消费频率')

# 收入vs消费频率
plt.subplot(1, 3, 3)
sns.scatterplot(x='Income', y='Frequency', hue='Cluster', data=customer_data, palette='viridis')
plt.title('收入 vs 消费频率')

plt.tight_layout()
plt.show()

# 为每个簇创建客户画像
for cluster in range(3):
    print(f"\n客户群体 {cluster} 画像:")
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    print(f"数量: {len(cluster_data)} 客户")
    print(f"平均年龄: {cluster_data['Age'].mean():.1f} 岁")
    print(f"平均收入: ${cluster_data['Income'].mean():.2f}")
    print(f"平均消费频率: {cluster_data['Frequency'].mean():.1f} 次/月")
```

  </div>
</div>

## K-Means的变体与优化

### K-Means++

K-Means++改进了初始质心的选择方法，使聚类结果更稳定：

1. 随机选择第一个质心
2. 对于每个后续质心，选择与已有质心距离较远的点
3. 这种方法减少了K-Means对初始值敏感的问题

### Mini-Batch K-Means

对于大规模数据集，Mini-Batch K-Means使用小批量数据进行训练，提高计算效率：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.cluster import MiniBatchKMeans
import time

# 生成大规模数据
X_large, _ = make_blobs(n_samples=10000, centers=5, cluster_std=0.6, random_state=0)

# 比较K-Means和Mini-Batch K-Means的性能
start_time = time.time()
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, random_state=0)
kmeans.fit(X_large)
kmeans_time = time.time() - start_time
print(f"K-Means运行时间: {kmeans_time:.4f}秒")

start_time = time.time()
mbk = MiniBatchKMeans(n_clusters=5, init='k-means++', batch_size=100, max_iter=100, random_state=0)
mbk.fit(X_large)
mbk_time = time.time() - start_time
print(f"Mini-Batch K-Means运行时间: {mbk_time:.4f}秒")
print(f"速度提升: {kmeans_time/mbk_time:.2f}倍")

# 比较聚类结果
kmeans_labels = kmeans.labels_
mbk_labels = mbk.labels_

# 计算两种方法的轮廓系数
kmeans_silhouette = silhouette_score(X_large, kmeans_labels)
mbk_silhouette = silhouette_score(X_large, mbk_labels)

print(f"K-Means轮廓系数: {kmeans_silhouette:.4f}")
print(f"Mini-Batch K-Means轮廓系数: {mbk_silhouette:.4f}")
```

  </div>
</div>

### 其他变体

- **K-Medoids**：使用实际数据点作为簇中心，对异常值更鲁棒
- **K-Means++**：优化初始质心选择
- **Fuzzy K-Means**：允许样本属于多个簇，每个簇有不同的隶属度
- **Spherical K-Means**：适用于文本等高维稀疏数据

## K-Means的局限性

尽管K-Means简单高效，但它有一些固有的局限性：

1. **需要预先指定K值**：选择合适的K值可能困难
2. **对初始质心敏感**：不同初始值可能导致不同结果
3. **假设簇是凸形且大小相似**：不适合识别复杂形状的簇
4. **对异常值敏感**：异常值会显著影响质心位置
5. **仅适用于数值特征**：类别特征需要特殊处理

<div class="visualization-container">
  <div class="visualization-title">K-Means的局限性</div>
  <div class="visualization-content">
    <img src="/images/kmeans_limitations.svg" alt="K-Means的局限性">
  </div>
  <div class="visualization-caption">
    图: K-Means在非凸形簇上的局限性。
  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>忽略特征缩放</strong>：未对特征进行标准化，导致高方差特征主导聚类结果</li>
      <li><strong>盲目选择K值</strong>：未使用肘部法则等方法验证K值的合理性</li>
      <li><strong>过度解读聚类结果</strong>：将聚类结果视为绝对真理，而非数据探索工具</li>
      <li><strong>忽略数据预处理</strong>：未处理异常值和缺失值，影响聚类质量</li>
    </ul>
  </div>
</div>

## 小结与思考

K-Means是一种简单而强大的聚类算法，尽管有一些局限性，但在许多实际应用中仍然非常有效。

### 关键要点回顾

- K-Means通过迭代优化将数据划分为K个簇
- 算法的目标是最小化样本到簇中心的距离平方和
- 选择合适的K值对聚类结果至关重要
- K-Means++等变体可以改进初始质心选择
- 对于大规模数据，Mini-Batch K-Means提供了计算效率

### 思考问题

1. 在什么情况下K-Means可能不是最佳选择？
2. 如何处理K-Means对异常值的敏感性？
3. 除了肘部法则和轮廓系数，还有哪些方法可以确定最佳K值？

<div class="practice-link">
  <a href="/projects/clustering.html" class="button">前往实践项目</a>
</div> 