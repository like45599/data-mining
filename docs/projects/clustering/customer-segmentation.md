# 客户分群分析

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：中级</li>
      <li><strong>类型</strong>：聚类分析</li>
      <!-- <li><strong>预计时间</strong>：5-7小时</li> -->
      <li><strong>技能点</strong>：数据标准化、K-Means聚类、聚类评估、业务解释</li>
      <li><strong>对应知识模块</strong>：<a href="/core/clustering/kmeans.html">聚类算法</a></li>
    </ul>
  </div>
</div>

## 项目背景

客户分群是企业理解客户行为和偏好的重要策略，通过将相似的客户分组，企业可以开发针对性的营销策略、产品推荐和服务方案。传统的客户分群通常基于人口统计学特征（如年龄、性别、收入），但现代分析方法可以结合交易历史、浏览行为和互动模式等多维数据，创建更精细和有意义的客户群体。

在这个项目中，我们将使用聚类算法对零售商的客户数据进行分群，发现不同客户群体的特征和行为模式，为业务决策提供支持。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>RFM分析是客户分群的经典方法，基于三个关键指标：最近一次购买时间(Recency)、购买频率(Frequency)和消费金额(Monetary)。这种简单而有效的方法至今仍被广泛应用，并可以与现代聚类算法结合，创建更精细的客户画像。</p>
  </div>
</div>

## 数据集介绍

本项目使用的数据集包含一家在线零售商一年内的交易记录，共有约50,000条交易和以下字段：

- **CustomerID**：客户唯一标识符
- **InvoiceNo**：发票编号
- **InvoiceDate**：交易日期和时间
- **StockCode**：产品代码
- **Description**：产品描述
- **Quantity**：购买数量
- **UnitPrice**：单价
- **Country**：客户所在国家

我们将从这些交易数据中提取客户级别的特征，然后应用聚类算法进行分群。

## 项目目标

1. 从交易数据中提取客户级别的特征
2. 应用K-Means等聚类算法对客户进行分群
3. 评估聚类结果并确定最佳簇数
4. 分析不同客户群体的特征和行为模式
5. 提出基于分群结果的业务建议

## 实施步骤

### 步骤1：数据加载与预处理

首先，我们加载数据并进行必要的预处理。

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

# 加载数据
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# 查看数据基本信息
print(df.info())
print(df.head())

# 数据清洗
# 移除缺失的CustomerID
df = df.dropna(subset=['CustomerID'])

# 移除取消的订单（Quantity < 0）
df = df[df['Quantity'] > 0]

# 移除单价为0或负数的记录
df = df[df['UnitPrice'] > 0]

# 将CustomerID转换为整数
df['CustomerID'] = df['CustomerID'].astype(int)

# 将InvoiceDate转换为日期时间格式
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 创建总金额列
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# 查看清洗后的数据
print("清洗后的数据形状:", df.shape)
print(df.describe())

# 查看每个国家的客户数量
country_counts = df.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar')
plt.title('各国家客户数量')
plt.xlabel('国家')
plt.ylabel('客户数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 步骤2：特征提取 - RFM分析

接下来，我们使用RFM方法从交易数据中提取客户级别的特征。

```python
# 设置分析截止日期（数据集中的最后一天加1天）
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# 计算RFM指标
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
})

# 重命名列
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 查看RFM数据
print(rfm.head())
print(rfm.describe())

# 可视化RFM分布
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(rfm['Recency'], bins=30, kde=True)
plt.title('Recency分布')

plt.subplot(1, 3, 2)
sns.histplot(rfm['Frequency'], bins=30, kde=True)
plt.title('Frequency分布')

plt.subplot(1, 3, 3)
sns.histplot(rfm['Monetary'], bins=30, kde=True)
plt.title('Monetary分布')
plt.tight_layout()
plt.show()

# 查看RFM指标之间的相关性
plt.figure(figsize=(10, 8))
sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('RFM指标相关性')
plt.show()
```

### 步骤3：特征处理与标准化

在应用聚类算法之前，我们需要处理异常值并标准化特征。

```python
# 处理异常值
# 使用IQR方法识别异常值
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

# 应用异常值处理
rfm_clean = remove_outliers(rfm, ['Recency', 'Frequency', 'Monetary'])
print(f"移除异常值前: {rfm.shape[0]}行, 移除后: {rfm_clean.shape[0]}行")

# 对数变换处理偏斜分布
rfm_log = rfm_clean.copy()
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

# 可视化变换后的分布
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(rfm_log['Recency'], bins=30, kde=True)
plt.title('Recency分布')

plt.subplot(1, 3, 2)
sns.histplot(rfm_log['Frequency'], bins=30, kde=True)
plt.title('Log(Frequency)分布')

plt.subplot(1, 3, 3)
sns.histplot(rfm_log['Monetary'], bins=30, kde=True)
plt.title('Log(Monetary)分布')
plt.tight_layout()
plt.show()

# 准备聚类特征
X = rfm_log.copy()
features = ['Recency', 'Frequency', 'Monetary']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换回DataFrame以便于分析
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
print(X_scaled_df.head())
```

### 步骤4：确定最佳簇数

使用肘部法则和轮廓系数确定最佳簇数。

```python
# 肘部法则
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 可视化肘部法则
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('簇数 (k)')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('簇数 (k)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法')
plt.grid(True)
plt.tight_layout()
plt.show()

# 根据结果选择最佳簇数
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"最佳簇数 (基于轮廓系数): {best_k}")
```

### 步骤5：应用K-Means聚类

使用确定的最佳簇数应用K-Means聚类。

```python
# 应用K-Means聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# 查看每个簇的大小
cluster_sizes = rfm['Cluster'].value_counts().sort_index()
print("各簇大小:")
print(cluster_sizes)

# 计算每个簇的中心点（原始尺度）
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                              columns=features)
cluster_centers['Cluster'] = range(best_k)
print("\n簇中心点:")
print(cluster_centers)

# 可视化聚类结果（使用PCA降维到2D）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for i in range(best_k):
    plt.scatter(X_pca[rfm['Cluster'] == i, 0], X_pca[rfm['Cluster'] == i, 1], label=f'Cluster {i}')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], 
            s=300, c='red', marker='X', label='Centroids')
plt.title('客户分群 (PCA降维可视化)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend()
plt.grid(True)
plt.show()
```

### 步骤6：分析客户群体特征

分析每个客户群体的特征，并为业务提供建议。

```python
# 分析每个簇的RFM特征
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

# 可视化每个簇的特征
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.barplot(x='Cluster', y='Recency', data=cluster_analysis)
plt.title('各簇的平均最近购买天数')

plt.subplot(1, 3, 2)
sns.barplot(x='Cluster', y='Frequency', data=cluster_analysis)
plt.title('各簇的平均购买频率')

plt.subplot(1, 3, 3)
sns.barplot(x='Cluster', y='Monetary', data=cluster_analysis)
plt.title('各簇的平均消费金额')
plt.tight_layout()
plt.show()

# 雷达图可视化各簇特征
from math import pi

# 标准化数据用于雷达图
radar_df = cluster_analysis.copy()
for feature in features:
    radar_df[feature] = (radar_df[feature] - radar_df[feature].min()) / (radar_df[feature].max() - radar_df[feature].min())
    # 对于Recency，值越小越好，所以取反
    if feature == 'Recency':
        radar_df[feature] = 1 - radar_df[feature]

# 设置雷达图
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
plt.title('客户群体特征雷达图')
plt.show()

# 客户群体命名和业务建议
cluster_names = {
    0: "高价值忠诚客户",
    1: "潜力客户",
    2: "休眠客户",
    3: "新客户"
}

business_recommendations = {
    0: "提供VIP服务，发送个性化优惠，鼓励推荐新客户",
    1: "提供购买激励，增加购买频率，推荐相关产品",
    2: "发送重新激活邮件，提供特别折扣，了解流失原因",
    3: "提供良好的首次体验，鼓励第二次购买，收集反馈"
}

# 创建最终的客户分群报告
cluster_report = pd.DataFrame({
    'Cluster': range(best_k),
    'Name': [cluster_names.get(i, f"Cluster {i}") for i in range(best_k)],
    'Size': cluster_sizes.values,
    'Recency': cluster_analysis['Recency'],
    'Frequency': cluster_analysis['Frequency'],
    'Monetary': cluster_analysis['Monetary'],
    'Business Recommendations': [business_recommendations.get(i, "") for i in range(best_k)]
})

print("客户分群报告:")
print(cluster_report)
```

## 结果分析

通过K-Means聚类算法，我们成功将客户分为几个不同的群体，每个群体具有不同的购买行为特征：

1. **高价值忠诚客户**：这些客户最近有购买，购买频率高，消费金额大。他们是企业的核心客户群体，贡献了大部分收入。
2. **潜力客户**：这些客户购买频率中等，消费金额适中，有成为高价值客户的潜力。
3. **休眠客户**：这些客户很久没有购买了，但之前有过较好的购买记录。他们是重新激活营销的目标。
4. **新客户**：这些客户最近才开始购买，频率和金额都较低。需要特别关注以提高留存率。

这种客户分群可以帮助企业制定针对性的营销策略，提高客户满意度和忠诚度，最终提升销售业绩和客户终身价值。

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **扩展特征集**：除了RFM指标，尝试加入产品类别偏好、购买时间模式等特征
2. **尝试其他聚类算法**：比较层次聚类、DBSCAN等算法与K-Means的结果差异
3. **时间序列分析**：分析客户群体随时间的变化，识别客户迁移模式
4. **预测模型**：基于聚类结果构建客户流失预测或下一次购买预测模型
5. **推荐系统**：为每个客户群体开发定制的产品推荐策略

## 小结与反思

通过这个项目，我们学习了如何使用聚类算法对客户进行分群，并从数据中提取有价值的业务洞察。客户分群是理解客户行为和偏好的强大工具，可以帮助企业优化营销策略、提高客户满意度和忠诚度。

在实际应用中，客户分群应该是一个持续的过程，随着新数据的积累和业务环境的变化，分群结果需要定期更新。此外，分群结果的解释和应用需要结合具体的业务背景和目标，才能发挥最大价值。

### 思考问题

1. 除了RFM指标，还有哪些客户特征可能对分群有价值？如何获取和整合这些特征？
2. 不同行业的客户分群可能有什么差异？例如，电子商务、金融服务和内容订阅服务的客户分群应该关注哪些不同的指标？
3. 如何评估客户分群的业务价值？有哪些指标可以衡量分群策略的有效性？

<div class="practice-link">
  <a href="/projects/clustering/image-segmentation.html" class="button">下一个项目：图像颜色分割</a>
</div> 