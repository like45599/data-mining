# 缺失值处理

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>了解缺失值产生的原因及其影响</li>
      <li>掌握缺失值检测与分析方法</li>
      <li>学习多种缺失值处理策略及其适用场景</li>
      <li>实践常用的缺失值处理技术</li>
    </ul>
  </div>
</div>

## 缺失值概述

缺失值是数据分析和机器学习中常见的问题，有效处理缺失值对模型性能至关重要。

### 缺失值产生的原因

缺失值通常由以下原因导致：

1. **数据收集问题**：如调查问卷未回答、传感器故障
2. **数据整合问题**：合并多个来源的数据时信息不完整
3. **数据处理错误**：导入、转换或清洗过程中的错误
4. **隐私保护**：有意隐藏某些敏感信息
5. **结构性缺失**：某些条件下不需要收集的数据

### 缺失值的类型

根据缺失机制，缺失值可分为三类：

1. **完全随机缺失(MCAR, Missing Completely At Random)**
   - 缺失完全随机，与任何观测或未观测变量无关
   - 例如：实验设备随机故障导致的数据丢失

2. **随机缺失(MAR, Missing At Random)**
   - 缺失概率只与观测到的其他变量有关
   - 例如：年龄较大的人更可能不回答收入问题

3. **非随机缺失(MNAR, Missing Not At Random)**
   - 缺失概率与未观测到的变量或缺失值本身有关
   - 例如：高收入人群不愿透露收入

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>统计学家唐纳德·鲁宾(Donald Rubin)于1976年首次系统地提出了缺失数据机制的分类(MCAR、MAR、MNAR)，这一理论框架至今仍是处理缺失值的基础。不同类型的缺失机制需要不同的处理策略，选择合适的方法对分析结果的可靠性至关重要。</p>
  </div>
</div>

## 缺失值分析

### 1. 检测缺失值

在处理缺失值前，首先需要全面了解数据中缺失值的情况：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('data.csv')

# 检查每列的缺失值数量
missing_values = df.isnull().sum()
print(missing_values)

# 计算缺失比例
missing_ratio = df.isnull().sum() / len(df) * 100
print(missing_ratio)

# 可视化缺失值
plt.figure(figsize=(12, 6))
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
sns.barplot(x=missing_ratio.index, y=missing_ratio.values)
plt.title('缺失值比例')
plt.xticks(rotation=45)
plt.ylabel('缺失百分比')
plt.show()
```

  </div>
</div>

### 2. 缺失模式分析

理解缺失值之间的关系有助于选择合适的处理策略：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 缺失值热图
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('缺失值分布热图')
plt.show()

# 缺失值相关性
missing_binary = df.isnull().astype(int)
plt.figure(figsize=(10, 8))
sns.heatmap(missing_binary.corr(), annot=True, cmap='coolwarm')
plt.title('缺失值相关性热图')
plt.show()
```

  </div>
</div>

<div class="visualization-container">
  <div class="visualization-title">缺失值模式可视化</div>
  <div class="visualization-content">
    <img src="/images/missing_pattern_en.svg" alt="缺失值模式可视化">
  </div>
  <div class="visualization-caption">
    图: 缺失值分布热图。黄色区域表示缺失值，可以观察到缺失模式的分布情况。
  </div>
</div>

## 缺失值处理策略

### 1. 删除含缺失值的数据

最简单的处理方法，但可能导致信息丢失：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 删除所有含缺失值的行
df_dropped_rows = df.dropna()

# 删除缺失比例超过50%的列
threshold = len(df) * 0.5
df_dropped_cols = df.dropna(axis=1, thresh=threshold)

# 只删除特定列缺失的行
df_dropped_specific = df.dropna(subset=['income', 'age'])
```

  </div>
</div>

**适用场景**：
- 缺失值比例很小（<5%）
- 数据量充足，删除不会显著减少样本量
- 缺失值完全随机(MCAR)

**缺点**：
- 可能引入偏差，特别是当缺失不是完全随机时
- 减少样本量，降低统计功效
- 可能丢失重要信息

### 2. 填充缺失值

#### 2.1 统计量填充

使用统计量代替缺失值：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 均值填充
df_mean = df.copy()
df_mean.fillna(df_mean.mean(), inplace=True)

# 中位数填充
df_median = df.copy()
df_median.fillna(df_median.median(), inplace=True)

# 众数填充
df_mode = df.copy()
for col in df_mode.columns:
    if df_mode[col].dtype == 'object':  # 类别变量
        df_mode[col].fillna(df_mode[col].mode()[0], inplace=True)
```

  </div>
</div>

<div class="interactive-component">
  <div class="interactive-title">填充方法比较</div>
  <div class="interactive-content">
    <missing-value-imputation></missing-value-imputation>
  </div>
  <div class="interactive-caption">
    交互组件：尝试不同的填充方法，观察对数据分布的影响。
  </div>
</div>

#### 2.2 高级填充方法

利用数据间的关系进行更智能的填充：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN填充
imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# 多重插补（使用MICE算法的简化版本）
imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

  </div>
</div>

## 缺失值处理评估

### 1. 比较不同方法的效果

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 原始数据分布
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['income'].dropna(), kde=True)
plt.title('原始数据分布')

# 均值填充后的分布
plt.subplot(1, 3, 2)
sns.histplot(df_mean['income'], kde=True)
plt.title('均值填充后的分布')

# KNN填充后的分布
plt.subplot(1, 3, 3)
sns.histplot(df_knn['income'], kde=True)
plt.title('KNN填充后的分布')

plt.tight_layout()
plt.show()
```

  </div>
</div>

## 实践建议

### 1. 缺失值处理流程

1. **探索性分析**：
   - 检测缺失值的位置、数量和比例
   - 分析缺失模式和机制
   - 可视化缺失值分布

2. **制定处理策略**：
   - 根据缺失机制选择合适的处理方法
   - 考虑特征的重要性和数据结构
   - 可能需要不同特征采用不同策略

3. **实施与评估**：
   - 实施选定的填充方法
   - 比较不同方法的效果
   - 验证处理后数据的质量和一致性

### 2. 实际应用技巧

- **先分析后处理**：深入了解缺失原因再选择方法
- **特征相关性**：利用特征间关系改进填充效果
- **领域知识**：结合业务理解指导缺失值处理
- **敏感性分析**：测试不同填充方法对最终结果的影响
- **保留不确定性**：考虑使用多重插补保留估计的不确定性

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>盲目删除</strong>：不分析缺失机制就直接删除含缺失值的样本</li>
      <li><strong>过度依赖均值</strong>：对所有特征都使用均值填充，忽略数据分布特性</li>
      <li><strong>忽略缺失值相关性</strong>：未考虑特征间关系进行填充</li>
      <li><strong>数据泄露</strong>：使用测试集信息填充训练集缺失值</li>
    </ul>
  </div>
</div>

## 小结与思考

缺失值处理是数据预处理中的关键步骤，合适的处理方法可以提高模型性能并确保分析结果的可靠性。

### 关键要点回顾

- 缺失值可能由多种原因导致，包括数据收集问题、隐私保护等
- 缺失机制分为MCAR、MAR和MNAR三种类型
- 处理策略包括删除法和多种填充方法
- 选择合适的处理方法需要考虑缺失机制、数据特性和分析目标

### 思考问题

1. 如何判断数据中的缺失机制类型？
2. 在什么情况下，删除含缺失值的样本比填充更合适？
3. 如何评估缺失值处理方法的有效性？

<div class="practice-link">
  <a href="/projects/preprocessing.html" class="button">前往实践项目</a>
</div> 