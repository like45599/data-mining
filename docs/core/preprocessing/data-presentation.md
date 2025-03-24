# 数据表示方法

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解不同数据类型的特性与表示方法</li>
      <li>掌握常见数据结构的使用场景</li>
      <li>学习数据表示的标准化和归一化方法</li>
      <li>了解特征工程的基础技术</li>
    </ul>
  </div>
</div>

## 数据类型介绍

在数据挖掘中，数据类型的正确识别和处理是成功的关键。以下是常见的数据类型：

### 1. 数值型数据

数值型数据可以分为连续型和离散型：

- **连续数值**：可以取任意实数值，如温度（25.5°C）、身高（175.2cm）
- **离散数值**：只能取特定的值，通常是整数，如年龄（18岁）、数量（5个）

**处理建议**：通常需要进行标准化以消除量纲影响，常用方法有：

- **Z-Score标准化**：$z = \frac{x - \mu}{\sigma}$ 其中$\mu$是均值，$\sigma$是标准差

- **Min-Max归一化**：$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建示例数据
df = pd.DataFrame({'height': [165, 170, 175, 180, 185]})

# Z-Score标准化
scaler = StandardScaler()
df['height_zscore'] = scaler.fit_transform(df[['height']])

# Min-Max归一化
min_max_scaler = MinMaxScaler()
df['height_minmax'] = min_max_scaler.fit_transform(df[['height']])

print(df)
```

  </div>
</div>

### 2. 类别型数据

类别型数据表示分类信息，分为：

- **名义变量**：没有内在顺序，如性别（男/女）、颜色（红/蓝/绿）
- **有序变量**：有明确顺序，如教育程度（小学/中学/大学）、满意度（低/中/高）

**处理建议**：通常需要编码转换为数值，常用方法有：

- **One-Hot编码**：将类别转换为二进制向量，适合名义变量
- **标签编码**：将类别映射为整数，适合有序变量
- **目标编码**：根据目标变量的均值替换类别，适合高基数特征

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 创建示例数据
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'green'],
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

# One-Hot编码
encoder = OneHotEncoder(sparse=False)
color_encoded = encoder.fit_transform(df[['color']])
color_df = pd.DataFrame(
    color_encoded, 
    columns=[f'color_{c}' for c in encoder.categories_[0]]
)

# 标签编码
label_encoder = LabelEncoder()
df['size_encoded'] = label_encoder.fit_transform(df['size'])

# 合并结果
result = pd.concat([df, color_df], axis=1)
print(result)
```

  </div>
</div>

### 3. 时间序列数据

时间序列数据随时间变化，具有时序特性：

- **时间戳**：特定时间点的观测值，如股票价格、传感器读数
- **时间区间**：跨越一段时间的数据，如通话时长、活动持续时间
- **周期性数据**：具有重复模式，如季节性销售、每日温度变化

**处理建议**：

- 提取时间特征（年、月、日、小时、工作日等）
- 滑动窗口聚合（均值、最大值、最小值等）
- 处理季节性和趋势

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np

# 创建时间序列数据
date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['value'] = np.random.randint(0, 100, size=len(date_rng))

# 提取时间特征
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 滑动窗口
df['rolling_mean_3d'] = df['value'].rolling(window=3).mean()

print(df.head())
```

  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>在数据科学的早期，数据表示方法主要是基于关系数据库模型。直到1970年代，E.F. Codd提出了关系模型，才开始系统地思考数据表示问题。现代数据表示方法融合了统计学、计算机科学和领域知识，形成了丰富的数据预处理技术体系。</p>
  </div>
</div>

## 数据结构与存储

### 1. 表格型数据

最常见的数据形式，如电子表格和关系数据库表：

- **行（记录）**：代表单个实体或实例
- **列（特征）**：代表实体的属性或特征
- **单元格**：特定实体的特定属性值

<div class="code-example">
  <div class="code-example__title">Python实现</div>
  <div class="code-example__content">

```python
import pandas as pd

# 创建DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Boston', 'Chicago']
}
df = pd.DataFrame(data)
print(df)
```

  </div>
</div>

### 2. 矩阵和张量

支持高级计算和算法实现：

- **矩阵**：二维数组，如图像数据、距离矩阵
- **张量**：多维数组，常用于深度学习

<div class="code-example">
  <div class="code-example__title">Python实现</div>
  <div class="code-example__content">

```python
import numpy as np

# 创建矩阵
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("矩阵:\n", matrix)

# 创建3D张量
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("张量:\n", tensor)
```

  </div>
</div>

### 3. 图结构

表示实体之间的关系：

- **节点（顶点）**：代表实体
- **边**：代表实体间的关系
- **权重**：关系的强度或重要性

<div class="interactive-component">
  <div class="interactive-component__title">图结构可视化</div>
  <div class="interactive-component__content">
    <graph-visualization></graph-visualization>
  </div>
</div>

<div class="code-example">
  <div class="code-example__title">Python实现</div>
  <div class="code-example__content">

```python
import networkx as nx
import matplotlib.pyplot as plt

# 创建图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])

# 可视化
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=15, font_weight='bold')
plt.title("简单的图结构")
plt.show()
```

  </div>
</div>

## 特征表示技术

### 1. 特征缩放

确保不同特征的量纲一致，避免某些特征主导模型：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化: 均值为0，标准差为1
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

# 归一化: 缩放到[0,1]区间
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

# 稳健缩放: 对异常值不敏感
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

  </div>
</div>

<div class="interactive-component">
  <div class="interactive-component__title">特征缩放效果对比</div>
  <div class="interactive-component__content">
    <feature-scaling-demo></feature-scaling-demo>
  </div>
</div>

### 2. 特征编码

将非数值特征转换为算法可用的数值表示：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# 标签编码: 适用于有序分类变量
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-Hot编码: 适用于无序分类变量
onehot_encoder = OneHotEncoder()
X_encoded = onehot_encoder.fit_transform(X_categorical)

# 序数编码: 适用于有序分类变量，保留顺序信息
ordinal_encoder = OrdinalEncoder(categories=[['低', '中', '高']])
X_ordinal = ordinal_encoder.fit_transform(X_categorical)
```

  </div>
</div>

### 3. 特征选择

从原始特征集中选择最相关或最重要的特征：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# 基于统计检验的特征选择
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# 基于模型的特征选择
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

  </div>
</div>

## 数据可视化与探索

数据可视化是理解数据分布和关系的重要工具：

### 1. 单变量分析

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

# 数值分布
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, len(numeric_cols), i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} 分布')
plt.tight_layout()
plt.show()
```

  </div>
</div>

### 2. 多变量分析

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 相关性热图
plt.figure(figsize=(10, 8))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('特征相关性')
plt.show()

# 散点图矩阵
sns.pairplot(df[numeric_cols + ['target']], hue='target')
plt.suptitle('特征对之间的关系', y=1.02)
plt.show()
```

  </div>
</div>

<div class="interactive-component">
  <div class="interactive-component__title">交互式数据探索</div>
  <div class="interactive-component__content">
    <data-explorer></data-explorer>
  </div>
</div>

## 实践技巧

### 1. 处理高基数类别特征

当类别特征有大量唯一值时：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 频次编码
frequency_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(frequency_map)

# 分组稀有类别
def group_rare_categories(series, threshold=0.05):
    value_counts = series.value_counts(normalize=True)
    # 找出占比小于阈值的类别
    rare_categories = value_counts[value_counts < threshold].index.tolist()
    # 替换稀有类别
    return series.replace(rare_categories, 'Other')

df['category_grouped'] = group_rare_categories(df['category'])
```

  </div>
</div>

### 2. 数据泄露预防

避免模型训练中的数据泄露：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 错误示范：使用全部数据进行标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 正确做法：仅使用训练集进行拟合
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 仅使用训练数据拟合
X_test_scaled = scaler.transform(X_test)  # 使用训练数据的参数转换测试数据
```

  </div>
</div>

### 3. 高效数据处理技巧

处理大型数据集时的优化方法：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 减少内存使用
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'初始内存使用: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'优化后内存使用: {end_mem:.2f} MB')
    print(f'减少了: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# 使用示例
df_optimized = reduce_mem_usage(df)
```

  </div>
</div>

## 小结与思考

数据表示是数据挖掘的基础步骤，良好的数据表示可以显著提高后续模型的性能。在实际应用中，需要根据数据特点和问题需求，选择合适的表示方法。

### 关键要点回顾

- 不同数据类型需要不同的处理方法
- 特征缩放可以消除量纲影响
- 类别特征需要转换为数值形式
- 时间序列数据需要提取时间特征
- 数据可视化有助于理解数据分布和关系

### 思考问题

1. 在处理类别特征时，如何选择合适的编码方法？
2. 特征缩放对哪些机器学习算法影响较大？
3. 如何处理既有数值特征又有类别特征的混合数据集？

<BackToPath />

<div class="practice-link">
  <a href="/projects/preprocessing.html" class="button">前往实践项目</a>
</div> 