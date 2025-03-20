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

- **Z-Score标准化**：$z = \frac{x - \mu}{\sigma}$，其中$\mu$是均值，$\sigma$是标准差
- **Min-Max归一化**：$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$

**代码示例**：

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


### 2. 类别型数据

类别型数据表示分类信息，分为：

- **名义变量**：没有内在顺序，如性别（男/女）、颜色（红/蓝/绿）
- **有序变量**：有明确顺序，如教育程度（小学/中学/大学）、满意度（低/中/高）

**处理建议**：通常需要编码转换为数值，常用方法有：

- **One-Hot编码**：将类别转换为二进制向量，适合名义变量
- **标签编码**：将类别映射为整数，适合有序变量
- **目标编码**：根据目标变量的均值替换类别，适合高基数特征

**代码示例**：
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


### 3. 时间序列数据

时间序列数据随时间变化，具有时序特性：

- **时间戳**：特定时间点的观测值，如股票价格、传感器读数
- **时间区间**：跨越一段时间的数据，如通话时长、活动持续时间
- **周期性数据**：具有重复模式，如季节性销售、每日温度变化

**处理建议**：

- 提取时间特征（年、月、日、小时、工作日等）
- 滑动窗口聚合（均值、最大值、最小值等）
- 处理季节性和趋势

**代码示例**：
```python
import pandas as pd
import numpy as np
创建时间序列数据
date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['value'] = np.random.randint(0, 100, size=len(date_rng))
提取时间特征
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
# 滑动窗口
df['rolling_mean_3d'] = df['value'].rolling(window=3).mean()
print(df.head())
```

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

**Python实现**：通常使用Pandas的DataFrame进行操作

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


### 2. 矩阵和张量

支持高级计算和算法实现：

- **矩阵**：二维数组，如图像数据、距离矩阵
- **张量**：多维数组，常用于深度学习

**Python实现**：通常使用NumPy或PyTorch/TensorFlow的张量
```python
import numpy as np
# 创建矩阵
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("矩阵:\n", matrix)
# 创建3D张量
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("张量:\n", tensor)
```


### 3. 图结构

表示实体之间的关系：

- **节点（顶点）**：代表实体
- **边**：代表实体间的关系
- **权重**：关系的强度或重要性

**Python实现**：通常使用NetworkX或PyTorch Geometric
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


## 特征表示技术

### 1. 特征缩放

确保不同特征的量纲一致，避免某些特征主导模型：
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


### 2. 特征编码

将非数值特征转换为算法可用的数值表示：
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


### 3. 特征选择

从原始特征集中选择最相关或最重要的特征：
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
# 基于统计检验的特征选择
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
#基于模型的特征选择
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```


## 实践技巧

### 1. 数据加载与检查

首先要了解数据的基本情况：
