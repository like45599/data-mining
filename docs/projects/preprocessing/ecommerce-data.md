# 电商用户数据清洗与分析

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：入门级</li>
      <li><strong>类型</strong>：数据清洗与预处理</li>
      <!-- <li><strong>预计时间</strong>：3-5小时</li> -->
      <li><strong>技能点</strong>：缺失值处理、异常检测、数据转换、特征创建</li>
      <li><strong>对应知识模块</strong>：<a href="/core/preprocessing/data-presentation.html">数据预处理</a></li>
    </ul>
  </div>
</div>

## 项目背景

电子商务平台每天产生大量用户行为数据，包括浏览、搜索、加入购物车和购买等活动。这些数据对于了解用户行为模式、优化产品推荐和提高转化率至关重要。然而，原始数据通常包含缺失值、异常值和不一致的格式，需要进行清洗和预处理才能用于后续分析。

在这个项目中，我们将处理一个电商平台的用户行为数据集，通过数据清洗和预处理，为后续的用户行为分析做准备。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>数据科学家通常花费60-70%的时间在数据清洗和预处理上。高质量的数据预处理不仅可以提高模型性能，还能减少后续分析中的错误和偏差。</p>
  </div>
</div>

## 数据集介绍

本项目使用的数据集包含一个电商平台一周内的用户行为数据，共有10,000条记录和以下字段：

- **user_id**：用户ID
- **session_id**：会话ID
- **timestamp**：活动时间戳
- **page_url**：访问的页面URL
- **event_type**：事件类型（view、cart、purchase）
- **product_id**：产品ID
- **category**：产品类别
- **price**：产品价格
- **user_agent**：用户浏览器信息
- **user_region**：用户所在地区

数据集中存在多种数据质量问题，包括缺失值、异常值、格式不一致和重复记录等。

## 项目目标

1. 识别并处理数据集中的缺失值
2. 检测并处理异常值和离群点
3. 标准化和转换数据格式
4. 创建有意义的派生特征
5. 准备干净的数据集用于后续分析

## 实施步骤

### 步骤1：数据加载与初步探索

首先，我们加载数据并进行初步探索，了解数据的基本情况。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('ecommerce_data.csv')

# 查看数据基本信息
print(df.info())
print(df.describe())

# 查看前几行数据
print(df.head())

# 检查缺失值
print(df.isnull().sum())

# 检查重复记录
print(f"重复记录数: {df.duplicated().sum()}")
```

### 步骤2：处理缺失值

根据初步探索，我们发现数据集中存在缺失值，需要采取适当的策略处理。

```python
# 检查每列的缺失比例
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)

# 处理缺失的产品价格 - 使用同类别产品的中位数填充
df['price'] = df.groupby('category')['price'].transform(lambda x: x.fillna(x.median()))

# 处理缺失的用户地区 - 使用众数填充
most_common_region = df['user_region'].mode()[0]
df['user_region'] = df['user_region'].fillna(most_common_region)

# 处理缺失的产品类别 - 从product_id推断
# 假设我们有一个产品映射字典
product_category_map = {...}  # 实际项目中需要构建这个映射
df['category'] = df.apply(lambda row: product_category_map.get(row['product_id'], row['category']) 
                         if pd.isnull(row['category']) else row['category'], axis=1)

# 删除缺失关键信息的记录
df = df.dropna(subset=['user_id', 'session_id', 'timestamp', 'event_type'])

# 检查处理后的缺失值情况
print(df.isnull().sum())
```

### 步骤3：检测和处理异常值

接下来，我们需要检测和处理数据集中的异常值。

```python
# 检查价格的分布
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['price'])
plt.title('价格分布')
plt.show()

# 使用IQR方法检测价格异常值
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
price_outliers = ((df['price'] < lower_bound) | (df['price'] > upper_bound))
print(f"价格异常值数量: {price_outliers.sum()}")

# 处理价格异常值 - 将极端值限制在合理范围内
df['price_cleaned'] = df['price'].clip(lower_bound, upper_bound)

# 检查时间戳的有效性
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
invalid_timestamps = df['timestamp'].isnull()
print(f"无效时间戳数量: {invalid_timestamps.sum()}")

# 删除无效时间戳的记录
df = df.dropna(subset=['timestamp'])

# 检查事件类型的有效性
valid_event_types = ['view', 'cart', 'purchase']
invalid_events = ~df['event_type'].isin(valid_event_types)
print(f"无效事件类型数量: {invalid_events.sum()}")

# 修正事件类型 - 假设我们有一些规则来修正
df.loc[df['event_type'] == 'add_to_cart', 'event_type'] = 'cart'
df.loc[df['event_type'] == 'buy', 'event_type'] = 'purchase'
```

### 步骤4：数据标准化和格式转换

为了便于后续分析，我们需要标准化数据格式。

```python
# 标准化时间戳格式
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# 从URL中提取页面类型
def extract_page_type(url):
    if 'product' in url:
        return 'product_page'
    elif 'category' in url:
        return 'category_page'
    elif 'search' in url:
        return 'search_page'
    elif 'cart' in url:
        return 'cart_page'
    elif 'checkout' in url:
        return 'checkout_page'
    else:
        return 'other'

df['page_type'] = df['page_url'].apply(extract_page_type)

# 标准化产品类别
df['category'] = df['category'].str.lower().str.strip()

# 从user_agent提取设备类型
def extract_device_type(user_agent):
    if pd.isnull(user_agent):
        return 'unknown'
    user_agent = user_agent.lower()
    if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    else:
        return 'desktop'

df['device_type'] = df['user_agent'].apply(extract_device_type)
```

### 步骤5：创建派生特征

为了增强数据的分析价值，我们创建一些派生特征。

```python
# 计算会话持续时间
session_start = df.groupby('session_id')['timestamp'].min()
session_end = df.groupby('session_id')['timestamp'].max()
session_duration = (session_end - session_start).dt.total_seconds() / 60  # 转换为分钟

# 将会话持续时间添加到原始数据框
session_duration_df = pd.DataFrame({'session_duration': session_duration})
session_duration_df.reset_index(inplace=True)
df = pd.merge(df, session_duration_df, on='session_id', how='left')

# 计算用户在每个会话中的活动数量
activity_count = df.groupby(['session_id', 'event_type']).size().unstack(fill_value=0)
activity_count.columns = [f'{col}_count' for col in activity_count.columns]
activity_count.reset_index(inplace=True)
df = pd.merge(df, activity_count, on='session_id', how='left')

# 创建购买标志
df['has_purchase'] = df['session_id'].isin(df[df['event_type'] == 'purchase']['session_id'])

# 计算价格区间
price_bins = [0, 10, 50, 100, 500, float('inf')]
price_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
df['price_range'] = pd.cut(df['price_cleaned'], bins=price_bins, labels=price_labels)
```

### 步骤6：数据验证与导出

最后，我们验证清洗后的数据并导出用于后续分析。

```python
# 检查数据质量
print("清洗后的数据形状:", df.shape)
print("缺失值情况:")
print(df.isnull().sum())

# 检查数据一致性
print("事件类型分布:")
print(df['event_type'].value_counts())
print("设备类型分布:")
print(df['device_type'].value_counts())

# 导出清洗后的数据
df.to_csv('ecommerce_data_cleaned.csv', index=False)

# 创建汇总数据用于分析
session_summary = df.groupby('session_id').agg({
    'user_id': 'first',
    'date': 'first',
    'device_type': 'first',
    'view_count': 'first',
    'cart_count': 'first',
    'purchase_count': 'first',
    'session_duration': 'first',
    'has_purchase': 'first'
})

session_summary.to_csv('session_summary.csv', index=True)
```

## 结果分析

通过数据清洗和预处理，我们解决了以下数据质量问题：

1. **缺失值处理**：填充了缺失的价格、用户地区和产品类别，删除了缺失关键信息的记录
2. **异常值处理**：识别并处理了价格异常值，修正了无效的时间戳和事件类型
3. **数据标准化**：统一了时间格式，标准化了产品类别，提取了页面类型和设备类型
4. **特征创建**：创建了会话持续时间、活动计数、购买标志和价格区间等派生特征

清洗后的数据集更加完整、一致和有结构，为后续的用户行为分析提供了坚实的基础。

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **高级缺失值填充**：使用机器学习模型预测缺失值
2. **用户行为序列分析**：分析用户在一个会话中的行为序列和转化路径
3. **异常检测算法**：应用隔离森林等算法自动检测多维异常
4. **特征重要性分析**：评估不同特征对购买行为的预测能力
5. **数据可视化**：创建交互式仪表板展示清洗后的数据洞察

## 小结与反思

通过这个项目，我们学习了如何处理电商用户数据中常见的数据质量问题，并为后续分析做好准备。数据预处理是数据分析和挖掘的基础，良好的数据质量对于得出可靠的结论至关重要。

在实际应用中，这类数据清洗工作可以帮助电商平台更好地理解用户行为，优化用户体验，提高转化率和销售额。例如，通过分析不同设备类型的转化率差异，可以针对性地优化移动端或桌面端的用户界面。

### 思考问题

1. 在处理缺失值时，我们应该考虑哪些因素来选择合适的填充策略？
2. 异常值是否总是应该被移除或修正？在什么情况下异常值可能包含有价值的信息？
3. 如何评估数据清洗和预处理的效果？有哪些指标可以衡量数据质量的提升？

<div class="practice-link">
  <a href="/projects/preprocessing/medical-missing-values.html" class="button">下一个项目：医疗数据缺失值处理</a>
</div> 