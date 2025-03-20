# 销售额预测

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：中级</li>
      <li><strong>类型</strong>：回归 - 中级</li>
      <!-- <li><strong>预计时间</strong>：4-6小时</li> -->
      <li><strong>技能点</strong>：时间序列分析、特征工程、回归模型</li>
      <li><strong>对应知识模块</strong>：<a href="/core/regression/linear-regression.html">回归分析</a></li>
    </ul>
  </div>
</div>

## 项目背景

销售额预测是企业运营中的关键任务，准确的预测可以帮助企业优化库存管理、人力资源规划和营销策略。通过分析历史销售数据和相关因素（如促销活动、季节性、价格变动等），我们可以构建模型来预测未来的销售表现。

在这个项目中，我们将使用零售商的销售数据，构建一个销售额预测模型，帮助企业做出更明智的决策。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>沃尔玛等大型零售商每天处理数百万笔交易，他们使用先进的预测模型来优化库存和定价。据估计，准确的销售预测每年可以为大型零售商节省数百万美元的库存成本，并显著减少产品浪费。</p>
  </div>
</div>

## 数据集介绍

我们将使用一个零售销售数据集，包含某连锁超市两年内的每日销售记录。数据集包含以下特征：

- Date：销售日期
- Store：商店ID
- Item：商品ID
- Sales：销售数量
- Price：商品价格
- Promotion：是否有促销活动（1表示有，0表示无）
- Holiday：是否是假日（1表示是，0表示否）
- Temperature：当天温度
- Fuel_Price：燃油价格
- CPI：消费者价格指数
- Unemployment：失业率

目标变量是Sales（销售数量），我们的任务是预测未来一段时间内的销售量。

## 项目目标

1. 构建一个能够准确预测销售额的回归模型
2. 识别影响销售的关键因素
3. 分析季节性和趋势对销售的影响
4. 评估不同预测方法的性能
5. 提供销售预测的可视化和业务建议

## 实施步骤

### 1. 数据探索与可视化

首先，我们需要了解数据的基本特征和分布：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# 加载数据
df = pd.read_csv('retail_sales.csv')

# 将日期列转换为日期类型
df['Date'] = pd.to_datetime(df['Date'])

# 查看数据基本信息
print(df.info())
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 可视化销售趋势
plt.figure(figsize=(12, 6))
sales_by_date = df.groupby('Date')['Sales'].sum().reset_index()
plt.plot(sales_by_date['Date'], sales_by_date['Sales'])
plt.title('每日总销售量趋势')
plt.xlabel('日期')
plt.ylabel('销售量')
plt.grid(True)
plt.show()

# 查看销售的季节性模式
plt.figure(figsize=(12, 6))
sales_by_month = df.groupby(df['Date'].dt.month)['Sales'].mean().reset_index()
plt.bar(sales_by_month['Date'], sales_by_month['Sales'])
plt.title('月度平均销售量')
plt.xlabel('月份')
plt.ylabel('平均销售量')
plt.xticks(range(1, 13))
plt.grid(True, axis='y')
plt.show()

# 查看促销活动对销售的影响
plt.figure(figsize=(10, 6))
sns.boxplot(x='Promotion', y='Sales', data=df)
plt.title('促销活动对销售的影响')
plt.xlabel('是否有促销')
plt.ylabel('销售量')
plt.show()

# 查看温度对销售的影响
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature'], df['Sales'], alpha=0.5)
plt.title('温度与销售量的关系')
plt.xlabel('温度')
plt.ylabel('销售量')
plt.grid(True)
plt.show()
```

### 2. 特征工程

创建有助于预测的新特征：

```python
# 从日期提取时间特征
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# 创建季节特征
df['Season'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                          2 if x in [3, 4, 5] else 
                                          3 if x in [6, 7, 8] else 4)

# 创建是否周末特征
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# 创建滞后特征（前一天、前一周的销售量）
df_lag = df.copy()
df_lag = df_lag.sort_values(['Store', 'Item', 'Date'])
df_lag['Sales_Lag1'] = df_lag.groupby(['Store', 'Item'])['Sales'].shift(1)
df_lag['Sales_Lag7'] = df_lag.groupby(['Store', 'Item'])['Sales'].shift(7)

# 移除包含NaN的行（由于创建滞后特征）
df_lag = df_lag.dropna()

# 查看新特征
print(df_lag.head())
```

### 3. 时间序列分解

分析销售数据的趋势、季节性和残差成分：

```python
# 选择一个特定商店和商品的销售数据进行时间序列分解
store_item_sales = df[(df['Store'] == 1) & (df['Item'] == 1)].set_index('Date')['Sales']

# 确保索引是等间隔的
store_item_sales = store_item_sales.asfreq('D')

# 时间序列分解
decomposition = seasonal_decompose(store_item_sales, model='multiplicative', period=7)

# 可视化分解结果
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=ax1)
ax1.set_title('观测值')
decomposition.trend.plot(ax=ax2)
ax2.set_title('趋势')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('季节性')
decomposition.resid.plot(ax=ax4)
ax4.set_title('残差')
plt.tight_layout()
plt.show()
```

### 4. 模型训练与评估

我们将尝试多种回归模型，并比较它们的性能：

```python
# 准备特征和目标变量
X = df_lag.drop(['Date', 'Sales'], axis=1)
y = df_lag['Sales']

# 将分类变量转换为哑变量
X = pd.get_dummies(X, columns=['Store', 'Item', 'Season'], drop_first=True)

# 划分训练集和测试集（按时间顺序）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"{model_name}:")
    print(f"训练集RMSE: {train_rmse:.2f}")
    print(f"测试集RMSE: {test_rmse:.2f}")
    print(f"训练集R²: {train_r2:.2f}")
    print(f"测试集R²: {test_r2:.2f}")
    print(f"测试集MAE: {test_mae:.2f}")
    
    return model, y_test_pred

# 线性回归
lr_model, lr_pred = evaluate_model(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test, "线性回归模型")
print("\n")

# 随机森林回归
rf_model, rf_pred = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
                                  X_train, X_test, y_train, y_test, "随机森林回归模型")
```

### 5. 特征重要性分析

了解哪些因素对销售影响最大：

```python
# 使用随机森林模型分析特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# 显示前15个最重要的特征
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('特征重要性（前15名）')
plt.tight_layout()
plt.show()
```

### 6. 预测可视化

可视化模型预测结果与实际值的对比：

```python
# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='实际销售量')
plt.plot(rf_pred, label='预测销售量', alpha=0.7)
plt.title('随机森林模型：实际销售量 vs 预测销售量')
plt.xlabel('样本索引')
plt.ylabel('销售量')
plt.legend()
plt.grid(True)
plt.show()

# 预测误差分析
errors = y_test - rf_pred
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50)
plt.title('预测误差分布')
plt.xlabel('预测误差')
plt.ylabel('频率')
plt.grid(True)
plt.show()
```

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **高级时间序列模型**：尝试使用ARIMA、SARIMA或Prophet等专门的时间序列模型
2. **多步预测**：构建能够预测未来多个时间点销售量的模型
3. **集成预测**：结合多个模型的预测结果，提高预测准确性
4. **特征选择**：使用特征选择技术，找出最有预测力的特征子集
5. **交叉验证**：实现时间序列交叉验证，获得更稳健的模型评估

## 小结与反思

通过这个项目，我们学习了如何构建销售额预测模型，从数据探索到模型评估的完整流程。我们发现促销活动、季节性和历史销售数据对预测未来销售有显著影响。

在实际应用中，销售预测模型可以帮助企业优化库存管理，减少库存成本和产品浪费，同时确保产品供应充足，满足客户需求。此外，准确的销售预测还可以帮助企业制定更有效的营销策略和促销计划。

### 思考问题

1. 除了我们使用的特征外，还有哪些因素可能影响销售？如何获取这些数据？
2. 销售预测模型在哪些情况下可能失效？如何应对这些挑战？
3. 如何将销售预测结果转化为具体的业务决策和行动？

<div class="practice-link">
  <a href="/projects/regression/anomaly-detection.html" class="button">下一个项目：异常检测与预测</a>
</div> 