# 回归实践案例

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>通过实际案例学习回归分析的完整流程</li>
      <li>掌握数据预处理、特征工程和模型选择的实践技巧</li>
      <li>学习如何解释回归模型结果并提出业务建议</li>
      <li>了解不同领域回归分析的应用方法</li>
    </ul>
  </div>
</div>

## 案例一：房价预测

房价预测是回归分析的经典应用场景，我们将使用波士顿房价数据集进行演示。

### 问题描述

预测波士顿地区的房屋价格，帮助购房者和房地产开发商做出决策。

### 数据探索

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

# 加载数据
boston = load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names

# 创建DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['PRICE'] = y

# 查看数据基本信息
print(df.info())
print("\n统计摘要:")
print(df.describe())

# 查看目标变量分布
plt.figure(figsize=(10, 6))
plt.hist(df['PRICE'], bins=30)
plt.xlabel('价格（千美元）')
plt.ylabel('频数')
plt.title('波士顿房价分布')
plt.show()

# 相关性分析
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性矩阵')
plt.show()

# 散点图矩阵
sns.pairplot(df[['PRICE', 'RM', 'LSTAT', 'DIS', 'NOX']])
plt.suptitle('主要特征与房价的关系', y=1.02)
plt.show()
```

  </div>
</div>

### 数据预处理

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建预处理管道
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# 应用预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

  </div>
</div>

### 模型训练与评估

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 创建模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# 训练和评估
results = {}
for name, model in models.items():
    # 训练模型
    model.fit(X_train_processed, y_train)
    
    # 预测
    y_pred = model.predict(X_test_processed)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 存储结果
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

# 显示结果
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R²: {metrics['R²']:.4f}")

# 可视化比较
plt.figure(figsize=(12, 6))
names = list(results.keys())
mse_values = [results[name]['MSE'] for name in names]
r2_values = [results[name]['R²'] for name in names]

x = np.arange(len(names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

rects1 = ax1.bar(x - width/2, mse_values, width, label='MSE', color='red', alpha=0.7)
rects2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='blue', alpha=0.7)

ax1.set_xlabel('模型')
ax1.set_ylabel('MSE', color='red')
ax2.set_ylabel('R²', color='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.tick_params(axis='y', labelcolor='red')
ax2.tick_params(axis='y', labelcolor='blue')

fig.tight_layout()
plt.title('不同模型性能比较')
plt.show()
```

  </div>
</div>

### 特征重要性分析

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 使用随机森林的特征重要性
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('特征重要性')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# 前5个最重要的特征
print("最重要的5个特征:")
for i in range(5):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

  </div>
</div>

### 模型优化

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import GridSearchCV

# 选择表现最好的模型进行优化（假设是梯度提升）
best_model = GradientBoostingRegressor(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# 创建网格搜索
grid_search = GridSearchCV(
    best_model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 训练
grid_search.fit(X_train_processed, y_train)

# 最佳参数
print("最佳参数:")
print(grid_search.best_params_)

# 使用最佳模型预测
best_gbr = grid_search.best_estimator_
y_pred = best_gbr.predict(X_test_processed)

# 评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"优化后的模型性能:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
```

  </div>
</div>

### 结果可视化

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 预测值与实际值比较
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('预测值与实际值比较')
plt.grid(True)
plt.show()

# 残差分析
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.plot([y_pred.min(), y_pred.max()], [0, 0], 'k--', lw=2)
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.title('残差分析')
plt.grid(True)
plt.show()

# 残差分布
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差分布')
plt.grid(True)
plt.show()
```

  </div>
</div>

### 业务建议

基于模型结果，我们可以提出以下业务建议：

1. **关注重要特征**：根据特征重要性，房屋的平均房间数(RM)和低收入人口比例(LSTAT)对房价影响最大，开发商应优先考虑这些因素
2. **定价策略**：使用模型预测合理的房价区间，避免定价过高或过低
3. **投资决策**：根据模型预测，识别潜在的低估或高估房产
4. **市场细分**：根据不同特征组合，将市场细分为不同价格区间的子市场

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>波士顿房价数据集是由卡内基梅隆大学的Harrison和Rubinfeld于1978年收集的，包含了波士顿郊区506个人口普查区的房价中位数和13个潜在影响因素。尽管这个数据集已有几十年历史，但它仍然是回归分析教学和研究的经典数据集。</p>
  </div>
</div>

## 案例二：销售预测

销售预测是企业决策的重要依据，我们将使用零售销售数据进行演示。

### 问题描述

预测未来几个月的产品销售量，帮助企业优化库存管理和营销策略。

### 数据探索

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 加载数据（假设我们有一个包含日期和销售量的CSV文件）
# sales_data = pd.read_csv('sales_data.csv')
# 为了演示，我们生成一些模拟数据
np.random.seed(42)
date_rng = pd.date_range(start='2018-01-01', end='2020-12-31', freq='D')
sales_data = pd.DataFrame(date_rng, columns=['date'])
sales_data['sales'] = np.random.randint(100, 200, size=(len(date_rng))) + \
                     20 * np.sin(np.arange(len(date_rng)) * 2 * np.pi / 365) + \
                     np.random.normal(0, 10, size=len(date_rng))
sales_data['weekday'] = sales_data['date'].dt.dayofweek
sales_data['month'] = sales_data['date'].dt.month
sales_data['year'] = sales_data['date'].dt.year
sales_data['day'] = sales_data['date'].dt.day

# 查看数据
print(sales_data.head())
print("\n统计摘要:")
print(sales_data.describe())

# 可视化销售趋势
plt.figure(figsize=(15, 7))
plt.plot(sales_data['date'], sales_data['sales'])
plt.title('销售趋势')
plt.xlabel('日期')
plt.ylabel('销售量')
plt.grid(True)
plt.show()

# 按月份聚合
monthly_sales = sales_data.groupby(['year', 'month'])['sales'].sum().reset_index()
monthly_sales['year_month'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str)

plt.figure(figsize=(15, 7))
plt.bar(monthly_sales['year_month'], monthly_sales['sales'])
plt.title('月度销售量')
plt.xlabel('年-月')
plt.ylabel('销售量')
plt.xticks(rotation=90)
plt.grid(True, axis='y')
plt.show()

# 季节性分解
# 将数据转换为时间序列格式
ts_data = sales_data.set_index('date')['sales']
decomposition = seasonal_decompose(ts_data, model='additive', period=365)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
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

# 按星期几分析
weekday_sales = sales_data.groupby('weekday')['sales'].mean().reset_index()
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekday_sales['weekday_name'] = weekday_sales['weekday'].apply(lambda x: weekday_names[x])

plt.figure(figsize=(10, 6))
plt.bar(weekday_sales['weekday_name'], weekday_sales['sales'])
plt.title('不同星期几的平均销售量')
plt.xlabel('星期几')
plt.ylabel('平均销售量')
plt.grid(True, axis='y')
plt.show()
```

  </div>
</div>

### 特征工程

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 创建滞后特征
for i in range(1, 8):
    sales_data[f'sales_lag_{i}'] = sales_data['sales'].shift(i)

# 创建滚动平均特征
sales_data['sales_ma_7'] = sales_data['sales'].rolling(window=7).mean()
sales_data['sales_ma_30'] = sales_data['sales'].rolling(window=30).mean()

# 创建季节性特征
sales_data['is_weekend'] = sales_data['weekday'].apply(lambda x: 1 if x >= 5 else 0)
sales_data['is_month_start'] = sales_data['day'].apply(lambda x: 1 if x <= 5 else 0)
sales_data['is_month_end'] = sales_data['day'].apply(lambda x: 1 if x >= 25 else 0)

# 去除NaN值
sales_data = sales_data.dropna()

# 准备特征和目标变量
features = ['weekday', 'month', 'is_weekend', 'is_month_start', 'is_month_end',
            'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4', 
            'sales_lag_5', 'sales_lag_6', 'sales_lag_7',
            'sales_ma_7', 'sales_ma_30']

X = sales_data[features]
y = sales_data['sales']

# 划分训练集和测试集（按时间顺序）
train_size = int(len(sales_data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```

  </div>
</div>

### 模型训练与评估

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 创建模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# 训练和评估
results = {}
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 存储结果
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# 显示结果
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R²']:.4f}")
```

  </div>
</div>

### 预测可视化

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 使用表现最好的模型进行预测（假设是梯度提升）
best_model = models['Gradient Boosting']
y_pred = best_model.predict(X_test)

# 创建包含日期、实际值和预测值的DataFrame
pred_df = pd.DataFrame({
    'date': sales_data['date'].iloc[train_size:].reset_index(drop=True),
    'actual': y_test.reset_index(drop=True),
    'predicted': y_pred
})

# 可视化预测结果
plt.figure(figsize=(15, 7))
plt.plot(pred_df['date'], pred_df['actual'], label='实际销售量')
plt.plot(pred_df['date'], pred_df['predicted'], label='预测销售量', alpha=0.7)
plt.title('销售量预测')
plt.xlabel('日期')
plt.ylabel('销售量')
plt.legend()
plt.grid(True)
plt.show()

# 预测误差分析
pred_df['error'] = pred_df['actual'] - pred_df['predicted']
plt.figure(figsize=(15, 7))
plt.plot(pred_df['date'], pred_df['error'])
plt.axhline(y=0, color='r', linestyle='-')
plt.title('预测误差')
plt.xlabel('日期')
plt.ylabel('误差')
plt.grid(True)
plt.show()

# 误差分布
plt.figure(figsize=(10, 6))
plt.hist(pred_df['error'], bins=30)
plt.title('误差分布')
plt.xlabel('误差')
plt.ylabel('频数')
plt.grid(True)
plt.show()
```

  </div>
</div>

### 特征重要性分析

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 使用梯度提升模型的特征重要性
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('特征重要性')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# 前5个最重要的特征
print("最重要的5个特征:")
for i in range(5):
    print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")
```

  </div>
</div>

### 业务建议

基于销售预测模型，我们可以提出以下业务建议：

1. **库存管理**：根据预测的销售量调整库存水平，减少库存成本和缺货风险
2. **促销活动**：在预测销售低谷期策划促销活动，提高销售量
3. **人力资源规划**：根据预测的销售高峰期合理安排人员配置
4. **供应链优化**：提前通知供应商预期的需求变化，优化供应链
5. **季节性策略**：针对发现的季节性模式制定相应的营销和产品策略

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>忽略外部因素</strong>：销售预测应考虑市场趋势、竞争对手活动等外部因素</li>
      <li><strong>过度依赖历史数据</strong>：市场环境变化可能导致历史模式不再适用</li>
      <li><strong>忽略预测区间</strong>：提供预测区间比单点预测更有价值</li>
      <li><strong>缺乏持续更新</strong>：销售预测模型应定期更新以适应新数据</li>
    </ul>
  </div>
</div>

## 小结与思考

通过实际案例，我们学习了回归分析在房价预测和销售预测中的应用。这些案例展示了从数据探索到模型部署的完整流程。

### 关键要点回顾

- 数据探索和可视化是理解数据特性的重要步骤
- 特征工程对模型性能有显著影响
- 不同回归模型适用于不同类型的问题
- 模型评估应使用多种指标
- 模型结果的解释和业务建议是数据分析的最终目标

### 思考问题

1. 如何将领域知识融入回归模型的构建过程？
2. 在实际业务场景中，如何平衡模型复杂度和可解释性？
3. 如何处理回归分析中的数据质量问题？

<BackToPath />

<div class="practice-link">
  <a href="/projects/regression.html" class="button">前往实践项目</a>
</div> 