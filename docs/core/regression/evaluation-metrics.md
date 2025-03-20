# 回归评估指标

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解常用回归评估指标的计算方法和意义</li>
      <li>掌握不同评估指标的适用场景和局限性</li>
      <li>学习如何使用交叉验证评估回归模型</li>
      <li>了解如何选择合适的评估指标进行模型比较</li>
    </ul>
  </div>
</div>

## 常用回归评估指标

评估回归模型性能的指标有多种，每种指标都有其特定的用途和解释。

### 均方误差(MSE)

均方误差是最常用的回归评估指标之一，计算预测值与实际值差异的平方的平均值：

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

其中：
- $y_i$ 是实际值
- $\hat{y}_i$ 是预测值
- $n$ 是样本数量

MSE对较大的误差给予更高的惩罚，但单位是目标变量的平方。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.metrics import mean_squared_error

# 计算MSE
mse = mean_squared_error(y_true, y_pred)
print(f"均方误差(MSE): {mse:.4f}")
```

  </div>
</div>

### 均方根误差(RMSE)

均方根误差是MSE的平方根，使得单位与目标变量相同，更易于解释：

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# 计算RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"均方根误差(RMSE): {rmse:.4f}")
```

  </div>
</div>

### 平均绝对误差(MAE)

平均绝对误差计算预测值与实际值差异的绝对值的平均值：

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

MAE对异常值不如MSE敏感，且易于解释。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.metrics import mean_absolute_error

# 计算MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"平均绝对误差(MAE): {mae:.4f}")
```

  </div>
</div>

### 决定系数(R²)

决定系数衡量模型解释的目标变量方差比例，范围通常在0到1之间（可以为负）：

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

其中：
- $\bar{y}$ 是实际值的平均值

R²值为1表示完美拟合，0表示模型不比简单地预测平均值好。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.metrics import r2_score

# 计算R²
r2 = r2_score(y_true, y_pred)
print(f"决定系数(R²): {r2:.4f}")
```

  </div>
</div>

### 调整R²

调整R²考虑了特征数量，对添加不相关特征进行惩罚：

$$\text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

其中：
- $n$ 是样本数量
- $p$ 是特征数量

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

# 计算调整R²
adj_r2 = adjusted_r2_score(y_true, y_pred, X.shape[1])
print(f"调整决定系数(Adjusted R²): {adj_r2:.4f}")
```

  </div>
</div>

### 平均绝对百分比误差(MAPE)

平均绝对百分比误差计算预测值与实际值差异的百分比的平均值：

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

MAPE提供了相对误差的度量，但在实际值接近零时可能出现问题。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 计算MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"平均绝对百分比误差(MAPE): {mape:.4f}%")
```

  </div>
</div>

### 中位数绝对误差(MedAE)

中位数绝对误差计算预测值与实际值差异的绝对值的中位数：

$$MedAE = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, ..., |y_n - \hat{y}_n|)$$

MedAE对异常值更加鲁棒。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.metrics import median_absolute_error

# 计算MedAE
medae = median_absolute_error(y_true, y_pred)
print(f"中位数绝对误差(MedAE): {medae:.4f}")
```

  </div>
</div>

## 评估指标的选择

不同的评估指标适用于不同的场景：

1. **MSE/RMSE**：当较大误差需要更高惩罚时使用，对异常值敏感
2. **MAE**：当所有误差应被平等对待时使用，对异常值较不敏感
3. **R²**：当需要比较不同尺度的目标变量时使用
4. **调整R²**：当比较具有不同特征数量的模型时使用
5. **MAPE**：当相对误差更重要时使用，但目标变量不应接近零
6. **MedAE**：当数据中存在异常值时使用


## 使用交叉验证评估模型

交叉验证通过在不同的数据子集上训练和评估模型，提供更可靠的性能估计。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# 创建模型
model = LinearRegression()

# 创建K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 计算交叉验证分数
mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# 显示结果
print("MSE交叉验证分数:")
for i, mse in enumerate(mse_scores):
    print(f"折{i+1}: {mse:.4f}")
print(f"平均MSE: {mse_scores.mean():.4f}")
print(f"标准差: {mse_scores.std():.4f}")

print("\nR²交叉验证分数:")
for i, r2 in enumerate(r2_scores):
    print(f"折{i+1}: {r2:.4f}")
print(f"平均R²: {r2_scores.mean():.4f}")
print(f"标准差: {r2_scores.std():.4f}")
```

  </div>
</div>

### 学习曲线

学习曲线显示模型性能如何随训练集大小变化，有助于诊断过拟合或欠拟合。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# 计算学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# 计算平均值和标准差
train_mean = -train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = -test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.grid()
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="训练集MSE")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="验证集MSE")
plt.xlabel("训练样本数")
plt.ylabel("MSE")
plt.title("学习曲线")
plt.legend(loc="best")
plt.show()
```

  </div>
</div>

## 比较多个模型

在实际应用中，通常需要比较多个回归模型的性能。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

# 创建模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'Neural Network': MLPRegressor(random_state=42)
}

# 评估指标
metrics = {
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'R²': 'r2'
}

# 计算交叉验证分数
results = {}
for name, model in models.items():
    model_results = {}
    for metric_name, metric in metrics.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=metric)
        if metric.startswith('neg_'):
            scores = -scores
        model_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    results[name] = model_results

# 显示结果
for model_name, model_results in results.items():
    print(f"\n{model_name}:")
    for metric_name, values in model_results.items():
        print(f"  {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")

# 可视化比较
plt.figure(figsize=(12, 8))
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.25

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# MSE
mse_means = [results[name]['MSE']['mean'] for name in model_names]
mse_stds = [results[name]['MSE']['std'] for name in model_names]
ax1.bar(x, mse_means, width, yerr=mse_stds, label='MSE', color='red', alpha=0.7)
ax1.set_ylabel('MSE')
ax1.set_title('均方误差(MSE)比较')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right')

# MAE
mae_means = [results[name]['MAE']['mean'] for name in model_names]
mae_stds = [results[name]['MAE']['std'] for name in model_names]
ax2.bar(x, mae_means, width, yerr=mae_stds, label='MAE', color='blue', alpha=0.7)
ax2.set_ylabel('MAE')
ax2.set_title('平均绝对误差(MAE)比较')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right')

# R²
r2_means = [results[name]['R²']['mean'] for name in model_names]
r2_stds = [results[name]['R²']['std'] for name in model_names]
ax3.bar(x, r2_means, width, yerr=r2_stds, label='R²', color='green', alpha=0.7)
ax3.set_ylabel('R²')
ax3.set_title('决定系数(R²)比较')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, rotation=45, ha='right')

fig.tight_layout()
plt.show()
```

  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>仅依赖单一指标</strong>：不同指标反映模型性能的不同方面</li>
      <li><strong>忽略交叉验证</strong>：单次训练/测试分割可能导致高方差估计</li>
      <li><strong>过度解读R²</strong>：高R²不一定意味着模型预测能力强</li>
      <li><strong>忽略领域知识</strong>：选择指标时应考虑业务需求</li>
    </ul>
  </div>
</div>

## 小结与思考

回归评估指标是选择和优化回归模型的重要工具，不同指标适用于不同场景。

### 关键要点回顾

- MSE和RMSE对较大误差给予更高惩罚，适用于异常值影响较大的场景
- MAE和MedAE对异常值较不敏感，提供更稳健的评估
- R²和调整R²衡量模型解释的方差比例，便于比较不同尺度的问题
- 交叉验证提供更可靠的性能估计，减少过拟合风险
- 学习曲线帮助诊断模型是否存在过拟合或欠拟合

### 思考问题

1. 在什么情况下应该选择MSE而非MAE作为评估指标？
2. 为什么R²可能为负值，这意味着什么？
3. 如何根据业务需求选择最合适的评估指标？

<div class="practice-link">
  <a href="/projects/regression.html" class="button">前往实践项目</a>
</div> 