# 线性回归

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解线性回归的基本原理和假设</li>
      <li>掌握简单线性回归和多元线性回归的区别</li>
      <li>学习如何评估线性回归模型的性能</li>
      <li>了解正则化技术如何改进线性回归</li>
    </ul>
  </div>
</div>

## 线性回归概述

线性回归是最基础、应用最广泛的回归分析方法，用于建立因变量（目标）与一个或多个自变量（特征）之间的关系模型。线性回归假设特征和目标之间存在线性关系。

### 简单线性回归

简单线性回归只涉及一个自变量和一个因变量，其数学表达式为：

$$y = w_0 + w_1x + \varepsilon$$

其中：
- $y$ 是因变量（预测目标）
- $x$ 是自变量（特征）
- $w_0$ 是截距（偏置项）
- $w_1$ 是斜率（权重）
- $\varepsilon$ 是误差项

简单线性回归的目标是找到最佳的 $w_0$ 和 $w_1$ 值，使得预测值与实际值之间的误差最小。

### 多元线性回归

多元线性回归涉及多个自变量，其数学表达式为：

$$y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n + \varepsilon$$

或者用矩阵形式表示：

$$y = \mathbf{X}\mathbf{w} + \varepsilon$$

其中：
- $y$ 是因变量
- $x_1, x_2, ..., x_n$ 是自变量
- $w_0, w_1, w_2, ..., w_n$ 是模型参数
- $\varepsilon$ 是误差项
- $\mathbf{X}$ 是特征矩阵
- $\mathbf{w}$ 是参数向量

## 线性回归的假设

线性回归模型基于以下假设：

1. **线性关系**：自变量和因变量之间存在线性关系
2. **独立性**：观测值之间相互独立
3. **同方差性**：误差项具有恒定的方差
4. **正态性**：误差项服从正态分布
5. **无多重共线性**：自变量之间不存在完全线性相关

当这些假设被满足时，线性回归模型能够提供无偏且有效的参数估计。

## 参数估计方法

### 最小二乘法

最小二乘法是最常用的参数估计方法，其目标是最小化残差平方和（RSS）：

$$RSS = (y - X\beta)^T(y - X\beta)$$

对于简单线性回归，最小二乘估计的解析解为：

$$w_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

$$w_0 = \bar{y} - w_1\bar{x}$$

对于多元线性回归，参数的矩阵形式解为：

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### 梯度下降法

当数据量很大时，计算 $(\mathbf{X}^T\mathbf{X})^{-1}$ 可能计算量过大，此时可以使用梯度下降法迭代求解：

1. 初始化参数 $\mathbf{w}$
2. 计算损失函数对参数的梯度
3. 沿着梯度的反方向更新参数
4. 重复步骤2和3直到收敛

更新规则为：

$$w_j := w_j - \alpha \frac{\partial}{\partial w_j} RSS$$

其中 $\alpha$ 是学习率。

<div class="code-example">
  <div class="code-example__title">代码示例：简单线性回归</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 使用sklearn的LinearRegression
model = LinearRegression()
model.fit(X, y)

# 获取参数
w0 = model.intercept_[0]
w1 = model.coef_[0][0]
print(f"截距 (w0): {w0:.4f}")
print(f"斜率 (w1): {w1:.4f}")

# 预测
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='数据点')
plt.plot(X_new, y_pred, 'r-', linewidth=2, label=f'y = {w0:.2f} + {w1:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('简单线性回归')
plt.legend()
plt.grid(True)
plt.show()

# 评估模型
y_pred_all = model.predict(X)
mse = mean_squared_error(y, y_pred_all)
r2 = r2_score(y, y_pred_all)
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")
```

  </div>
</div>

<div class="code-example">
  <div class="code-example__title">代码示例：多元线性回归</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 加载数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 获取参数
w0 = model.intercept_
w = model.coef_
print(f"截距 (w0): {w0:.4f}")
print("特征权重 (w):")
for i, feature_name in enumerate(housing.feature_names):
    print(f"  {feature_name}: {w[i]:.4f}")

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.grid(True)
plt.show()
```

  </div>
</div>

## 正则化技术

当特征数量较多或特征之间存在多重共线性时，标准线性回归可能会导致过拟合。正则化技术通过向损失函数添加惩罚项来减少过拟合风险。

### Ridge回归（L2正则化）

Ridge回归通过添加系数平方和的惩罚项来减小模型复杂度：

$$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha ||\mathbf{w}||_2^2$$

其中 $\alpha$ 是正则化强度参数。Ridge回归会减小所有系数的大小，但不会使系数变为零。

### Lasso回归（L1正则化）

Lasso回归使用系数绝对值和作为惩罚项：

$$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}i)^2 + \alpha ||\mathbf{w}||_1$$

Lasso回归的一个重要特性是它可以将一些系数精确地缩减为零，从而实现特征选择

### 弹性网络（Elastic Net）

弹性网络结合了Ridge和Lasso的惩罚项：

$$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha_1 ||\mathbf{w}||_1 + \alpha_2 ||\mathbf{w}||_2^2$$

弹性网络克服了Lasso在处理高度相关特征时的一些限制，同时保留了特征选择的能力。

<div class="code-example">
  <div class="code-example__title">代码示例：正则化线性回归</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# 加载数据
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建模型
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# 训练模型
ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)
elastic.fit(X_train_scaled, y_train)

# 预测
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_elastic = elastic.predict(X_test_scaled)

# 计算MSE
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)

print(f"Ridge MSE: {mse_ridge:.4f}")
print(f"Lasso MSE: {mse_lasso:.4f}")
print(f"ElasticNet MSE: {mse_elastic:.4f}")

# 可视化系数
plt.figure(figsize=(12, 6))
plt.plot(ridge.coef_, 's-', label='Ridge')
plt.plot(lasso.coef_, 'o-', label='Lasso')
plt.plot(elastic.coef_, '^-', label='ElasticNet')
plt.xlabel('特征索引')
plt.ylabel('系数值')
plt.title('不同正则化方法的系数比较')
plt.legend()
plt.grid(True)
plt.show()
```

  </div>
</div>

## 线性回归的优缺点

### 优点

- **简单易解释**：模型简单，参数具有明确的解释
- **计算效率高**：特别是对于小到中等规模的数据集
- **无需调整超参数**：标准线性回归没有需要调整的超参数
- **可作为基线模型**：为更复杂的模型提供比较基准

### 缺点

- **假设限制**：假设特征和目标之间存在线性关系
- **对异常值敏感**：异常值可能对模型参数产生显著影响
- **无法捕捉非线性关系**：当数据关系复杂时表现不佳
- **多重共线性问题**：当特征高度相关时，参数估计不稳定

## 实际应用案例

### 案例：预测房价

房价预测是线性回归的经典应用场景。以下是使用加州房价数据集的示例：

<div class="code-example">
  <div class="code-example__title">代码示例：加州房价预测</div>
  <div class="code-example__content">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# 数据探索
print("数据集形状:", X.shape)
print("特征名称:", housing.feature_names)
print("特征描述:")
print(X.describe())

# 相关性分析
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target
correlation = data.corr()
plt.figure(figsize=(12, 10))
plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation)), correlation.columns)
plt.title('特征相关性热图')
plt.show()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 获取系数
coefficients = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': model.coef_
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)

# 可视化系数
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel('系数值')
plt.title('特征系数')
plt.grid(True)
plt.show()

# 预测和评估
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"训练集 MSE: {train_mse:.4f}")
print(f"测试集 MSE: {test_mse:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('实际房价')
plt.ylabel('预测房价')
plt.title('实际房价 vs 预测房价')
plt.grid(True)
plt.show()

# 残差分析
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='r', linestyles='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.grid(True)
plt.show()
```

  </div>
</div>

## 总结

线性回归是数据挖掘和机器学习中最基础的算法之一，尽管简单，但在许多实际应用中表现良好。理解线性回归的原理和假设，掌握参数估计方法和正则化技术，对于构建有效的预测模型至关重要。

<div class="practice-link">
  <a href="/projects/prediction.html" class="button">前往实践项目</a>
</div> 