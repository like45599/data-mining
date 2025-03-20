# 非线性回归方法

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解非线性回归的基本概念和应用场景</li>
      <li>掌握多项式回归、决策树回归等常用非线性回归方法</li>
      <li>学习如何选择合适的非线性回归模型</li>
      <li>了解非线性回归中的过拟合问题及解决方案</li>
    </ul>
  </div>
</div>

## 非线性回归概述

非线性回归是一种用于建立自变量与因变量之间非线性关系的统计方法。当数据呈现明显的非线性特征时，线性回归模型往往无法准确捕捉数据的真实关系，此时需要使用非线性回归模型。

### 应用场景

非线性回归在以下场景中特别有用：

- **生物学生长曲线**：如细菌生长、种群动态等遵循指数或S形曲线
- **物理现象**：如放射性衰变、热力学过程等
- **经济学**：如消费者行为、市场饱和度分析
- **药物反应**：药物剂量与效果的非线性关系

### 与线性回归的区别

| 特性 | 线性回归 | 非线性回归 |
|------|----------|------------|
| 模型形式 | $y = w_0 + w_1x_1 + ... + w_nx_n$ | 可以是任何函数形式，如指数、对数、多项式等 |
| 参数估计 | 通常有解析解（最小二乘法） | 通常需要迭代优化算法 |
| 解释性 | 较强，系数直接表示变量影响 | 可能较弱，取决于具体模型 |
| 计算复杂度 | 较低 | 较高 |
| 过拟合风险 | 较低 | 较高 |

## 常用非线性回归方法

### 1. 多项式回归

多项式回归是线性回归的扩展，通过添加自变量的高次项来捕捉非线性关系。

**数学表达式**：$y = w_0 + w_1x + w_2x^2 + ... + w_nx^n$

**优点**：
- 实现简单，可以使用线性回归的框架
- 对于中等复杂度的非线性关系效果好

**缺点**：
- 高次多项式容易过拟合
- 对异常值敏感

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 生成非线性数据
np.random.seed(0)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(100) * 0.1

# 创建多项式回归模型
degrees = [1, 3, 5, 9]
plt.figure(figsize=(14, 5))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())
    
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression)
    ])
    pipeline.fit(X, y)
    
    # 预测
    X_test = np.linspace(0, 1, 100)[:, np.newaxis]
    plt.plot(X_test, pipeline.predict(X_test), label=f"度数 {degree}")
    plt.scatter(X, y, color='navy', s=30, marker='o', label="训练点")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.title(f"{degree}次多项式")

plt.tight_layout()
plt.show()
```

  </div>
</div>

### 2. 决策树回归

决策树回归通过将特征空间划分为多个区域，并在每个区域内使用常数值进行预测。

**优点**：
- 可以捕捉复杂的非线性关系
- 不需要对数据进行标准化
- 对异常值不敏感

**缺点**：
- 容易过拟合
- 不能外推到训练数据范围之外

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.randn(80) * 0.1

# 拟合决策树
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = regr.predict(X_test)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, color="darkorange", label="数据点")
plt.plot(X_test, y_pred, color="cornflowerblue", label="决策树预测", linewidth=2)
plt.xlabel("特征")
plt.ylabel("目标")
plt.title("决策树回归")
plt.legend()
plt.show()
```

  </div>
</div>

### 3. 支持向量回归(SVR)

SVR是支持向量机在回归问题上的应用，通过允许一定的误差，寻找最大间隔超平面。

**优点**：
- 对高维数据有效
- 通过核技巧可以处理复杂的非线性关系
- 对异常值较为鲁棒

**缺点**：
- 计算复杂度高
- 参数调优困难
- 不适合大规模数据集

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# 创建SVR模型
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=0.1, coef0=1)

# 拟合模型
svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=2,
                 label='{} 模型'.format(kernel_label[ix]))
    axes[ix].scatter(X, y, color='darkorange', label='数据点')
    axes[ix].set_xlabel('特征')
    axes[ix].set_ylabel('目标')
    axes[ix].set_title('{} 核'.format(kernel_label[ix]))
    axes[ix].legend()
fig.tight_layout()
plt.show()
```

  </div>
</div>

### 4. 随机森林回归

随机森林回归是一种集成学习方法，通过构建多个决策树并取平均值来提高预测性能和减少过拟合。

**优点**：
- 预测精度高
- 不易过拟合
- 可以处理高维数据
- 可以评估特征重要性

**缺点**：
- 计算复杂度较高
- 模型解释性较差
- 对极端值的预测可能不准确

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.randn(100) * 0.1

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = rf.predict(X_test)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='数据点')
plt.plot(X_test, y_pred, color='navy', label='随机森林预测')
plt.xlabel('特征')
plt.ylabel('目标')
plt.title('随机森林回归')
plt.legend()
plt.show()
```

  </div>
</div>

### 5. 神经网络回归

神经网络可以学习复杂的非线性关系，特别是在大数据集上表现优异。

**优点**：
- 可以建模极其复杂的非线性关系
- 适合大规模数据
- 可以自动学习特征表示

**缺点**：
- 需要大量数据
- 计算资源需求高
- 参数调优困难
- 解释性差

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.randn(100) * 0.1

# 创建神经网络回归模型
nn_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=2000,
    random_state=42
)

# 拟合模型
nn_reg.fit(X, y)

# 预测
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = nn_reg.predict(X_test)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='数据点')
plt.plot(X_test, y_pred, color='navy', label='神经网络预测')
plt.xlabel('特征')
plt.ylabel('目标')
plt.title('神经网络回归')
plt.legend()
plt.show()
```

  </div>
</div>

## 模型选择与过拟合处理

### 如何选择合适的非线性回归模型

选择合适的非线性回归模型需要考虑以下因素：

1. **数据特性**：了解数据的分布和潜在的非线性关系类型
2. **样本量**：复杂模型需要更多数据支持
3. **计算资源**：某些模型（如神经网络）需要更多计算资源
4. **解释性需求**：如果需要解释模型，多项式回归可能更合适
5. **预测精度要求**：通常复杂模型可以提供更高的预测精度

### 处理过拟合的方法

非线性回归模型容易过拟合，以下是一些常用的处理方法：

1. **正则化**：
   - L1正则化（Lasso）：促使部分特征系数为零
   - L2正则化（Ridge）：减小所有特征系数的大小

2. **交叉验证**：使用k折交叉验证选择最佳模型复杂度

3. **特征选择**：减少不相关特征，降低模型复杂度

4. **早停法**：在迭代算法中，当验证集性能开始下降时停止训练

5. **集成方法**：结合多个简单模型的预测结果

<div class="code-example">
  <div class="code-example__title">代码示例：正则化多项式回归</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# 生成数据
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1
X = X.reshape(-1, 1)

# 创建多项式回归模型，使用Ridge正则化
degrees = [1, 4, 15]
alphas = [0, 0.001, 1.0]

plt.figure(figsize=(14, 8))
for i, degree in enumerate(degrees):
    for j, alpha in enumerate(alphas):
        ax = plt.subplot(len(degrees), len(alphas), i * len(alphas) + j + 1)
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        model.fit(X, y)
        
        # 预测
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        plt.plot(X_test, model.predict(X_test), label=f"Model")
        plt.scatter(X, y, color='navy', s=30, marker='o', label="Training points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.title(f"degree={degree}, alpha={alpha}")
        
plt.tight_layout()
plt.show()
```

  </div>
</div>

## 实际应用案例

### 案例：预测房价

在房地产领域，房价与多个因素（如面积、位置、房龄等）之间通常存在非线性关系。以下是使用非线性回归预测房价的示例：

<div class="code-example">
  <div class="code-example__title">代码示例：房价预测</div>
  <div class="code-example__content">

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 加载加州房价数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建梯度提升回归模型
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

# 训练模型
gbr.fit(X_train_scaled, y_train)

# 预测
y_pred = gbr.predict(X_test_scaled)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 可视化特征重要性
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(housing.feature_names)[sorted_idx])
plt.title('特征重要性')
plt.tight_layout()
plt.show()

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.tight_layout()
plt.show()
```

  </div>
</div>

## 总结

非线性回归是处理复杂数据关系的强大工具，但选择合适的模型和避免过拟合是关键挑战。通过理解不同模型的特性、适当的正则化和交叉验证，可以构建既准确又稳健的非线性回归模型。

<div class="practice-link">
  <a href="/projects/prediction.html" class="button">前往实践项目</a>
</div>
