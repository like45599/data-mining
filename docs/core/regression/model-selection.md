# 回归模型选择

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解不同回归模型的适用场景和局限性</li>
      <li>掌握模型选择的方法和策略</li>
      <li>学习如何使用交叉验证和网格搜索优化模型</li>
      <li>了解模型集成和堆叠技术提高预测性能</li>
    </ul>
  </div>
</div>

## 回归模型比较

不同的回归模型有各自的优缺点和适用场景。以下是常见回归模型的比较：

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>优点</th>
        <th>缺点</th>
        <th>适用场景</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>线性回归</td>
        <td>
          - 简单易解释<br>
          - 训练速度快<br>
          - 低方差
        </td>
        <td>
          - 假设线性关系<br>
          - 对异常值敏感<br>
          - 无法捕捉非线性关系
        </td>
        <td>
          - 特征与目标变量呈线性关系<br>
          - 需要可解释性<br>
          - 数据量较小
        </td>
      </tr>
      <tr>
        <td>决策树回归</td>
        <td>
          - 可捕捉非线性关系<br>
          - 无需特征缩放<br>
          - 可处理分类特征
        </td>
        <td>
          - 容易过拟合<br>
          - 不稳定<br>
          - 预测不连续
        </td>
        <td>
          - 特征与目标变量呈非线性关系<br>
          - 存在特征交互<br>
          - 需要可解释性
        </td>
      </tr>
      <tr>
        <td>随机森林回归</td>
        <td>
          - 减少过拟合<br>
          - 提供特征重要性<br>
          - 鲁棒性强
        </td>
        <td>
          - 计算复杂度高<br>
          - 可解释性降低<br>
          - 超参数多
        </td>
        <td>
          - 特征与目标变量关系复杂<br>
          - 数据噪声较大<br>
          - 需要稳定性
        </td>
      </tr>
      <tr>
        <td>梯度提升回归</td>
        <td>
          - 高预测精度<br>
          - 可处理不同类型特征<br>
          - 提供特征重要性
        </td>
        <td>
          - 计算复杂度高<br>
          - 容易过拟合<br>
          - 参数调优复杂
        </td>
        <td>
          - 需要高预测精度<br>
          - 有足够计算资源<br>
          - 数据结构复杂
        </td>
      </tr>
      <tr>
        <td>支持向量回归</td>
        <td>
          - 处理非线性关系<br>
          - 对异常值鲁棒<br>
          - 泛化能力强
        </td>
        <td>
          - 计算复杂度高<br>
          - 参数敏感<br>
          - 不适合大数据集
        </td>
        <td>
          - 中小型数据集<br>
          - 存在异常值<br>
          - 需要高泛化能力
        </td>
      </tr>
      <tr>
        <td>神经网络回归</td>
        <td>
          - 处理复杂非线性关系<br>
          - 自动特征提取<br>
          - 适应大数据集
        </td>
        <td>
          - 计算资源需求高<br>
          - 可解释性差<br>
          - 需要大量数据
        </td>
        <td>
          - 复杂非线性关系<br>
          - 大型数据集<br>
          - 不需要高可解释性
        </td>
      </tr>
    </tbody>
  </table>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>没有一种回归模型在所有场景下都表现最佳，这就是所谓的"没有免费的午餐"定理。模型选择应该基于数据特性、问题需求和计算资源等因素综合考虑。</p>
  </div>
</div>

## 模型选择策略

选择合适的回归模型通常遵循以下步骤：

1. **理解问题**：明确预测目标和评估指标
2. **探索数据**：分析特征与目标变量的关系
3. **特征工程**：创建、转换和选择特征
4. **基准模型**：建立简单模型作为基准
5. **模型比较**：尝试不同类型的模型
6. **超参数调优**：优化选定模型的参数
7. **模型验证**：使用交叉验证评估模型性能
8. **模型集成**：组合多个模型提高性能

<div class="visualization-container">
  <div class="visualization-title">回归模型选择流程</div>
  <div class="visualization-content">
    <img src="/images/regression_model_selection.png" alt="回归模型选择流程">
  </div>
  <div class="visualization-caption">
    图1: 回归模型选择流程。从数据分析开始，经过模型评估和优化，最终选择最佳模型。
  </div>
</div>

## 使用交叉验证选择模型

交叉验证是比较不同模型性能的可靠方法。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 加载数据
# X, y = load_your_data()

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

# 创建管道
pipelines = {}
for name, model in models.items():
    if name in ['SVR', 'Ridge', 'Lasso', 'Neural Network']:
        pipelines[name] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        pipelines[name] = Pipeline([
            ('model', model)
        ])

# 设置交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 评估模型
results = {}
for name, pipeline in pipelines.items():
    # 计算负MSE（sklearn使用负分数表示需要最小化的指标）
    mse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
    results[name] = {
        'MSE': {
            'mean': mse_scores.mean(),
            'std': mse_scores.std()
        },
        'R²': {
            'mean': r2_scores.mean(),
            'std': r2_scores.std()
        }
    }

# 显示结果
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']['mean']:.4f} ± {metrics['MSE']['std']:.4f}")
    print(f"  R²: {metrics['R²']['mean']:.4f} ± {metrics['R²']['std']:.4f}")

# 可视化比较
plt.figure(figsize=(12, 6))
names = list(results.keys())
mse_means = [results[name]['MSE']['mean'] for name in names]
mse_stds = [results[name]['MSE']['std'] for name in names]

plt.barh(names, mse_means, xerr=mse_stds, alpha=0.7)
plt.xlabel('MSE')
plt.title('不同回归模型的MSE比较')
plt.grid(axis='x')
plt.tight_layout()
plt.show()
```

  </div>
</div>

## 超参数优化

使用网格搜索或随机搜索优化模型超参数。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import GridSearchCV

# 假设我们选择了随机森林作为最佳模型
best_model = RandomForestRegressor(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
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
grid_search.fit(X, y)

# 最佳参数
print("最佳参数:")
print(grid_search.best_params_)

# 最佳性能
print(f"最佳MSE: {-grid_search.best_score_:.4f}")

# 使用最佳参数创建模型
best_rf = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    random_state=42
)
```

  </div>
</div>

## 模型集成

组合多个模型可以提高预测性能。

### 投票回归

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import VotingRegressor

# 选择表现最好的几个模型
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR(kernel='rbf', C=10))
]

# 创建投票回归器
voting_reg = VotingRegressor(estimators=estimators)

# 评估
scores = cross_val_score(voting_reg, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"投票回归MSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

  </div>
</div>

### 堆叠回归

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import StackingRegressor

# 定义基础模型
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10))
    ]))
]

# 定义元模型
meta_model = Ridge()

# 创建堆叠回归器
stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# 评估
scores = cross_val_score(stacking_reg, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"堆叠回归MSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

  </div>
</div>

## 特征重要性分析

了解特征对模型预测的影响可以帮助我们选择更合适的模型。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.title('特征重要性')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [f'特征 {i}' for i in indices], rotation=90)
plt.tight_layout()
plt.show()
```

  </div>
</div>

## 模型解释性

在某些应用中，模型的可解释性与预测性能同样重要。

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>解释性</th>
        <th>解释方法</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>线性回归</td>
        <td>高</td>
        <td>系数大小和符号</td>
      </tr>
      <tr>
        <td>决策树</td>
        <td>中高</td>
        <td>树结构和决策路径</td>
      </tr>
      <tr>
        <td>随机森林</td>
        <td>中</td>
        <td>特征重要性</td>
      </tr>
      <tr>
        <td>梯度提升树</td>
        <td>中</td>
        <td>特征重要性、部分依赖图</td>
      </tr>
      <tr>
        <td>支持向量回归</td>
        <td>低</td>
        <td>SHAP值、排列重要性</td>
      </tr>
      <tr>
        <td>神经网络</td>
        <td>很低</td>
        <td>SHAP值、排列重要性</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>过度追求复杂模型</strong>：简单模型可能更稳定且易于维护</li>
      <li><strong>忽略领域知识</strong>：领域知识可以指导特征工程和模型选择</li>
      <li><strong>过度依赖单一评估指标</strong>：应考虑多种指标和业务需求</li>
      <li><strong>忽略计算成本</strong>：在生产环境中，模型的训练和预测时间也很重要</li>
    </ul>
  </div>
</div>

## 小结与思考

选择合适的回归模型是数据科学工作流程中的关键步骤，需要综合考虑多种因素。

### 关键要点回顾

- 不同回归模型有各自的优缺点和适用场景
- 交叉验证是比较模型性能的可靠方法
- 超参数调优可以显著提高模型性能
- 模型集成通常比单个模型表现更好
- 模型选择应平衡预测性能、解释性和计算成本

### 思考问题

1. 在什么情况下应该选择简单模型而非复杂模型？
2. 如何平衡模型的预测性能和解释性？
3. 模型集成为什么通常能提高预测性能？

<div class="practice-link">
  <a href="/projects/regression.html" class="button">前往实践项目</a>
</div> 