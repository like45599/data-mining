# 决策树算法

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解决策树的基本原理和构建过程</li>
      <li>掌握决策树的分裂标准和剪枝技术</li>
      <li>学习决策树在分类和回归问题中的应用</li>
      <li>了解决策树的优缺点及其改进方法</li>
    </ul>
  </div>
</div>

## 决策树基本原理

决策树是一种树形结构的分类和回归模型，通过一系列问题将数据划分为不同的子集，最终得到预测结果。

### 决策树结构

决策树由以下部分组成：

- **根节点**：包含所有样本的起始节点
- **内部节点**：表示特征的测试条件
- **分支**：表示测试条件的结果
- **叶节点**：表示最终的分类或回归结果

<div class="visualization-container">
  <div class="visualization-title">决策树结构示例</div>
  <div class="visualization-content">
    <img src="/images/decision_tree_structure.svg" alt="决策树结构示例">
  </div>
  <div class="visualization-caption">
    图: 一个简单的决策树示例。根节点和内部节点表示特征测试，分支表示测试结果，叶节点表示最终分类。
  </div>
</div>

### 决策树构建过程

决策树的构建通常采用自顶向下的递归方式，主要步骤包括：

1. **选择最佳特征**：使用信息增益、基尼不纯度等指标选择最佳分裂特征
2. **分裂数据集**：根据选定的特征将数据集分为子集
3. **递归构建子树**：对每个子集重复上述过程
4. **确定停止条件**：当满足特定条件（如达到最大深度、节点样本数过少）时停止分裂

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>决策树算法的早期版本CART(Classification and Regression Trees)由Leo Breiman等人于1984年提出。而更早的ID3算法则由Ross Quinlan在1979年开发。决策树不仅是一种强大的机器学习算法，也是随机森林、梯度提升树等集成方法的基础。</p>
  </div>
</div>

## 分裂标准

决策树算法的核心是如何选择最佳分裂特征和分裂点。常用的分裂标准包括：

### 1. 信息增益（ID3算法）

基于信息熵的减少量，熵表示系统的不确定性：

$$\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

其中$p_i$是类别$i$的样本比例，$c$是类别数量。

信息增益计算为：

$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$

其中$A$是特征，$S_v$是特征$A$取值为$v$的样本子集。

### 2. 增益率（C4.5算法）

为克服信息增益偏向多值特征的缺点，C4.5引入了增益率：

$$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$

其中：

$$\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

### 3. 基尼不纯度（CART算法）

衡量集合的不纯度，值越小表示集合中的样本越纯净：

$$\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$$

基尼指数计算为：

$$\text{Gini\_Index}(S, A) = \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Gini}(S_v)$$

选择使基尼指数最小的特征作为最佳分裂特征。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练决策树模型（使用基尼不纯度）
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_gini.fit(X_train, y_train)

# 创建并训练决策树模型（使用信息增益）
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_entropy.fit(X_train, y_train)

# 预测
y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

# 评估
gini_accuracy = dt_gini.score(X_test, y_test)
entropy_accuracy = dt_entropy.score(X_test, y_test)

print(f"基尼不纯度决策树准确率: {gini_accuracy:.4f}")
print(f"信息增益决策树准确率: {entropy_accuracy:.4f}")

# 可视化决策树
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_gini, feature_names=iris.feature_names, 
               class_names=iris.target_names, filled=True)
plt.title("基尼不纯度决策树")
plt.show()
```

  </div>
</div>

## 决策树剪枝

决策树容易过拟合，剪枝是防止过拟合的重要技术。

### 预剪枝

在构建过程中预先停止树的生长：

- 限制树的最大深度
- 限制节点的最小样本数
- 限制分裂的最小信息增益

### 后剪枝

先构建完整树，然后自底向上剪去不重要的子树：

- 错误率降低剪枝
- 代价复杂度剪枝（CART算法使用）

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 加载数据
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义参数网格
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.02, 0.03, 0.04]  # 代价复杂度参数，用于后剪枝
}

# 创建网格搜索
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.4f}")

# 使用最佳模型预测
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"测试集准确率: {test_score:.4f}")

# 可视化最佳决策树
plt.figure(figsize=(15, 10))
tree.plot_tree(best_model, feature_names=cancer.feature_names, 
               class_names=['恶性', '良性'], filled=True, max_depth=3)
plt.title("最佳决策树（限制显示深度为3）")
plt.show()
```

  </div>
</div>

## 特征重要性

决策树可以计算特征重要性，帮助理解哪些特征对预测最有影响：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
import matplotlib.pyplot as plt

# 使用之前训练的最佳模型
feature_importance = best_model.feature_importances_

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': feature_importance
})

# 按重要性排序
importance_df = importance_df.sort_values('importance', ascending=False)

# 可视化前15个重要特征
plt.figure(figsize=(12, 8))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('决策树特征重要性')
plt.gca().invert_yaxis()  # 从上到下显示
plt.show()
```

  </div>
</div>

## 决策树回归

决策树不仅可用于分类，也可用于回归问题：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据
boston = load_boston()
X, y = boston.data, boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练决策树回归模型
regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

# 预测
y_pred = regressor.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('决策树回归：实际值 vs 预测值')
plt.show()
```

  </div>
</div>

## 决策树的优缺点

### 优点

1. **易于理解和解释**：决策树的决策过程直观可视化
2. **无需特征缩放**：对特征尺度不敏感
3. **能处理数值和类别特征**：无需特殊编码
4. **能处理多分类问题**：自然支持多类别分类
5. **可以处理缺失值**：能够学习处理缺失值的模式

### 缺点

1. **容易过拟合**：特别是树深度较大时
2. **不稳定**：数据微小变化可能导致树结构显著变化
3. **偏向于高基数特征**：倾向于选择取值较多的特征
4. **难以学习某些关系**：如XOR关系
5. **预测性能有限**：单棵树的预测能力通常不如集成方法

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>忽略剪枝</strong>：未进行适当剪枝导致过拟合</li>
      <li><strong>过度依赖单棵树</strong>：在复杂问题上未考虑使用集成方法</li>
      <li><strong>忽略类别不平衡</strong>：未设置class_weight参数处理不平衡数据</li>
      <li><strong>错误解读特征重要性</strong>：特征重要性不等同于因果关系</li>
    </ul>
  </div>
</div>

## 决策树的改进与扩展

为克服决策树的局限性，出现了多种改进和扩展方法：

1. **随机森林**：构建多棵树并取平均/多数投票
2. **梯度提升树**：通过梯度提升方法串行构建树
3. **XGBoost/LightGBM**：高效实现的梯度提升树库

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"随机森林准确率: {accuracy:.4f}")

# 与单棵决策树比较
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"单棵决策树准确率: {dt_accuracy:.4f}")
```

  </div>
</div>

## 小结与思考

决策树是一种直观且强大的机器学习算法，既可用于分类也可用于回归。尽管单棵决策树有一定局限性，但它是许多高级集成方法的基础。

### 关键要点回顾

- 决策树通过递归分裂数据构建树形结构
- 常用分裂标准包括信息增益、增益率和基尼不纯度
- 剪枝技术对防止过拟合至关重要
- 决策树可以计算特征重要性，提供模型解释性
- 随机森林等集成方法可以克服单棵树的局限性

### 思考问题

1. 在什么情况下应该选择决策树而非其他分类/回归算法？
2. 如何平衡决策树的复杂度和预测性能？
3. 为什么随机森林通常比单棵决策树表现更好？

<BackToPath />

<div class="practice-link">
  <a href="/projects/classification.html" class="button">前往实践项目</a>
</div> 