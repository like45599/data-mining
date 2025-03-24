# 支持向量机(SVM)

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解SVM的基本原理和数学基础</li>
      <li>掌握线性与非线性SVM的区别</li>
      <li>学习核函数的作用和选择方法</li>
      <li>实践SVM在分类问题中的应用</li>
    </ul>
  </div>
</div>

## SVM基本原理

支持向量机(Support Vector Machine, SVM)是一种强大的监督学习算法，广泛应用于分类和回归问题。SVM的核心思想是找到一个最优的超平面，使其能够最大化不同类别数据点之间的间隔。

### 线性可分情况

在最简单的二分类线性可分情况下，SVM尝试找到一个超平面，使得：

1. 能够正确分类所有训练样本
2. 到最近的训练样本点的距离（间隔）最大

<div class="visualization-container">
  <div class="visualization-title">线性SVM原理</div>
  <div class="visualization-content">
    <img src="/images/svm_linear.svg" alt="线性SVM原理图">
  </div>
  <div class="visualization-caption">
    图: 线性SVM的决策边界和支持向量。红色和蓝色点代表两个不同类别，黑色实线是决策边界，虚线表示间隔边界，圈出的点是支持向量。
  </div>
</div>

### 数学表达

线性SVM的优化问题可以表示为：

$$\min_{w, b} \frac{1}{2} ||w||^2$$

$$\text{s.t. } y_i(w^T x_i + b) \geq 1, \forall i=1,\ldots,n$$

其中：
- $w$ 是法向量，决定超平面的方向
- $b$ 是偏置项，决定超平面的位置
- $x_i$ 是特征向量
- $y_i$ 是类别标签（+1或-1）
- 约束条件确保所有样本都被正确分类且间隔至少为1

### 支持向量

支持向量是距离决策边界最近的数据点，它们满足：

$$y_i(w^T x_i + b) = 1$$

这些点对确定决策边界至关重要，而其他点则不影响模型。这是SVM的一个重要特性，使其在高维空间中仍然高效。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>SVM算法最早由Vladimir Vapnik和Alexey Chervonenkis在1963年提出，但直到1990年代才因为核方法的引入而广泛流行。SVM的理论基础来自统计学习理论中的VC维理论，这一理论为模型的泛化能力提供了理论保证。</p>
  </div>
</div>

## 软间隔SVM

### 处理线性不可分数据

实际应用中，数据通常不是完全线性可分的，可能存在噪声或异常值。软间隔SVM通过引入松弛变量，允许一些样本点违反约束条件：

$$\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0, \forall i=1,\ldots,n$$

其中：
- $\xi_i$ 是松弛变量，表示第i个样本的违反程度
- $C$ 是惩罚参数，控制间隔最大化和误分类样本最小化之间的权衡

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练SVM模型
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 可视化决策边界
def plot_decision_boundary(model, X, y):
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和样本点
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    
    # 标记支持向量
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')
    
    plt.title('SVM决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.show()

plot_decision_boundary(svm, X, y)
```

  </div>
</div>

### C参数的影响

参数C控制正则化强度，影响模型的复杂度：

- **大C值**：强调正确分类每个样本，可能导致过拟合
- **小C值**：允许更多误分类，但追求更大的间隔，通常泛化能力更好

<div class="visualization-container">
  <div class="visualization-title">C参数对SVM的影响</div>
  <div class="visualization-content">
    <img src="/images/svm_c_parameter.svg" alt="C参数对SVM的影响">
  </div>
  <div class="visualization-caption">
    图：不同C值对SVM决策边界的影响。左图C=0.1，中图C=1，右图C=10。
  </div>
</div>

## 核方法

### 处理非线性问题

当数据在原始特征空间中不是线性可分的，SVM使用核技巧将数据映射到更高维的特征空间，在那里可能变得线性可分。

核函数$K(x_i, x_j)$计算两个样本在高维特征空间中的内积，而无需显式计算映射：

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

其中$\phi$是从原始特征空间到高维特征空间的映射函数。

### 常用核函数

1. **线性核**：$K(x_i, x_j) = x_i^T x_j$
   - 适用于线性可分数据
   - 计算效率高

2. **多项式核**：$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$
   - 适用于有明确阶数关系的数据
   - 参数：$\gamma$（缩放）、$r$（偏置）、$d$（多项式阶数）

3. **径向基函数(RBF)核**：$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
   - 最常用的非线性核函数
   - 适用于各种复杂数据
   - 参数：$\gamma$（控制影响半径）

4. **Sigmoid核**：$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$
   - 来源于神经网络
   - 参数：$\gamma$（缩放）、$r$（偏置）

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

# 生成非线性数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 创建不同核函数的SVM模型
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
plt.figure(figsize=(16, 4))

for i, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, gamma=2)
    svm.fit(X, y)
    
    # 可视化
    plt.subplot(1, 4, i+1)
    
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点的类别
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和样本点
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f'核函数: {kernel}')
    plt.xlabel('特征1')
    plt.ylabel('特征2')

plt.tight_layout()
plt.show()
```

  </div>
</div>

### 核函数选择指南

- **线性核**：数据量大、特征多、线性可分
- **RBF核**：数据量中等、特征少、非线性关系
- **多项式核**：特征间存在多项式关系
- **Sigmoid核**：类神经网络问题

## SVM实践应用

### 1. 数据预处理

SVM对特征缩放非常敏感，预处理至关重要：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 创建预处理和SVM的流水线
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 标准化特征
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))  # SVM模型
])

# 使用流水线拟合数据
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
```

  </div>
</div>

### 2. 参数调优

使用网格搜索找到最佳参数：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'poly', 'sigmoid']
}

# 创建网格搜索
grid_search = GridSearchCV(
    pipeline,  # 使用之前定义的流水线
    param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # 使用所有可用CPU
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.4f}")

# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

  </div>
</div>

### 3. 处理不平衡数据

对于类别不平衡的数据集，可以使用类别权重：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
# 使用'balanced'自动调整类别权重
svm = SVC(kernel='rbf', class_weight='balanced')

# 或手动指定权重
class_weights = {0: 1.0, 1: 5.0}  # 假设类别1是少数类
svm = SVC(kernel='rbf', class_weight=class_weights)
```

  </div>
</div>

### 4. SVM用于多分类

SVM本质上是二分类算法，但可以通过以下策略处理多分类问题：

- **一对一(OvO)**：为每对类别训练一个分类器，共需$\frac{n(n-1)}{2}$个分类器
- **一对多(OvR)**：为每个类别训练一个分类器（该类别vs其他所有类别），共需$n$个分类器

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载多分类数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 一对一策略(默认)
svm_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
svm_ovo.fit(X_train, y_train)

# 一对多策略
svm_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
svm_ovr.fit(X_train, y_train)

# 评估
y_pred_ovo = svm_ovo.predict(X_test)
y_pred_ovr = svm_ovr.predict(X_test)

print("一对一策略评估报告:")
print(classification_report(y_test, y_pred_ovo, target_names=iris.target_names))

print("一对多策略评估报告:")
print(classification_report(y_test, y_pred_ovr, target_names=iris.target_names))
```

  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>忽略特征缩放</strong>：SVM对特征尺度非常敏感，未缩放的特征可能导致性能严重下降</li>
      <li><strong>盲目使用RBF核</strong>：虽然RBF核功能强大，但对于大规模线性可分数据，线性核可能更高效</li>
      <li><strong>过度调参</strong>：过度优化C和gamma可能导致过拟合</li>
      <li><strong>忽略类别不平衡</strong>：在不平衡数据上未设置class_weight会导致偏向多数类</li>
    </ul>
  </div>
</div>

## 小结与思考

SVM是一种强大的分类算法，通过最大化间隔和核技巧，能够有效处理线性和非线性分类问题。

### 关键要点回顾

- SVM寻找最大间隔的决策边界
- 支持向量是决定边界的关键点
- 软间隔SVM通过C参数处理噪声和异常值
- 核方法使SVM能够处理非线性问题
- 特征缩放对SVM性能至关重要

### 思考问题

1. SVM与逻辑回归相比有哪些优缺点？
2. 在什么情况下应该选择SVM而非其他分类算法？
3. 如何处理SVM在大规模数据集上的计算效率问题？

<BackToPath />

<div class="practice-link">
  <a href="/projects/classification.html" class="button">前往实践项目</a>
</div> 