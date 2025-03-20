# 房价预测模型

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：中级</li>
      <li><strong>类型</strong>：回归 - 中级</li>
      <!-- <li><strong>预计时间</strong>：4-6小时</li> -->
      <li><strong>技能点</strong>：特征工程、回归模型、模型评估</li>
      <li><strong>对应知识模块</strong>：<a href="/core/regression/linear-regression.html">回归分析</a></li>
    </ul>
  </div>
</div>

## 项目背景

房价预测是机器学习中的经典问题，对购房者、销售者和投资者都有重要意义。通过分析房屋特征（如面积、位置、房间数量等），我们可以构建模型来预测房屋的市场价值。

在这个项目中，我们将使用波士顿郊区的房屋数据集，构建一个房价预测模型，探索影响房价的关键因素。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>许多房地产网站如Zillow都使用机器学习算法提供自动估价服务。Zillow的"Zestimate"使用机器学习模型为美国超过1亿套房产提供估价，平均误差率约为2-3%。</p>
  </div>
</div>

## 数据集介绍

我们将使用波士顿房价数据集，这是一个包含波士顿郊区506个社区的房屋信息的数据集。每个样本有13个特征：

- CRIM：城镇人均犯罪率
- ZN：占地面积超过25,000平方英尺的住宅用地比例
- INDUS：每个城镇非零售业务的比例
- CHAS：查尔斯河虚拟变量（1表示靠河，0表示不靠河）
- NOX：一氧化氮浓度
- RM：每栋住宅的平均房间数
- AGE：1940年之前建成的自住单位比例
- DIS：到波士顿五个就业中心的加权距离
- RAD：径向公路的可达性指数
- TAX：每10,000美元的全额财产税率
- PTRATIO：城镇师生比例
- B：1000(Bk - 0.63)^2，其中Bk是城镇黑人比例
- LSTAT：人口中地位较低人群的百分比

目标变量是MEDV，即自住房的中位数价值（单位：千美元）。

## 项目目标

1. 构建一个能够准确预测房价的回归模型
2. 识别影响房价的关键特征
3. 比较不同回归算法的性能
4. 评估模型在不同评估指标下的表现

## 实施步骤

### 1. 数据探索与可视化

首先，我们需要了解数据的基本特征和分布：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 加载数据
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# 查看数据基本信息
print(df.info())
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 可视化目标变量分布
plt.figure(figsize=(10, 6))
sns.histplot(df['MEDV'], kde=True)
plt.title('房价分布')
plt.xlabel('房价（千美元）')
plt.ylabel('频率')
plt.show()

# 查看特征与目标变量的相关性
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性矩阵')
plt.show()

# 查看重要特征与房价的散点图
important_features = ['RM', 'LSTAT', 'PTRATIO', 'DIS']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(important_features):
    sns.scatterplot(x=feature, y='MEDV', data=df, ax=axes[i])
    axes[i].set_title(f'{feature} vs 房价')

plt.tight_layout()
plt.show()
```

### 2. 数据预处理

接下来，我们需要准备数据用于模型训练：

```python
# 分离特征和目标变量
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. 模型训练与评估

我们将尝试多种回归模型，并比较它们的性能：

```python
# 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
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
    
    print(f"训练集RMSE: {train_rmse:.2f}")
    print(f"测试集RMSE: {test_rmse:.2f}")
    print(f"训练集R²: {train_r2:.2f}")
    print(f"测试集R²: {test_r2:.2f}")
    print(f"测试集MAE: {test_mae:.2f}")
    
    return model, y_test_pred

# 线性回归
print("线性回归模型:")
lr_model, lr_pred = evaluate_model(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test)
print("\n")

# 岭回归
print("岭回归模型:")
ridge_model, ridge_pred = evaluate_model(Ridge(alpha=1.0), X_train_scaled, X_test_scaled, y_train, y_test)
print("\n")

# Lasso回归
print("Lasso回归模型:")
lasso_model, lasso_pred = evaluate_model(Lasso(alpha=0.1), X_train_scaled, X_test_scaled, y_train, y_test)
print("\n")

# 决策树回归
print("决策树回归模型:")
dt_model, dt_pred = evaluate_model(DecisionTreeRegressor(max_depth=5), X_train, X_test, y_train, y_test)
print("\n")

# 随机森林回归
print("随机森林回归模型:")
rf_model, rf_pred = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=10), X_train, X_test, y_train, y_test)
```

### 4. 特征重要性分析

了解哪些特征对房价影响最大：

```python
# 使用随机森林模型分析特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('特征重要性')
plt.tight_layout()
plt.show()
```

### 5. 预测可视化

可视化模型预测结果与实际值的对比：

```python
# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('实际房价')
plt.ylabel('预测房价')
plt.title('随机森林模型：实际房价 vs 预测房价')
plt.tight_layout()
plt.show()

# 残差分析
residuals = y_test - rf_pred
plt.figure(figsize=(10, 6))
plt.scatter(rf_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测房价')
plt.ylabel('残差')
plt.title('残差分析')
plt.tight_layout()
plt.show()
```

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **特征工程**：创建新特征，如房间面积比（RM/LSTAT）或交互特征
2. **超参数调优**：使用网格搜索或随机搜索优化模型参数
3. **集成方法**：尝试使用堆叠或投票等集成方法提高预测性能
4. **非线性变换**：对特征或目标变量应用对数或多项式变换
5. **交叉验证**：实现k折交叉验证，获得更稳健的模型评估

## 小结与反思

通过这个项目，我们学习了如何构建房价预测模型，从数据探索到模型评估的完整流程。我们发现房间数量、低收入人口比例和师生比等因素对房价有显著影响。

在实际应用中，房价预测模型可以帮助购房者评估房屋的合理价格，帮助销售者制定合适的定价策略，也可以帮助开发商识别有潜力的区域。

### 思考问题

1. 除了我们使用的特征外，还有哪些因素可能影响房价？如何获取这些数据？
2. 我们的模型在哪些类型的房屋上预测效果较差？为什么？
3. 如何将这个模型应用到其他城市？需要考虑哪些因素？

<div class="practice-link">
  <a href="/projects/regression/sales-forecast.html" class="button">下一个项目：销售额预测</a>
</div> 