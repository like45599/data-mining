# 泰坦尼克号生存预测

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：入门级</li>
      <li><strong>类型</strong>：二元分类</li>
      <li><strong>预计时间</strong>：4-6小时</li>
      <li><strong>技能点</strong>：数据清洗、特征工程、分类算法、模型评估</li>
    </ul>
  </div>
</div>

## 项目背景

1912年4月15日，被誉为"永不沉没"的豪华客轮泰坦尼克号在首航期间与冰山相撞沉没，2224名乘客和船员中有1502人遇难。这场悲剧震惊了国际社会，并导致了船舶安全条例的改进。

在这个项目中，我们将分析泰坦尼克号的乘客数据，尝试回答"什么样的人更有可能生存？"这个问题。通过构建预测模型，我们可以发现影响生存率的关键因素。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>泰坦尼克号的沉没展示了社会阶层在灾难中的影响。当时的"女士和儿童优先"政策以及不同舱位的位置导致了生存率的显著差异。一等舱乘客的生存率约为62%，二等舱为41%，而三等舱仅为25%。</p>
  </div>
</div>

## 数据集介绍

本项目使用的数据集包含泰坦尼克号上891名乘客的信息。每位乘客的记录包含以下特征：

- **PassengerId**：乘客ID
- **Survived**：是否存活（0=否，1=是）
- **Pclass**：船票等级（1=一等舱，2=二等舱，3=三等舱）
- **Name**：乘客姓名
- **Sex**：性别
- **Age**：年龄
- **SibSp**：同行的兄弟姐妹/配偶数量
- **Parch**：同行的父母/子女数量
- **Ticket**：船票号码
- **Fare**：船票价格
- **Cabin**：客舱号码
- **Embarked**：登船港口（C=瑟堡，Q=皇后镇，S=南安普顿）

数据集中存在一些缺失值，特别是Age、Cabin和Embarked特征。处理这些缺失值将是数据预处理的重要部分。

## 项目目标

1. 探索和分析泰坦尼克号乘客数据
2. 处理缺失值和准备特征
3. 构建分类模型预测乘客生存情况
4. 评估模型性能并解释结果
5. 提交预测结果

## 实施步骤

### 步骤1：数据探索与可视化

首先，我们需要加载数据并进行初步探索，了解数据的基本特征和分布。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 加载数据
train_data = pd.read_csv('titanic_train.csv')

# 查看数据基本信息
print(train_data.shape)
train_data.head()

# 检查缺失值
train_data.isnull().sum()

# 基本统计描述
train_data.describe()

# 生存率概览
train_data['Survived'].value_counts(normalize=True)

# 可视化不同特征与生存率的关系
plt.figure(figsize=(12, 5))

plt.subplot(131)
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Gender')

plt.subplot(132)
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Passenger Class')

plt.subplot(133)
sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.title('Survival Rate by Embarkation Port')

plt.tight_layout()
plt.show()
```

### 步骤2：数据预处理

接下来，我们需要处理缺失值，并将分类特征转换为模型可用的格式。

```python
# 处理缺失的年龄数据
age_median = train_data['Age'].median()
train_data['Age'].fillna(age_median, inplace=True)

# 处理缺失的登船港口数据
embarked_mode = train_data['Embarked'].mode()[0]
train_data['Embarked'].fillna(embarked_mode, inplace=True)

# 创建新特征：家庭规模
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# 创建新特征：是否单独旅行
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

# 从姓名中提取称谓
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 将称谓分组
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Don": "Royalty",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Sir": "Royalty",
    "Capt": "Officer",
    "Ms": "Mrs"
}
train_data['Title'] = train_data['Title'].map(title_mapping)

# 选择要使用的特征
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']

# 对分类特征进行独热编码
train_encoded = pd.get_dummies(train_data[features])

# 准备特征矩阵和目标变量
X = train_encoded
y = train_data['Survived']
```

### 步骤3：模型构建与评估

现在我们可以构建分类模型并评估其性能。

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 决策树模型
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_val)
dt_accuracy = accuracy_score(y_val, dt_pred)
print(f"决策树准确率: {dt_accuracy:.4f}")

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_pred)
print(f"随机森林准确率: {rf_accuracy:.4f}")

# SVM模型
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_val)
svm_accuracy = accuracy_score(y_val, svm_pred)
print(f"SVM准确率: {svm_accuracy:.4f}")

# 交叉验证
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} 交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 查看最佳模型的混淆矩阵和分类报告
print("随机森林混淆矩阵:")
print(confusion_matrix(y_val, rf_pred))
print("\n随机森林分类报告:")
print(classification_report(y_val, rf_pred))

# 特征重要性
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
```

### 步骤4：模型优化

我们可以通过超参数调优来进一步提高模型性能。

```python
from sklearn.model_selection import GridSearchCV

# 随机森林参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# 最佳参数
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)

# 使用最佳参数的模型
best_rf_model = grid_search.best_estimator_
```

### 步骤5：预测测试集

最后，我们使用最佳模型对测试集进行预测。

```python
# 加载测试数据
test_data = pd.read_csv('titanic_test.csv')

# 对测试数据进行相同的预处理
# (重复上面的预处理步骤)

# 使用最佳模型预测
test_predictions = best_rf_model.predict(test_encoded)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)
```

## 结果分析

通过我们的分析和建模，我们发现以下因素对泰坦尼克号乘客生存率有显著影响：

1. **性别**：女性的生存率显著高于男性
2. **船票等级**：一等舱乘客生存率高于二等舱和三等舱
3. **年龄**：儿童的生存率高于成人
4. **家庭规模**：中等规模家庭的成员生存率较高
5. **称谓**：反映社会地位的称谓与生存率相关

这些发现与历史记录一致，反映了"女士和儿童优先"的救生政策以及社会阶层对生存机会的影响。

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **特征工程**：尝试创建更多有意义的特征，如票价区间、年龄组等
2. **模型集成**：使用投票或堆叠方法组合多个模型的预测
3. **生存概率分析**：不仅预测生存与否，还分析不同特征组合的生存概率
4. **可视化**：创建更高级的可视化，如决策树可视化或生存率热图
5. **缺失值处理**：尝试更复杂的缺失值填充方法，如基于相似乘客特征的填充

## 小结与反思

通过这个项目，我们学习了如何处理一个完整的分类问题，从数据探索到模型部署。泰坦尼克号数据集虽小，但包含了数据挖掘中常见的挑战，如缺失值处理、特征工程和模型选择。

在实际应用中，这类分析可以帮助我们理解影响某一结果的关键因素，从而制定更有效的策略和政策。例如，在灾难管理中，类似的分析可以帮助识别高风险群体，优先分配资源。

### 思考问题

1. 我们的模型是否存在偏见？如何确保预测的公平性？
2. 如何将这种分析方法应用到其他灾难或事件预测中？
3. 除了准确率外，还有哪些指标可以评估模型的有效性？

<div class="practice-link">
  <a href="/projects/iris.html" class="button">下一个项目：鸢尾花分类</a>
</div> 