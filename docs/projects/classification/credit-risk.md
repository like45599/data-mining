# 信用风险评估

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：高级</li>
      <li><strong>类型</strong>：分类</li>
      <!-- <li><strong>预计时间</strong>：6-8小时</li> -->
      <li><strong>技能点</strong>：不平衡数据处理、特征工程、模型解释、评估指标选择</li>
      <li><strong>对应知识模块</strong>：<a href="/core/classification/svm.html">分类算法</a></li>
    </ul>
  </div>
</div>

## 项目背景

信用风险评估是金融机构的核心业务之一，用于判断借款人是否有能力按时偿还贷款。准确的信用风险评估可以帮助金融机构降低坏账率，同时为更多合格的借款人提供金融服务。

传统的信用评分模型主要依赖于借款人的信用历史、收入水平和债务比率等因素。随着大数据和机器学习技术的发展，现代信用风险评估可以整合更多维度的数据，包括交易行为、社交网络和心理特征等，构建更全面和准确的风险预测模型。

在这个项目中，我们将使用德国信用数据集，构建一个信用风险评估模型，预测借款人是否会违约。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>信用评分的概念最早由美国公平艾萨克公司(Fair Isaac Corporation)在1950年代提出，发展成为今天广泛使用的FICO评分系统。现代信用评分系统通常将借款人的信用状况量化为300-850之间的分数，分数越高表示信用风险越低。</p>
  </div>
</div>

## 数据集介绍

德国信用数据集包含1000名借款人的信息，每个借款人有20个特征，包括：

- 个人信息：年龄、性别、婚姻状况等
- 财务状况：收入、储蓄、现有信贷数量等
- 就业信息：就业类型、就业年限等
- 住房信息：住房类型、居住时间等
- 信用历史：过往信用记录、贷款目的等

目标变量是二元分类：好客户(0)或坏客户(1)，其中坏客户表示存在违约风险。

数据集的一个重要特点是类别不平衡，好客户占70%，坏客户占30%。这种不平衡反映了现实世界中的信用风险分布，但也给模型训练带来了挑战。

## 项目目标

1. 构建一个能够准确预测借款人信用风险的分类模型
2. 处理数据集中的类别不平衡问题
3. 识别影响信用风险的关键因素
4. 评估模型在不同评估指标下的表现
5. 提供模型解释和业务建议

## 实施步骤

### 1. 数据探索与预处理

首先，我们需要了解数据的基本特征和质量：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 加载数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ['status', 'duration', 'credit_history', 'purpose', 'amount', 
                'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 
                'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 
                'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target']
df = pd.read_csv(url, sep=' ', header=None, names=column_names)

# 将目标变量转换为0和1（原始数据中是1和2）
df['target'] = df['target'].map({1: 0, 2: 1})

# 查看数据基本信息
print(df.info())
print(df.describe())

# 检查类别分布
print(df['target'].value_counts(normalize=True))

# 可视化类别分布
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('信用风险分布')
plt.xlabel('风险类别 (0=好客户, 1=坏客户)')
plt.ylabel('数量')
plt.show()

# 探索数值特征与目标变量的关系
numerical_features = ['duration', 'amount', 'age', 'existing_credits']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.boxplot(x='target', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'{feature} vs 信用风险')
    axes[i].set_xlabel('风险类别 (0=好客户, 1=坏客户)')

plt.tight_layout()
plt.show()

# 探索类别特征与目标变量的关系
categorical_features = ['credit_history', 'purpose', 'personal_status_sex', 'savings']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    cross_tab = pd.crosstab(df[feature], df['target'], normalize='index')
    cross_tab.plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(f'{feature} vs 信用风险')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('比例')
    axes[i].legend(['好客户', '坏客户'])

plt.tight_layout()
plt.show()
```

接下来，我们需要处理数据中的类别特征和数值特征：

```python
# 识别数值特征和类别特征
numerical_features = ['duration', 'amount', 'installment_rate', 'present_residence', 
                      'age', 'existing_credits', 'num_dependents']
categorical_features = ['status', 'credit_history', 'purpose', 'savings', 
                        'employment_duration', 'personal_status_sex', 'other_debtors', 
                        'property', 'other_installment_plans', 'housing', 'job', 
                        'telephone', 'foreign_worker']

# 分割数据
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 应用预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 检查处理后的特征维度
print(f"处理后的训练集形状: {X_train_processed.shape}")
```

### 2. 处理类别不平衡

信用风险数据通常存在类别不平衡问题，我们可以使用多种方法处理：

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# 查看原始类别分布
print(f"原始训练集类别分布: {Counter(y_train)}")

# 使用SMOTE过采样
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
print(f"SMOTE后训练集类别分布: {Counter(y_train_smote)}")

# 使用随机欠采样
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_processed, y_train)
print(f"欠采样后训练集类别分布: {Counter(y_train_rus)}")

# 使用组合采样（SMOTE + 欠采样）
over = SMOTE(sampling_strategy=0.5, random_state=42)  # 将少数类过采样到多数类的50%
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # 将多数类欠采样到少数类的80%
combined_sampling = ImbPipeline(steps=[('over', over), ('under', under)])
X_train_combined, y_train_combined = combined_sampling.fit_resample(X_train_processed, y_train)
print(f"组合采样后训练集类别分布: {Counter(y_train_combined)}")
```

### 3. 模型训练与评估

我们将尝试多种分类算法，并使用适合不平衡数据的评估指标：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score

# 定义评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # 打印评估结果
    print(f"\n{model_name} 评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.show()
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend()
    plt.show()
    
    # 精确率-召回率曲线
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve)
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'{model_name} 精确率-召回率曲线')
    plt.show()
    
    return model, accuracy, precision, recall, f1, auc

# 训练和评估多个模型
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
}

results = []
for name, model in models.items():
    model_result = evaluate_model(model, X_train_smote, y_train_smote, X_test_processed, y_test, name)
    results.append((name,) + model_result[1:])

# 比较不同模型的性能
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
print("\n模型性能比较:")
print(results_df)

# 可视化比较
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4. 特征重要性分析

了解哪些特征对信用风险预测最重要：

```python
# 使用随机森林的特征重要性
rf_model = models['Random Forest']
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

# 获取特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 可视化前15个重要特征
plt.figure(figsize=(12, 8))
plt.title('特征重要性')
plt.bar(range(15), importances[indices[:15]], align='center')
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=90)
plt.tight_layout()
plt.show()

# 使用SHAP值进行更详细的解释
import shap

# 选择一个模型进行解释（例如，随机森林）
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_processed)

# 可视化SHAP值
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values[1], X_test_processed, feature_names=feature_names)
```

### 5. 模型优化

使用交叉验证和网格搜索优化最佳模型：

```python
from sklearn.model_selection import GridSearchCV

# 假设随机森林是最佳模型
best_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用网格搜索找到最佳参数
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # 使用F1分数作为优化目标
    n_jobs=-1
)

grid_search.fit(X_train_smote, y_train_smote)

# 打印最佳参数
print("最佳参数:")
print(grid_search.best_params_)

# 使用最佳参数的模型进行评估
best_rf = grid_search.best_estimator_
evaluate_model(best_rf, X_train_smote, y_train_smote, X_test_processed, y_test, "优化后的随机森林")
```

### 6. 阈值优化

在信用风险评估中，错误分类的成本是不对称的。我们可以通过调整分类阈值来平衡精确率和召回率：

```python
# 获取预测概率
y_prob = best_rf.predict_proba(X_test_processed)[:, 1]

# 计算不同阈值下的精确率和召回率
thresholds = np.arange(0, 1, 0.01)
precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_threshold))
    recall_scores.append(recall_score(y_test, y_pred_threshold))
    f1_scores.append(f1_score(y_test, y_pred_threshold))

# 可视化不同阈值下的性能
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='精确率')
plt.plot(thresholds, recall_scores, label='召回率')
plt.plot(thresholds, f1_scores, label='F1分数')
plt.xlabel('阈值')
plt.ylabel('分数')
plt.title('不同阈值下的模型性能')
plt.legend()
plt.grid(True)
plt.show()

# 找到F1分数最高的阈值
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"最佳阈值: {best_threshold:.2f}")
print(f"在最佳阈值下的F1分数: {max(f1_scores):.4f}")

# 使用最佳阈值进行最终预测
y_pred_final = (y_prob >= best_threshold).astype(int)
print("\n使用最佳阈值的最终评估结果:")
print(classification_report(y_test, y_pred_final))
```

### 7. 业务解释与建议

最后，我们需要将模型结果转化为业务洞察：

```python
# 计算拒绝率和坏账率
def calculate_business_metrics(y_true, y_pred, y_prob, threshold):
    # 使用给定阈值的预测
    y_pred_threshold = (y_prob >= threshold).astype(int)
    
    # 计算拒绝率（预测为坏客户的比例）
    rejection_rate = np.mean(y_pred_threshold)
    
    # 计算坏账率（在预测为好客户中实际为坏客户的比例）
    approved_indices = y_pred_threshold == 0
    if np.sum(approved_indices) > 0:
        bad_debt_rate = np.mean(y_true[approved_indices] == 1)
    else:
        bad_debt_rate = 0
    
    return rejection_rate, bad_debt_rate

# 计算不同阈值下的业务指标
business_metrics = []
for threshold in thresholds:
    rejection_rate, bad_debt_rate = calculate_business_metrics(y_test, y_pred, y_prob, threshold)
    business_metrics.append((threshold, rejection_rate, bad_debt_rate))

business_df = pd.DataFrame(business_metrics, columns=['Threshold', 'Rejection Rate', 'Bad Debt Rate'])

# 可视化业务指标
plt.figure(figsize=(10, 6))
plt.plot(business_df['Threshold'], business_df['Rejection Rate'], label='拒绝率')
plt.plot(business_df['Threshold'], business_df['Bad Debt Rate'], label='坏账率')
plt.xlabel('阈值')
plt.ylabel('比率')
plt.title('不同阈值下的业务指标')
plt.legend()
plt.grid(True)
plt.show()

# 根据业务目标选择阈值
# 例如，如果我们希望坏账率不超过5%
target_bad_debt = 0.05
valid_thresholds = business_df[business_df['Bad Debt Rate'] <= target_bad_debt]
if not valid_thresholds.empty:
    business_threshold = valid_thresholds.iloc[np.argmin(valid_thresholds['Rejection Rate'])]['Threshold']
    print(f"为达到坏账率不超过5%的目标，建议使用阈值: {business_threshold:.2f}")
    print(f"在此阈值下的拒绝率: {valid_thresholds.iloc[np.argmin(valid_thresholds['Rejection Rate'])]['Rejection Rate']:.2f}")
    print(f"在此阈值下的坏账率: {valid_thresholds.iloc[np.argmin(valid_thresholds['Rejection Rate'])]['Bad Debt Rate']:.2f}")
else:
    print("无法找到满足坏账率目标的阈值")
```

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **成本敏感学习**：考虑不同类型错误的业务成本，构建成本敏感的模型
2. **特征工程进阶**：创建交互特征、多项式特征或基于领域知识的特征
3. **模型解释进阶**：使用LIME或SHAP等工具提供个体预测的解释
4. **公平性分析**：评估模型在不同人口统计群体上的表现，检测和减轻潜在偏见
5. **部署考量**：设计模型监控和更新策略，处理概念漂移问题

## 小结与反思

通过这个项目，我们学习了如何构建信用风险评估模型，处理类别不平衡问题，并将模型结果转化为业务决策。信用风险评估是一个复杂的任务，需要平衡多种因素，包括预测准确性、业务目标和伦理考量。

在实际应用中，信用风险模型需要定期更新和监控，以适应不断变化的经济环境和客户行为。此外，模型的公平性和透明度也是重要的考量因素，确保信贷决策不会对特定群体造成不公平的影响。

### 思考问题

1. 在信用风险评估中，如何平衡模型的预测性能和可解释性需求？
2. 不同的采样方法如何影响模型对少数类（高风险客户）的预测能力？
3. 如何将信用风险模型与业务流程集成，使其成为有效的决策支持工具？

<div class="practice-link">
  <a href="/projects/clustering/customer-segmentation.html" class="button">下一个模块：聚类分析项目</a>
</div> 