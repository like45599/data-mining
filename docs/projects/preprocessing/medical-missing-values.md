# 医疗数据缺失值处理

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：中级</li>
      <li><strong>类型</strong>：缺失值处理</li>
      <!-- <li><strong>预计时间</strong>：5-7小时</li> -->
      <li><strong>技能点</strong>：缺失值模式分析、多重插补、KNN填充、模型预测填充</li>
      <li><strong>对应知识模块</strong>：<a href="/core/preprocessing/data-presentation.html">数据预处理</a></li>
    </ul>
  </div>
</div>

## 项目背景

医疗数据在临床研究、疾病预测和治疗方案优化中起着关键作用。然而，医疗数据集通常包含大量缺失值，这可能是由于设备故障、患者未完成所有检查、记录错误或数据输入问题等原因造成的。不当的缺失值处理可能导致偏差的结论，影响医疗决策的准确性。

在这个项目中，我们将处理一个包含多种缺失值的医疗数据集，比较不同缺失值处理方法的效果，并为后续的疾病预测模型准备高质量的数据。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>医疗数据中的缺失值通常不是随机发生的。例如，某些检查可能只对特定症状的患者进行，导致数据的缺失与患者的健康状况相关。这种非随机缺失模式需要特殊处理，以避免引入偏差。</p>
  </div>
</div>

## 数据集介绍

本项目使用的数据集包含5,000名患者的医疗记录，包括以下字段：

- **patient_id**：患者ID
- **age**：年龄
- **gender**：性别
- **bmi**：体重指数
- **blood_pressure_systolic**：收缩压
- **blood_pressure_diastolic**：舒张压
- **heart_rate**：心率
- **cholesterol**：胆固醇水平
- **glucose**：血糖水平
- **smoking**：吸烟状态
- **alcohol_consumption**：酒精消费水平
- **physical_activity**：身体活动水平
- **family_history**：家族病史
- **medication**：当前用药情况
- **diagnosis**：诊断结果

数据集中存在不同类型和比例的缺失值，需要采用多种方法进行处理和比较。

## 项目目标

1. 分析数据集中缺失值的模式和特征
2. 实现并比较多种缺失值处理方法
3. 评估不同缺失值处理方法对后续分析的影响
4. 选择最佳的缺失值处理策略
5. 准备完整的数据集用于疾病预测模型

## 实施步骤

### 步骤1：数据加载与缺失值分析

首先，我们加载数据并分析缺失值的模式和特征。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('medical_data.csv')

# 查看数据基本信息
print(df.info())
print(df.describe())

# 分析缺失值
missing = df.isnull().sum()
missing_percent = missing / len(df) * 100
missing_df = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percent', ascending=False)
print(missing_df)

# 可视化缺失值模式
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title('缺失值矩阵')
plt.show()

plt.figure(figsize=(12, 6))
msno.heatmap(df)
plt.title('缺失值相关性热图')
plt.show()

# 分析缺失值与目标变量的关系
# 创建缺失标志列
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[f'{col}_missing'] = df[col].isnull().astype(int)

# 分析缺失标志与诊断结果的关系
for col in [c for c in df.columns if c.endswith('_missing')]:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='diagnosis', data=df)
    plt.title(f'{col} vs Diagnosis')
    plt.show()
```

### 步骤2：准备数据用于缺失值处理比较

为了比较不同缺失值处理方法的效果，我们需要准备一个完整的子集作为参考。

```python
# 选择完整记录的子集作为参考
complete_cols = ['age', 'gender', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                'heart_rate', 'cholesterol', 'glucose']
complete_subset = df.dropna(subset=complete_cols).copy()

# 从完整子集中随机引入缺失值，用于方法比较
np.random.seed(42)
df_test = complete_subset.copy()
mask = np.random.rand(*df_test[complete_cols].shape) < 0.2  # 20%的缺失率
df_test.loc[:, complete_cols] = df_test[complete_cols].mask(mask)

# 保存原始完整值用于评估
true_values = complete_subset[complete_cols].copy()
```

### 步骤3：实现并比较不同的缺失值处理方法

接下来，我们实现并比较多种缺失值处理方法。

```python
# 方法1：简单填充（均值/中位数/众数）
def simple_imputation(df, numeric_cols, categorical_cols):
    df_imputed = df.copy()
    
    # 数值型特征使用中位数填充
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # 分类特征使用众数填充
    if categorical_cols:
        for col in categorical_cols:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
    
    return df_imputed

# 方法2：KNN填充
def knn_imputation(df, cols, n_neighbors=5):
    df_imputed = df.copy()
    
    # 标准化数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
    
    # KNN填充
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed[cols] = imputer.fit_transform(df_scaled)
    
    # 反标准化
    df_imputed[cols] = scaler.inverse_transform(df_imputed[cols])
    
    return df_imputed

# 方法3：多重插补（使用IterativeImputer）
def iterative_imputation(df, cols, max_iter=10, random_state=0):
    df_imputed = df.copy()
    
    # 使用随机森林作为估计器
    estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state)
    
    df_imputed[cols] = imputer.fit_transform(df[cols])
    
    return df_imputed

# 方法4：基于分组的填充
def group_imputation(df, target_cols, group_cols):
    df_imputed = df.copy()
    
    for col in target_cols:
        # 对每个分组计算中位数
        group_medians = df.groupby(group_cols)[col].transform('median')
        # 使用分组中位数填充
        df_imputed[col] = df_imputed[col].fillna(group_medians)
        # 如果仍有缺失值（可能是整个组都缺失），使用全局中位数填充
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    
    return df_imputed

# 应用不同的填充方法
numeric_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
               'heart_rate', 'cholesterol', 'glucose']
categorical_cols = ['gender', 'smoking', 'alcohol_consumption', 'physical_activity']

# 应用各种方法
df_simple = simple_imputation(df_test, numeric_cols, categorical_cols)
df_knn = knn_imputation(df_test, numeric_cols)
df_iterative = iterative_imputation(df_test, numeric_cols)
df_group = group_imputation(df_test, numeric_cols, ['gender', 'age'])

# 评估不同方法的性能
def evaluate_imputation(imputed_df, true_df, cols):
    results = {}
    for col in cols:
        # 只考虑原本缺失的值
        mask = imputed_df[col].notnull() & df_test[col].isnull()
        if mask.sum() > 0:
            mse = mean_squared_error(true_df.loc[mask, col], imputed_df.loc[mask, col])
            results[col] = mse
    return results

# 评估各种方法
simple_results = evaluate_imputation(df_simple, true_values, numeric_cols)
knn_results = evaluate_imputation(df_knn, true_values, numeric_cols)
iterative_results = evaluate_imputation(df_iterative, true_values, numeric_cols)
group_results = evaluate_imputation(df_group, true_values, numeric_cols)

# 比较结果
results_df = pd.DataFrame({
    'Simple': simple_results,
    'KNN': knn_results,
    'Iterative': iterative_results,
    'Group': group_results
})

print(results_df)

# 可视化比较
results_df.plot(kind='bar', figsize=(12, 6))
plt.title('不同填充方法的MSE比较')
plt.ylabel('均方误差 (MSE)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 步骤4：选择最佳方法并处理完整数据集

根据比较结果，我们选择最佳的缺失值处理方法，并应用于完整数据集。

```python
# 假设迭代填充方法表现最好
best_method = 'Iterative'
print(f"选择 {best_method} 作为最佳填充方法")

# 处理完整数据集
# 首先处理数值型特征
df_complete = iterative_imputation(df, numeric_cols)

# 然后处理分类特征
for col in categorical_cols:
    if df_complete[col].isnull().sum() > 0:
        df_complete[col] = df_complete[col].fillna(df_complete[col].mode()[0])

# 检查处理后的缺失值情况
print("处理后的缺失值情况:")
print(df_complete.isnull().sum())

# 保存处理后的数据集
df_complete.to_csv('medical_data_complete.csv', index=False)
```

### 步骤5：评估缺失值处理对后续分析的影响

最后，我们评估缺失值处理对疾病预测模型的影响。

```python
# 准备用于预测的特征和目标变量
X = df_complete.drop(['patient_id', 'diagnosis'] + 
                    [c for c in df_complete.columns if c.endswith('_missing')], axis=1)
y = df_complete['diagnosis']

# 将分类特征转换为数值
X = pd.get_dummies(X, drop_first=True)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测和评估
y_pred = clf.predict(X_test)
print("分类报告:")
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('特征重要性')
plt.tight_layout()
plt.show()
```

## 结果分析

通过比较不同的缺失值处理方法，我们得出以下结论：

1. **迭代填充方法**在大多数特征上表现最好，特别是对于相关性高的特征
2. **KNN填充**在某些特征上表现良好，但计算成本较高
3. **分组填充**对于与分组变量强相关的特征效果好
4. **简单填充**虽然简单，但在大多数情况下精度较低

最终选择的迭代填充方法成功处理了数据集中的缺失值，为后续的疾病预测模型提供了高质量的输入数据。预测模型在测试集上取得了良好的性能，表明我们的缺失值处理策略是有效的。

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **缺失机制分析**：深入研究数据的缺失机制（MCAR、MAR、MNAR）
2. **敏感性分析**：评估不同缺失值处理方法对最终模型结果的敏感性
3. **多重插补进阶**：实现完整的多重插补流程，包括创建多个填充数据集和合并结果
4. **自定义填充模型**：为特定特征开发定制的预测模型用于填充
5. **缺失值模拟**：设计实验，通过在完整数据上模拟不同模式的缺失值，评估各种方法的鲁棒性

## 小结与反思

通过这个项目，我们学习了如何处理医疗数据中的缺失值，并比较了不同方法的效果。缺失值处理是医疗数据分析中的关键步骤，直接影响后续分析和预测的准确性。

在实际应用中，这类缺失值处理技术可以帮助医疗机构更好地利用不完整的患者数据，提高疾病预测和诊断的准确性。例如，通过适当的缺失值处理，即使某些检查结果缺失，也能为患者提供相对准确的风险评估。

### 思考问题

1. 在医疗数据中，缺失值本身可能包含信息（例如，医生没有要求某项检查）。如何在填充缺失值的同时保留这种信息？
2. 不同类型的医疗数据（如实验室检查、问卷调查、影像数据）可能需要不同的缺失值处理策略。如何为不同类型的数据选择合适的方法？
3. 在处理敏感的医疗数据时，如何平衡数据完整性和隐私保护的需求？

<div class="practice-link">
  <a href="/projects/classification/titanic.html" class="button">下一个模块：分类算法项目</a>
</div> 