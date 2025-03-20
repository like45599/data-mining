# 异常检测与预测

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：高级</li>
      <li><strong>类型</strong>：回归 - 高级</li>
      <!-- <li><strong>预计时间</strong>：5-7小时</li> -->
      <li><strong>技能点</strong>：异常检测、时间序列分析、预测建模</li>
      <li><strong>对应知识模块</strong>：<a href="/core/regression/linear-regression.html">回归分析</a></li>
    </ul>
  </div>
</div>

## 项目背景

异常检测是数据挖掘中的重要任务，用于识别数据中的异常模式或离群点。在许多领域，如网络安全、金融欺诈检测、设备故障预测等，及时发现异常情况至关重要。

在这个项目中，我们将结合异常检测和预测技术，构建一个能够识别异常并预测未来趋势的系统。我们将使用传感器数据，检测设备的异常运行状态，并预测可能的故障。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>预测性维护技术可以帮助企业减少计划外停机时间高达50%，延长设备寿命20-40%。据麦肯锡公司估计，基于物联网的预测性维护可以为工厂创造每年6300亿美元的价值。</p>
  </div>
</div>

## 数据集介绍

我们将使用一个模拟的工业传感器数据集，包含某设备在一年内的运行数据。数据集包含以下特征：

- timestamp：时间戳
- sensor1 ~ sensor10：10个不同传感器的读数（如温度、压力、振动等）
- operational_setting1 ~ operational_setting3：3个操作设置参数
- environment1 ~ environment2：2个环境参数（如环境温度、湿度）
- failure：是否发生故障（0表示正常，1表示故障）

数据集中包含了正常运行数据和少量故障数据，我们的任务是：
1. 检测传感器数据中的异常
2. 预测可能的故障发生时间

## 项目目标

1. 构建异常检测模型，识别传感器数据中的异常模式
2. 分析异常与设备故障的关系
3. 构建预测模型，预测设备故障的可能性
4. 评估不同异常检测和预测方法的性能
5. 提供可视化和业务建议，帮助制定预防性维护策略

## 实施步骤

### 1. 数据探索与可视化

首先，我们需要了解数据的基本特征和分布：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('sensor_data.csv')

# 将时间戳转换为日期时间类型
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 查看数据基本信息
print(df.info())
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 查看故障分布
print(df['failure'].value_counts())
print(f"故障率: {df['failure'].mean() * 100:.2f}%")

# 可视化传感器数据时间序列
plt.figure(figsize=(15, 10))
for i in range(1, 5):  # 只展示前4个传感器数据
    plt.subplot(4, 1, i)
    plt.plot(df['timestamp'], df[f'sensor{i}'])
    plt.title(f'Sensor {i} Readings')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 查看故障前后的传感器数据
failure_times = df[df['failure'] == 1]['timestamp'].tolist()
for failure_time in failure_times[:2]:  # 只查看前两次故障
    # 获取故障前24小时的数据
    start_time = failure_time - pd.Timedelta(hours=24)
    end_time = failure_time + pd.Timedelta(hours=2)
    failure_window = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    plt.figure(figsize=(15, 10))
    for i in range(1, 5):  # 只展示前4个传感器数据
        plt.subplot(4, 1, i)
        plt.plot(failure_window['timestamp'], failure_window[f'sensor{i}'])
        plt.axvline(x=failure_time, color='r', linestyle='--', label='Failure')
        plt.title(f'Sensor {i} Readings Before and After Failure')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# 查看传感器之间的相关性
sensor_cols = [f'sensor{i}' for i in range(1, 11)]
plt.figure(figsize=(12, 10))
sns.heatmap(df[sensor_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Sensors')
plt.show()
```

### 2. 特征工程

创建有助于异常检测和故障预测的新特征：

```python
# 创建滚动统计特征
window_sizes = [10, 30, 60]  # 不同的窗口大小（分钟）

for sensor in sensor_cols:
    for window in window_sizes:
        # 滚动平均
        df[f'{sensor}_mean_{window}m'] = df[sensor].rolling(window=window).mean()
        # 滚动标准差
        df[f'{sensor}_std_{window}m'] = df[sensor].rolling(window=window).std()
        # 滚动最大值与最小值之差
        df[f'{sensor}_range_{window}m'] = df[sensor].rolling(window=window).max() - df[sensor].rolling(window=window).min()

# 创建传感器之间的比率特征
for i in range(1, 10):
    for j in range(i+1, 11):
        df[f'ratio_sensor{i}_sensor{j}'] = df[f'sensor{i}'] / df[f'sensor{j}']

# 创建时间特征
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 创建故障前导标记
# 标记故障前12小时的数据
lead_hours = 12
df['failure_lead'] = 0

for failure_time in failure_times:
    lead_start = failure_time - pd.Timedelta(hours=lead_hours)
    df.loc[(df['timestamp'] >= lead_start) & (df['timestamp'] < failure_time), 'failure_lead'] = 1

# 移除包含NaN的行（由于创建滚动特征）
df = df.dropna()

# 查看新特征
print(df.columns.tolist())
```

### 3. 异常检测

使用隔离森林算法检测异常：

```python
# 选择用于异常检测的特征
anomaly_features = sensor_cols + [f'sensor{i}_mean_60m' for i in range(1, 11)] + [f'sensor{i}_std_60m' for i in range(1, 11)]

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[anomaly_features])

# 使用隔离森林进行异常检测
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = isolation_forest.fit_predict(scaled_features)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 将-1转换为1，表示异常

# 查看异常检测结果
print(f"检测到的异常比例: {df['anomaly'].mean() * 100:.2f}%")

# 可视化异常点
plt.figure(figsize=(15, 8))
for i in range(1, 3):  # 只展示前2个传感器数据
    plt.subplot(2, 1, i)
    plt.scatter(df['timestamp'], df[f'sensor{i}'], c=df['anomaly'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Anomaly')
    plt.title(f'Sensor {i} Readings with Anomalies')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 分析异常与故障的关系
anomaly_failure_crosstab = pd.crosstab(df['anomaly'], df['failure'])
print("异常与故障的关系:")
print(anomaly_failure_crosstab)

# 计算异常检测的性能指标（以故障为真实标签）
print("\n异常检测性能（以故障为真实标签）:")
print(classification_report(df['failure'], df['anomaly']))
```

### 4. 故障预测模型

构建模型预测未来可能的故障：

```python
# 选择用于故障预测的特征
failure_features = sensor_cols + \
                  [f'sensor{i}_mean_60m' for i in range(1, 11)] + \
                  [f'sensor{i}_std_60m' for i in range(1, 11)] + \
                  [f'sensor{i}_range_60m' for i in range(1, 11)] + \
                  ['anomaly'] + \
                  [col for col in df.columns if 'ratio_sensor' in col] + \
                  ['hour', 'day_of_week', 'is_weekend']

# 准备特征和目标变量（预测故障前导）
X = df[failure_features]
y = df['failure_lead']

# 划分训练集和测试集（按时间顺序）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 训练随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 预测
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

# 评估模型
print("训练集性能:")
print(classification_report(y_train, y_train_pred))
print("\n测试集性能:")
print(classification_report(y_test, y_test_pred))

# 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

### 5. 特征重要性分析

了解哪些因素对故障预测最重要：

```python
# 分析特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# 显示前20个最重要的特征
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Feature Importance for Failure Prediction')
plt.tight_layout()
plt.show()
```

### 6. 预测可视化与预警系统

可视化预测结果，并设计预警系统：

```python
# 将预测概率添加到测试数据中
test_data = df[train_size:].copy()
test_data['failure_prob'] = y_test_prob

# 可视化预测概率与实际故障
plt.figure(figsize=(15, 8))
plt.plot(test_data['timestamp'], test_data['failure_prob'], label='Failure Probability')
plt.scatter(test_data[test_data['failure'] == 1]['timestamp'], 
            test_data[test_data['failure'] == 1]['failure_prob'],
            color='red', marker='x', s=100, label='Actual Failure')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
plt.title('Failure Probability Prediction')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# 设计预警系统
warning_levels = [
    (0.7, 'High Risk - Immediate Maintenance Required'),
    (0.5, 'Medium Risk - Schedule Maintenance Soon'),
    (0.3, 'Low Risk - Monitor Closely')
]

# 为测试数据添加预警级别
test_data['warning_level'] = 'Normal'
for threshold, message in warning_levels:
    test_data.loc[test_data['failure_prob'] >= threshold, 'warning_level'] = message

# 显示预警分布
print(test_data['warning_level'].value_counts())

# 计算预警提前时间
warnings = test_data[test_data['warning_level'] != 'Normal'].copy()
failures = test_data[test_data['failure'] == 1].copy()

if not failures.empty and not warnings.empty:
    warning_times = []
    for failure_time in failures['timestamp']:
        # 找到故障前的最早预警
        prior_warnings = warnings[warnings['timestamp'] < failure_time]
        if not prior_warnings.empty:
            earliest_warning = prior_warnings['timestamp'].max()
            warning_lead_time = (failure_time - earliest_warning).total_seconds() / 3600  # 小时
            warning_times.append(warning_lead_time)
    
    if warning_times:
        print(f"平均预警提前时间: {np.mean(warning_times):.2f} 小时")
        print(f"最短预警提前时间: {np.min(warning_times):.2f} 小时")
        print(f"最长预警提前时间: {np.max(warning_times):.2f} 小时")
```

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **深度学习方法**：尝试使用LSTM或自编码器进行异常检测和故障预测
2. **多步预测**：构建能够预测未来多个时间点故障概率的模型
3. **集成异常检测**：结合多种异常检测算法，提高检测准确性
4. **在线学习**：实现能够从新数据中持续学习的模型
5. **维护成本优化**：考虑维护成本和停机损失，优化维护决策

## 小结与反思

通过这个项目，我们学习了如何结合异常检测和预测技术，构建一个预测性维护系统。我们发现某些传感器数据的异常模式可以作为设备故障的早期预警信号，通过及时干预可以避免严重故障的发生。

在实际应用中，预测性维护系统可以帮助企业减少计划外停机时间，延长设备寿命，降低维护成本。此外，通过持续监控和分析设备运行数据，企业可以不断优化设备性能和维护策略。

### 思考问题

1. 在实际工业环境中，如何处理传感器数据的噪声和缺失值问题？
2. 如何平衡预警系统的敏感性和准确性？过多的误报或漏报会带来什么问题？
3. 如何将预测性维护系统与企业的维护管理流程集成，实现最大价值？

<div class="practice-link">
  <a href="/projects/" class="button">返回项目列表</a>
</div> 