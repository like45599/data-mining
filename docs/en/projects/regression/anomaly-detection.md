# Anomaly Detection and Prediction

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Advanced</li>
      <li><strong>Type</strong>: Regression - Advanced</li>
      <!-- <li><strong>Estimated Time</strong>: 5-7 hours</li> -->
      <li><strong>Skills</strong>: Anomaly detection, time series analysis, predictive modeling</li>
      <li><strong>Related Knowledge Module</strong>: <a href="/core/regression/linear-regression.html">Regression Analysis</a></li>
    </ul>
  </div>
</div>

## Project Background

Anomaly detection is an important task in data mining, used to identify unusual patterns or outliers in data. Timely detection of anomalies is crucial in many fields such as cybersecurity, financial fraud detection, and equipment failure prediction.

In this project, we will combine anomaly detection and predictive techniques to build a system capable of detecting anomalies and predicting future trends. We will use sensor data to detect abnormal operating conditions of equipment and predict potential failures.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span> Did you know?
  </div>
  <div class="knowledge-card__content">
    <p>Predictive maintenance technologies can help businesses reduce unplanned downtime by up to 50% and extend equipment life by 20-40%. According to McKinsey, IoT-based predictive maintenance could generate $630 billion in annual value for factories.</p>
  </div>
</div>

## Dataset Introduction

We will use a simulated industrial sensor dataset containing operational data from a device over the course of one year. The dataset includes the following features:

- timestamp: Timestamp
- sensor1 ~ sensor10: Readings from 10 different sensors (e.g., temperature, pressure, vibration)
- operational_setting1 ~ operational_setting3: 3 operational setting parameters
- environment1 ~ environment2: 2 environmental parameters (e.g., ambient temperature, humidity)
- failure: Whether a failure occurred (0 for normal, 1 for failure)

The dataset contains normal operation data and a small amount of failure data. Our task is to:
1. Detect anomalies in the sensor data
2. Predict the potential failure occurrence time

## Project Goals

1. Build an anomaly detection model to identify abnormal patterns in the sensor data
2. Analyze the relationship between anomalies and equipment failures
3. Build a predictive model to forecast the likelihood of equipment failure
4. Evaluate the performance of different anomaly detection and prediction methods
5. Provide visualizations and business recommendations to help develop preventive maintenance strategies

## Implementation Steps

### 1. Data Exploration and Visualization

First, we need to understand the basic features and distributions of the data:

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

# Load the data
df = pd.read_csv('sensor_data.csv')

# Convert timestamp to datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# View basic data information
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# View failure distribution
print(df['failure'].value_counts())
print(f"Failure rate: {df['failure'].mean() * 100:.2f}%")

# Visualize sensor data time series
plt.figure(figsize=(15, 10))
for i in range(1, 5):  # Only show data from the first 4 sensors
    plt.subplot(4, 1, i)
    plt.plot(df['timestamp'], df[f'sensor{i}'])
    plt.title(f'Sensor {i} Readings')
    plt.grid(True)
plt.tight_layout()
plt.show()

# View sensor data before and after failure
failure_times = df[df['failure'] == 1]['timestamp'].tolist()
for failure_time in failure_times[:2]:  # Only view the first two failures
    # Get the data 24 hours before the failure
    start_time = failure_time - pd.Timedelta(hours=24)
    end_time = failure_time + pd.Timedelta(hours=2)
    failure_window = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    plt.figure(figsize=(15, 10))
    for i in range(1, 5):  # Only show data from the first 4 sensors
        plt.subplot(4, 1, i)
        plt.plot(failure_window['timestamp'], failure_window[f'sensor{i}'])
        plt.axvline(x=failure_time, color='r', linestyle='--', label='Failure')
        plt.title(f'Sensor {i} Readings Before and After Failure')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# View correlation between sensors
sensor_cols = [f'sensor{i}' for i in range(1, 11)]
plt.figure(figsize=(12, 10))
sns.heatmap(df[sensor_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Sensors')
plt.show()
```

### 2. Feature Engineering

Create new features that assist with anomaly detection and failure prediction:

```python
# Create rolling statistics features
window_sizes = [10, 30, 60]  # Different window sizes (minutes)

for sensor in sensor_cols:
    for window in window_sizes:
        # Rolling mean
        df[f'{sensor}_mean_{window}m'] = df[sensor].rolling(window=window).mean()
        # Rolling standard deviation
        df[f'{sensor}_std_{window}m'] = df[sensor].rolling(window=window).std()
        # Difference between rolling max and min
        df[f'{sensor}_range_{window}m'] = df[sensor].rolling(window=window).max() - df[sensor].rolling(window=window).min()

# Create sensor ratio features
for i in range(1, 10):
    for j in range(i+1, 11):
        df[f'ratio_sensor{i}_sensor{j}'] = df[f'sensor{i}'] / df[f'sensor{j}']

# Create time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Create failure lead labels
# Mark data for 12 hours before failure
lead_hours = 12
df['failure_lead'] = 0

for failure_time in failure_times:
    lead_start = failure_time - pd.Timedelta(hours=lead_hours)
    df.loc[(df['timestamp'] >= lead_start) & (df['timestamp'] < failure_time), 'failure_lead'] = 1

# Remove rows with NaN values (due to rolling feature creation)
df = df.dropna()

# View new features
print(df.columns.tolist())
```

### 3. Anomaly Detection

Use the Isolation Forest algorithm to detect anomalies:

```python
# Select features for anomaly detection
anomaly_features = sensor_cols + [f'sensor{i}_mean_60m' for i in range(1, 11)] + [f'sensor{i}_std_60m' for i in range(1, 11)]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[anomaly_features])

# Use Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = isolation_forest.fit_predict(scaled_features)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # Convert -1 to 1 for anomalies

# View anomaly detection results
print(f"Anomaly detection rate: {df['anomaly'].mean() * 100:.2f}%")

# Visualize anomalies
plt.figure(figsize=(15, 8))
for i in range(1, 3):  # Only show data from the first 2 sensors
    plt.subplot(2, 1, i)
    plt.scatter(df['timestamp'], df[f'sensor{i}'], c=df['anomaly'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Anomaly')
    plt.title(f'Sensor {i} Readings with Anomalies')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze the relationship between anomalies and failures
anomaly_failure_crosstab = pd.crosstab(df['anomaly'], df['failure'])
print("Relationship between anomalies and failures:")
print(anomaly_failure_crosstab)

# Evaluate anomaly detection performance (with failures as the true label)
print("\nAnomaly detection performance (with failures as the true label):")
print(classification_report(df['failure'], df['anomaly']))
```

### 4. Failure Prediction Model

Build a model to predict future potential failures:

```python
# Select features for failure prediction
failure_features = sensor_cols + \
                  [f'sensor{i}_mean_60m' for i in range(1, 11)] + \
                  [f'sensor{i}_std_60m' for i in range(1, 11)] + \
                  [f'sensor{i}_range_60m' for i in range(1, 11)] + \
                  ['anomaly'] + \
                  [col for col in df.columns if 'ratio_sensor' in col] + \
                  ['hour', 'day_of_week', 'is_weekend']

# Prepare features and target variable (predict failure lead)
X = df[failure_features]
y = df['failure_lead']

# Split into training and testing sets (time-based split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Training set performance:")
print(classification_report(y_train, y_train_pred))
print("\nTest set performance:")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
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

### 5. Feature Importance Analysis

Understand which features are most important for failure prediction:

```python
# Analyze feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Display the top 20 most important features
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Feature Importance for Failure Prediction')
plt.tight_layout()
plt.show()
```

### 6. Prediction Visualization and Early Warning System

Visualize the prediction results and design an early warning system:

```python
# Add prediction probabilities to test data
test_data = df[train_size:].copy()
test_data['failure_prob'] = y_test_prob

# Visualize prediction probabilities with actual failures
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

# Design early warning system
warning_levels = [
    (0.7, 'High Risk - Immediate Maintenance Required'),
    (0.5, 'Medium Risk - Schedule Maintenance Soon'),
    (0.3, 'Low Risk - Monitor Closely')
]

# Add warning level labels
test_data['warning_level'] = 'Normal'
for threshold, message in warning_levels:
    test_data.loc[test_data['failure_prob'] >= threshold, 'warning_level'] = message

# Display warning level distribution
print(test_data['warning_level'].value_counts())

# Calculate early warning lead time
warnings = test_data[test_data['warning_level'] != 'Normal'].copy()
failures = test_data[test_data['failure'] == 1].copy()

if not failures.empty and not warnings.empty:
    warning_times = []
    for failure_time in failures['timestamp']:
        # Find the earliest warning before the failure
        prior_warnings = warnings[warnings['timestamp'] < failure_time]
        if not prior_warnings.empty:
            earliest_warning = prior_warnings['timestamp'].max()
            warning_lead_time = (failure_time - earliest_warning).total_seconds() / 3600  # In hours
            warning_times.append(warning_lead_time)
    
    if warning_times:
        print(f"Average early warning lead time: {np.mean(warning_times):.2f} hours")
        print(f"Shortest early warning lead time: {np.min(warning_times):.2f} hours")
        print(f"Longest early warning lead time: {np.max(warning_times):.2f} hours")
```

## Advanced Challenges

If you have completed the basic tasks, you can try the following advanced challenges:

1. **Deep Learning Methods**: Try using LSTM or autoencoders for anomaly detection and failure prediction
2. **Multi-step Prediction**: Build models that predict failure probabilities for multiple future time points
3. **Ensemble Anomaly Detection**: Combine multiple anomaly detection algorithms to improve accuracy
4. **Online Learning**: Implement models that continuously learn from new data
5. **Maintenance Cost Optimization**: Consider maintenance costs and downtime losses to optimize maintenance decisions

## Conclusion and Reflection

Through this project, we learned how to combine anomaly detection and predictive techniques to build a predictive maintenance system. We discovered that certain anomaly patterns in sensor data can serve as early warning signals for equipment failures, allowing timely interventions to prevent severe failures.

In practical applications, predictive maintenance systems help businesses reduce unplanned downtime, extend equipment life, and lower maintenance costs. Moreover, by continuously monitoring and analyzing equipment operation data, businesses can further optimize equipment performance and maintenance strategies.

### Reflective Questions

1. How do you handle noise and missing values in sensor data in real-world industrial environments?
2. How do you balance the sensitivity and accuracy of an early warning system? What issues arise from too many false positives or missed warnings?
3. How can a predictive maintenance system be integrated into a company’s maintenance management process to achieve maximum value?

<div class="practice-link">
  <a href="/projects/" class="button">Return to Project List</a>
</div>
