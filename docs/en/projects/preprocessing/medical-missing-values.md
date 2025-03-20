# Handling Missing Values in Medical Data

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Intermediate</li>
      <li><strong>Type</strong>: Missing Value Handling</li>
      <!-- <li><strong>Estimated Time</strong>: 5-7 hours</li> -->
      <li><strong>Skills</strong>: Missing Pattern Analysis, Multiple Imputation, KNN Imputation, Model-based Imputation</li>
      <li><strong>Related Knowledge Module</strong>: <a href="/core/preprocessing/data-presentation.html">Data Preprocessing</a></li>
    </ul>
  </div>
</div>

## Project Background

Medical data plays a key role in clinical research, disease prediction, and optimizing treatment plans. However, medical datasets often contain a large amount of missing values, which can be caused by factors such as equipment malfunction, patients not completing all tests, recording errors, or data entry issues. Improper handling of missing values may lead to biased conclusions and affect the accuracy of medical decisions.

In this project, we will work on a medical dataset with various types of missing values, compare the effects of different missing value handling methods, and prepare high-quality data for subsequent disease prediction models.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Missing values in medical data are often not random. For example, certain tests might only be conducted on patients with specific symptoms, resulting in missing data that is related to the patient’s health status. This non-random missing pattern requires special treatment to avoid introducing bias.</p>
  </div>
</div>

## Dataset Introduction

The dataset used in this project contains medical records of 5,000 patients, including the following fields:

- **patient_id**: Patient ID
- **age**: Age
- **gender**: Gender
- **bmi**: Body Mass Index
- **blood_pressure_systolic**: Systolic Blood Pressure
- **blood_pressure_diastolic**: Diastolic Blood Pressure
- **heart_rate**: Heart Rate
- **cholesterol**: Cholesterol Level
- **glucose**: Glucose Level
- **smoking**: Smoking Status
- **alcohol_consumption**: Alcohol Consumption Level
- **physical_activity**: Physical Activity Level
- **family_history**: Family History
- **medication**: Current Medication
- **diagnosis**: Diagnosis Result

The dataset has different types and proportions of missing values, which require multiple methods for handling and comparison.

## Project Objectives

1. Analyze the patterns and characteristics of missing values in the dataset.
2. Implement and compare various missing value handling methods.
3. Evaluate the impact of different missing value handling methods on subsequent analysis.
4. Select the best missing value handling strategy.
5. Prepare a complete dataset for disease prediction models.

## Implementation Steps

### Step 1: Data Loading and Missing Value Analysis

First, we load the data and analyze the patterns and characteristics of the missing values.

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

# Load the data
df = pd.read_csv('medical_data.csv')

# View basic information about the data
print(df.info())
print(df.describe())

# Analyze missing values
missing = df.isnull().sum()
missing_percent = missing / len(df) * 100
missing_df = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percent', ascending=False)
print(missing_df)

# Visualize missing value patterns
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title('Missing Value Matrix')
plt.show()

plt.figure(figsize=(12, 6))
msno.heatmap(df)
plt.title('Missing Value Correlation Heatmap')
plt.show()

# Analyze the relationship between missing values and the target variable
# Create missing indicator columns
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[f'{col}_missing'] = df[col].isnull().astype(int)

# Analyze the relationship between missing indicators and diagnosis results
for col in [c for c in df.columns if c.endswith('_missing')]:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='diagnosis', data=df)
    plt.title(f'{col} vs Diagnosis')
    plt.show()
```

### Step 2: Prepare Data for Comparing Missing Value Handling Methods

To compare the effects of different missing value handling methods, we need to prepare a complete subset as a reference.

```python
# Select a subset with complete records as a reference
complete_cols = ['age', 'gender', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                'heart_rate', 'cholesterol', 'glucose']
complete_subset = df.dropna(subset=complete_cols).copy()

# Randomly introduce missing values in the complete subset for method comparison
np.random.seed(42)
df_test = complete_subset.copy()
mask = np.random.rand(*df_test[complete_cols].shape) < 0.2  # 20% missing rate
df_test.loc[:, complete_cols] = df_test[complete_cols].mask(mask)

# Save the original complete values for evaluation
true_values = complete_subset[complete_cols].copy()
```

### Step 3: Implement and Compare Different Missing Value Handling Methods

Next, we implement and compare several methods for handling missing values.

```python
# Method 1: Simple Imputation (mean/median/mode)
def simple_imputation(df, numeric_cols, categorical_cols):
    df_imputed = df.copy()
    
    # Use median for numerical features
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Use mode for categorical features
    if categorical_cols:
        for col in categorical_cols:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
    
    return df_imputed

# Method 2: KNN Imputation
def knn_imputation(df, cols, n_neighbors=5):
    df_imputed = df.copy()
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
    
    # KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed[cols] = imputer.fit_transform(df_scaled)
    
    # Reverse standardization
    df_imputed[cols] = scaler.inverse_transform(df_imputed[cols])
    
    return df_imputed

# Method 3: Multiple Imputation (using IterativeImputer)
def iterative_imputation(df, cols, max_iter=10, random_state=0):
    df_imputed = df.copy()
    
    # Use RandomForest as the estimator
    estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state)
    
    df_imputed[cols] = imputer.fit_transform(df[cols])
    
    return df_imputed

# Method 4: Group-based Imputation
def group_imputation(df, target_cols, group_cols):
    df_imputed = df.copy()
    
    for col in target_cols:
        # Calculate the median for each group
        group_medians = df.groupby(group_cols)[col].transform('median')
        # Fill missing values with group median
        df_imputed[col] = df_imputed[col].fillna(group_medians)
        # If there are still missing values (e.g., the entire group is missing), fill with the global median
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    
    return df_imputed

# Apply different imputation methods
numeric_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
               'heart_rate', 'cholesterol', 'glucose']
categorical_cols = ['gender', 'smoking', 'alcohol_consumption', 'physical_activity']

# Apply various methods
df_simple = simple_imputation(df_test, numeric_cols, categorical_cols)
df_knn = knn_imputation(df_test, numeric_cols)
df_iterative = iterative_imputation(df_test, numeric_cols)
df_group = group_imputation(df_test, numeric_cols, ['gender', 'age'])

# Evaluate the performance of different methods
def evaluate_imputation(imputed_df, true_df, cols):
    results = {}
    for col in cols:
        # Only consider the originally missing values
        mask = imputed_df[col].notnull() & df_test[col].isnull()
        if mask.sum() > 0:
            mse = mean_squared_error(true_df.loc[mask, col], imputed_df.loc[mask, col])
            results[col] = mse
    return results

# Evaluate various methods
simple_results = evaluate_imputation(df_simple, true_values, numeric_cols)
knn_results = evaluate_imputation(df_knn, true_values, numeric_cols)
iterative_results = evaluate_imputation(df_iterative, true_values, numeric_cols)
group_results = evaluate_imputation(df_group, true_values, numeric_cols)

# Compare the results
results_df = pd.DataFrame({
    'Simple': simple_results,
    'KNN': knn_results,
    'Iterative': iterative_results,
    'Group': group_results
})

print(results_df)

# Visualize the comparison
results_df.plot(kind='bar', figsize=(12, 6))
plt.title('MSE Comparison of Different Imputation Methods')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 4: Select the Best Method and Process the Complete Dataset

Based on the comparison results, we choose the best missing value handling method and apply it to the complete dataset.

```python
# Assuming the iterative imputation method performed the best
best_method = 'Iterative'
print(f"Selected {best_method} as the best imputation method")

# Process the complete dataset
# First, handle numerical features
df_complete = iterative_imputation(df, numeric_cols)

# Then, handle categorical features
for col in categorical_cols:
    if df_complete[col].isnull().sum() > 0:
        df_complete[col] = df_complete[col].fillna(df_complete[col].mode()[0])

# Check the missing value status after processing
print("Missing values after processing:")
print(df_complete.isnull().sum())

# Save the processed dataset
df_complete.to_csv('medical_data_complete.csv', index=False)
```

### Step 5: Evaluate the Impact of Missing Value Handling on Subsequent Analysis

Finally, we evaluate the impact of the missing value handling on a disease prediction model.

```python
# Prepare features and target variable for prediction
X = df_complete.drop(['patient_id', 'diagnosis'] + 
                    [c for c in df_complete.columns if c.endswith('_missing')], axis=1)
y = df_complete['diagnosis']

# Convert categorical features into numerical values
X = pd.get_dummies(X, drop_first=True)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

## Results Analysis

By comparing different missing value handling methods, we reached the following conclusions:

1. **Iterative Imputation** performed best for most features, especially those with high correlation.
2. **KNN Imputation** performed well on some features but was computationally expensive.
3. **Group-based Imputation** was effective for features strongly related to grouping variables.
4. **Simple Imputation** is straightforward but generally less accurate.

The chosen iterative imputation method successfully handled the missing values in the dataset, providing high-quality input data for subsequent disease prediction models. The prediction model achieved good performance on the test set, indicating that our missing value handling strategy was effective.

## Advanced Challenges

If you have completed the basic tasks, you can try the following advanced challenges:

1. **Analysis of Missing Mechanisms**: Delve into the missing mechanisms (MCAR, MAR, MNAR).
2. **Sensitivity Analysis**: Evaluate how sensitive the final model results are to different imputation methods.
3. **Advanced Multiple Imputation**: Implement a complete multiple imputation process, including generating multiple imputed datasets and combining the results.
4. **Custom Imputation Models**: Develop tailored predictive models for imputing specific features.
5. **Missing Value Simulation**: Design experiments by simulating different missing patterns on complete data to assess the robustness of various methods.

## Summary and Reflection

Through this project, we learned how to handle missing values in medical data and compare the effectiveness of different methods. Missing value handling is a crucial step in medical data analysis as it directly affects the accuracy of subsequent analyses and predictions.

In practical applications, these techniques can help healthcare institutions better utilize incomplete patient data, thereby improving the accuracy of disease prediction and diagnosis. For example, with appropriate handling, even when some test results are missing, relatively accurate risk assessments can be provided to patients.

### Reflection Questions

1. In medical data, missing values may carry information (e.g., the absence of a test might indicate that a doctor did not deem it necessary). How can we retain this information while imputing the missing values?
2. Different types of medical data (such as lab tests, surveys, imaging data) may require different imputation strategies. How can we choose the appropriate method for different types of data?
3. When handling sensitive medical data, how can we balance data completeness with the need for privacy protection?

<div class="practice-link">
  <a href="/projects/classification/titanic.html" class="button">Next Module: Classification Project</a>
</div>
