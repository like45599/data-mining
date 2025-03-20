# Missing Value Handling

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the causes and impacts of missing values</li>
      <li>Master methods for detecting and analyzing missing values</li>
      <li>Learn various strategies for handling missing values and their appropriate scenarios</li>
      <li>Practice common techniques for missing value processing</li>
    </ul>
  </div>
</div>

## Overview of Missing Values

Missing values are a common problem in data analysis and machine learning. Effectively handling missing values is crucial for model performance.

### Causes of Missing Values

Missing values are usually caused by the following reasons:

1. **Data Collection Issues**: For example, unanswered survey questions or sensor failures.
2. **Data Integration Issues**: Incomplete information when merging data from multiple sources.
3. **Data Processing Errors**: Errors during import, transformation, or cleaning processes.
4. **Privacy Protection**: Intentionally hiding sensitive information.
5. **Structural Missingness**: Data not required under certain conditions.

### Types of Missing Values

Based on the missing mechanism, missing values can be classified into three types:

1. **Missing Completely At Random (MCAR)**
   - The missingness occurs completely at random and is unrelated to any observed or unobserved variables.
   - For example: Data loss due to random equipment failure.
2. **Missing At Random (MAR)**
   - The probability of missing depends solely on other observed variables.
   - For example: Older individuals might be less likely to disclose their income.
3. **Missing Not At Random (MNAR)**
   - The probability of missing is related to unobserved variables or the missing value itself.
   - For example: High-income individuals might choose not to reveal their income.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Statisticians Donald Rubin first systematically proposed the classification of missing data mechanisms (MCAR, MAR, MNAR) in 1976. This framework remains the foundation for handling missing values today. Different missing mechanisms require different handling strategies, and selecting the right method is crucial for reliable analysis.</p>
  </div>
</div>

## Analyzing Missing Values

### 1. Detecting Missing Values

Before handling missing values, it is essential to thoroughly understand their extent in the data:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Check the number of missing values per column
missing_values = df.isnull().sum()
print(missing_values)

# Calculate the missing ratio
missing_ratio = df.isnull().sum() / len(df) * 100
print(missing_ratio)

# Visualize missing values
plt.figure(figsize=(12, 6))
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
sns.barplot(x=missing_ratio.index, y=missing_ratio.values)
plt.title('Missing Value Ratio')
plt.xticks(rotation=45)
plt.ylabel('Percentage Missing')
plt.show()
```

  </div>
</div>

### 2. Analyzing Missing Patterns

Understanding the relationships among missing values helps in choosing the right handling strategy:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Missing value heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Distribution Heatmap')
plt.show()

# Correlation of missing values
missing_binary = df.isnull().astype(int)
plt.figure(figsize=(10, 8))
sns.heatmap(missing_binary.corr(), annot=True, cmap='coolwarm')
plt.title('Missing Value Correlation Heatmap')
plt.show()
```

  </div>
</div>

<div class="visualization-container">
  <div class="visualization-title">Visualization of Missing Patterns</div>
  <div class="visualization-content">
    <img src="/images/missing_pattern_en.svg" alt="Visualization of Missing Patterns">
  </div>
  <div class="visualization-caption">
    Figure: Heatmap showing the distribution of missing values. The yellow areas indicate missing values, allowing observation of missing patterns.
  </div>
</div>

## Strategies for Handling Missing Values

### 1. Deleting Data with Missing Values

The simplest method is to delete rows or columns with missing values, though it may lead to information loss:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Delete all rows with missing values
df_dropped_rows = df.dropna()

# Delete columns with more than 50% missing values
threshold = len(df) * 0.5
df_dropped_cols = df.dropna(axis=1, thresh=threshold)

# Delete rows with missing values in specific columns
df_dropped_specific = df.dropna(subset=['income', 'age'])
```

  </div>
</div>

**Applicable Scenarios**:
- When the missing ratio is very low (less than 5%).
- When the dataset is large enough so that deletion doesn't significantly reduce the sample size.
- When missing values occur completely at random (MCAR).

**Drawbacks**:
- May introduce bias, especially if missingness is not completely random.
- Reduces the overall sample size, potentially lowering statistical power.
- Might result in the loss of valuable information.

### 2. Imputing Missing Values

#### 2.1 Imputation Using Statistical Measures

Replace missing values with statistics such as mean, median, or mode:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Mean imputation
df_mean = df.copy()
df_mean.fillna(df_mean.mean(), inplace=True)

# Median imputation
df_median = df.copy()
df_median.fillna(df_median.median(), inplace=True)

# Mode imputation for categorical variables
df_mode = df.copy()
for col in df_mode.columns:
    if df_mode[col].dtype == 'object':
        df_mode[col].fillna(df_mode[col].mode()[0], inplace=True)
```

  </div>
</div>

<div class="interactive-component">
  <div class="interactive-title">Comparison of Imputation Methods</div>
  <div class="interactive-content">
    <missing-value-imputation></missing-value-imputation>
  </div>
  <div class="interactive-caption">
    Interactive Component: Experiment with different imputation methods and observe their effects on the data distribution.
  </div>
</div>

#### 2.2 Advanced Imputation Methods

Use data relationships to perform more sophisticated imputations:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN imputation
imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Multiple Imputation using a simplified version of the MICE algorithm
imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

  </div>
</div>

## Evaluating Missing Value Handling

### 1. Comparing Different Methods

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Original data distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['income'].dropna(), kde=True)
plt.title('Original Data Distribution')

# Distribution after mean imputation
plt.subplot(1, 3, 2)
sns.histplot(df_mean['income'], kde=True)
plt.title('Distribution After Mean Imputation')

# Distribution after KNN imputation
plt.subplot(1, 3, 3)
sns.histplot(df_knn['income'], kde=True)
plt.title('Distribution After KNN Imputation')

plt.tight_layout()
plt.show()
```

  </div>
</div>

## Practical Recommendations

### 1. Missing Value Handling Workflow

1. **Exploratory Analysis**:
   - Detect the locations, counts, and ratios of missing values.
   - Analyze missing patterns and mechanisms.
   - Visualize the distribution of missing values.
2. **Develop a Handling Strategy**:
   - Choose an appropriate method based on the missing mechanism.
   - Consider feature importance and data structure.
   - Use different strategies for different features if needed.
3. **Implementation and Evaluation**:
   - Apply the selected imputation method.
   - Compare the effects of various methods.
   - Validate the quality and consistency of the processed data.

### 2. Practical Tips

- **Analyze Before Processing**: Understand the underlying reasons for missing values before choosing a method.
- **Consider Feature Correlations**: Utilize the relationships between features to enhance imputation.
- **Incorporate Domain Knowledge**: Let business or domain insights guide your missing value handling.
- **Perform Sensitivity Analysis**: Test how different imputation methods affect your final results.
- **Preserve Uncertainty**: Consider methods like multiple imputation to capture the uncertainty in estimates.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>Common Misconceptions
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Blind Deletion</strong>: Removing samples with missing values without analyzing the underlying mechanisms.</li>
      <li><strong>Overreliance on Mean Imputation</strong>: Using the mean for all features without considering their distribution.</li>
      <li><strong>Ignoring Missing Correlations</strong>: Failing to consider relationships between features during imputation.</li>
      <li><strong>Data Leakage</strong>: Using information from the test set to impute the training set.</li>
    </ul>
  </div>
</div>

## Summary and Reflection

Handling missing values is a critical step in data preprocessing. The right method can enhance model performance and ensure reliable analysis results.

### Key Takeaways

- Missing values can arise from various causes, including data collection issues, privacy protection, and more.
- Missing mechanisms are classified into MCAR, MAR, and MNAR.
- Handling strategies include both deletion and various imputation methods.
- The choice of method should consider the missing mechanism, data characteristics, and analysis objectives.

### Reflection Questions

1. How can we determine the mechanism behind missing values in a dataset?
2. Under what circumstances is it more appropriate to delete samples with missing values rather than impute them?
3. How can we evaluate the effectiveness of different missing value handling methods?

<div class="practice-link">
  <a href="/projects/preprocessing.html" class="button">Proceed to Practice Projects</a>
</div>

