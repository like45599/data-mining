# Data Representation Methods

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Key Takeaways
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the characteristics and representation methods of different data types</li>
      <li>Master the usage scenarios of common data structures</li>
      <li>Learn the methods of data standardization and normalization</li>
      <li>Understand the basic techniques of feature engineering</li>
    </ul>
  </div>
</div>

## Introduction to Data Types

Correct identification and handling of data types are crucial for success in data mining. Here are common data types:

### 1. Numerical Data

Numerical data can be divided into continuous and discrete types:

- **Continuous Numerical**: Can take any real value, such as temperature (25.5°C), height (175.2 cm).
- **Discrete Numerical**: Can only take specific values, usually integers, such as age (18 years), quantity (5 items).

**Processing Suggestions**: Typically, normalization is required to eliminate dimensional effects, and common methods include:

- **Z-Score Normalization**: $z = \frac{x - \mu}{\sigma}$ where $\mu$ is the mean and $\sigma$ is the standard deviation.
- **Min-Max Normalization**: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create sample data
df = pd.DataFrame({'height': [165, 170, 175, 180, 185]})

# Z-Score Normalization
scaler = StandardScaler()
df['height_zscore'] = scaler.fit_transform(df[['height']])

# Min-Max Normalization
min_max_scaler = MinMaxScaler()
df['height_minmax'] = min_max_scaler.fit_transform(df[['height']])

print(df)
```

  </div>
</div>

### 2. Categorical Data

Categorical data represents classification information and is divided into:

- **Nominal Variables**: No inherent order, such as gender (Male/Female), color (Red/Blue/Green).
- **Ordinal Variables**: Have a clear order, such as education level (Primary/Secondary/University), satisfaction (Low/Medium/High).

**Processing Suggestions**: Typically need to be encoded into numerical values using methods like:

- **One-Hot Encoding**: Converts categories into binary vectors, suitable for nominal variables.
- **Label Encoding**: Maps categories to integers, suitable for ordinal variables.
- **Target Encoding**: Replaces categories with the mean of the target variable, suitable for high-cardinality features.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Create sample data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'green'],
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

# One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
color_encoded = encoder.fit_transform(df[['color']])
color_df = pd.DataFrame(
    color_encoded, 
    columns=[f'color_{c}' for c in encoder.categories_[0]]
)

# Label Encoding
label_encoder = LabelEncoder()
df['size_encoded'] = label_encoder.fit_transform(df['size'])

# Merge the results
result = pd.concat([df, color_df], axis=1)
print(result)
```

  </div>
</div>

### 3. Time Series Data

Time series data changes over time and has sequential properties:

- **Timestamps**: Observations at specific time points, such as stock prices, sensor readings.
- **Time Intervals**: Data over a period, such as call duration, activity duration.
- **Periodic Data**: Has repeating patterns, such as seasonal sales, daily temperature changes.

**Processing Suggestions**:

- Extract time features (year, month, day, hour, weekday, etc.)
- Apply sliding window aggregation (mean, max, min, etc.)
- Handle seasonality and trends

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np

# Create time series data
date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['value'] = np.random.randint(0, 100, size=len(date_rng))

# Extract time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Sliding window
df['rolling_mean_3d'] = df['value'].rolling(window=3).mean()

print(df.head())
```

  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span> Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>In the early days of data science, data representation methods were primarily based on relational database models. It was not until the 1970s, when E.F. Codd proposed the relational model, that data representation issues began to be systematically thought through. Modern data representation methods integrate statistics, computer science, and domain knowledge, forming a rich set of data preprocessing techniques.</p>
  </div>
</div>

## Data Structures and Storage

### 1. Tabular Data

The most common form of data, such as spreadsheets and relational database tables:

- **Rows (Records)**: Represent individual entities or instances
- **Columns (Features)**: Represent attributes or features of the entities
- **Cells**: Represent specific attribute values of a particular entity

<div class="code-example">
  <div class="code-example__title">Python Implementation</div>
  <div class="code-example__content">

```python
import pandas as pd

# Create DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Boston', 'Chicago']
}
df = pd.DataFrame(data)
print(df)
```

  </div>
</div>

### 2. Matrices and Tensors

Support advanced computations and algorithm implementations:

- **Matrix**: Two-dimensional arrays, such as image data, distance matrices
- **Tensor**: Multi-dimensional arrays, commonly used in deep learning

<div class="code-example">
  <div class="code-example__title">Python Implementation</div>
  <div class="code-example__content">

```python
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix:\n", matrix)

# Create 3D tensor
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Tensor:\n", tensor)
```

  </div>
</div>

### 3. Graph Structures

Represent relationships between entities:

- **Nodes (Vertices)**: Represent entities
- **Edges**: Represent relationships between entities
- **Weights**: Represent the strength or importance of relationships

<div class="interactive-component">
  <div class="interactive-component__title">Graph Structure Visualization</div>
  <div class="interactive-component__content">
    <graph-visualization></graph-visualization>
  </div>
</div>

<div class="code-example">
  <div class="code-example__title">Python Implementation</div>
  <div class="code-example__content">

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])

# Visualization
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=15, font_weight='bold')
plt.title("Simple Graph Structure")
plt.show()
```

  </div>
</div>

## Feature Representation Techniques

### 1. Feature Scaling

Ensures that features with different scales don't dominate the model:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization: Mean = 0, Std = 1
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

# Normalization: Scale to [0,1] range
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

# Robust Scaling: Less sensitive to outliers
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

  </div>
</div>

<div class="interactive-component">
  <div class="interactive-component__title">Feature Scaling Comparison</div>
  <div class="interactive-component__content">
    <feature-scaling-demo></feature-scaling-demo>
  </div>
</div>

### 2. Feature Encoding

Converts non-numeric features into numerical representations usable by algorithms:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Label Encoding: Suitable for ordinal categorical variables
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-Hot Encoding: Suitable for nominal categorical variables
onehot_encoder = OneHotEncoder()
X_encoded = onehot_encoder.fit_transform(X_categorical)

# Ordinal Encoding: Suitable for ordinal categorical variables, retains order
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
X_ordinal = ordinal_encoder.fit_transform(X_categorical)
```

  </div>
</div>

### 3. Feature Selection

Selecting the most relevant or important features from the original feature set:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Statistical feature selection
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Model-based feature selection
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

  </div>
</div>

## Data Visualization and Exploration

Data visualization is a powerful tool for understanding data distribution and relationships:

### 1. Univariate Analysis

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Numerical distribution
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, len(numeric_cols), i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()
```

  </div>
</div>

### 2. Multivariate Analysis

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Pairplot
sns.pairplot(df[numeric_cols + ['target']], hue='target')
plt.suptitle('Relationships Between Features', y=1.02)
plt.show()
```

  </div>
</div>

<div class="interactive-component">
  <div class="interactive-component__title">Interactive Data Exploration</div>
  <div class="interactive-component__content">
    <data-explorer></data-explorer>
  </div>
</div>

## Practical Tips

### 1. Handling High Cardinality Categorical Features

For categorical features with many unique values:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Frequency encoding
frequency_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(frequency_map)

# Group rare categories
def group_rare_categories(series, threshold=0.05):
    value_counts = series.value_counts(normalize=True)
    # Find categories with a frequency lower than the threshold
    rare_categories = value_counts[value_counts < threshold].index.tolist()
    # Replace rare categories
    return series.replace(rare_categories, 'Other')

df['category_grouped'] = group_rare_categories(df['category'])
```

  </div>
</div>

### 2. Preventing Data Leakage

Avoid data leakage in model training:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Incorrect example: Using all data for normalization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Correct approach: Fit scaler only on training data
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data with training data parameters
```

  </div>
</div>

### 3. Efficient Data Handling Techniques

Optimizing methods for handling large datasets:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Reduce memory usage
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Initial memory usage: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Reduced memory usage: {end_mem:.2f} MB')
    print(f'Memory reduced by: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# Example usage
df_optimized = reduce_mem_usage(df)
```

  </div>
</div>

## Summary and Reflection

Data representation is a fundamental step in data mining. Proper data representation can significantly improve the performance of subsequent models. In practice, it is essential to choose appropriate representation methods based on data characteristics and problem requirements.

### Key Takeaways

- Different data types require different handling methods
- Feature scaling can eliminate dimensional effects
- Categorical features need to be converted to numerical form
- Time series data requires extracting time features
- Data visualization helps understand data distribution and relationships

### Reflection Questions

1. When handling categorical features, how do you choose the appropriate encoding method?
2. How does feature scaling affect machine learning algorithms?
3. How do you handle datasets with both numerical and categorical features?

<div class="practice-link">
  <a href="/projects/preprocessing.html" class="button">Go to Practice Projects</a>
</div>
