# House Price Prediction Model

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Intermediate</li>
      <li><strong>Type</strong>: Regression - Intermediate</li>
      <!-- <li><strong>Estimated Time</strong>: 4-6 hours</li> -->
      <li><strong>Skills</strong>: Feature Engineering, Regression Models, Model Evaluation</li>
      <li><strong>Relevant Knowledge Module</strong>: <a href="/en/core/regression/linear-regression.html">Regression Analysis</a></li>
    </ul>
  </div>
</div>

## Project Background

House price prediction is a classic problem in machine learning and is important for home buyers, sellers, and investors. By analyzing house features (such as area, location, number of rooms, etc.), we can build a model to predict the market value of a house.

In this project, we will use a housing dataset from the suburbs of Boston to build a house price prediction model and explore the key factors influencing house prices.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Many real estate websites like Zillow use machine learning algorithms to provide automatic valuation services. Zillow’s "Zestimate" uses machine learning models to estimate the price of over 100 million homes in the U.S., with an average error rate of about 2-3%.</p>
  </div>
</div>

## Dataset Introduction

We will use the Boston housing dataset, which contains information about 506 neighborhoods in the Boston suburbs. Each sample has 13 features:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for large-scale properties (over 25,000 sq. ft.)
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if near the river, 0 otherwise)
- NOX: Nitrogen oxide concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built before 1940
- DIS: Weighted distance to Boston's five employment centers
- RAD: Accessibility to radial highways
- TAX: Full property tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2, where Bk is the proportion of Black residents in a town
- LSTAT: Percentage of the population considered lower status

The target variable is MEDV, which is the median value of owner-occupied homes (in thousands of dollars).

## Project Goals

1. Build a regression model that can accurately predict house prices
2. Identify key features influencing house prices
3. Compare the performance of different regression algorithms
4. Evaluate the model’s performance under different evaluation metrics

## Implementation Steps

### 1. Data Exploration and Visualization

First, we need to understand the basic features and distribution of the data:

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

# Load data
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# View basic information
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['MEDV'], kde=True)
plt.title('House Price Distribution')
plt.xlabel('Price (in thousands of dollars)')
plt.ylabel('Frequency')
plt.show()

# View correlation between features and target variable
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Scatter plots for important features vs house price
important_features = ['RM', 'LSTAT', 'PTRATIO', 'DIS']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(important_features):
    sns.scatterplot(x=feature, y='MEDV', data=df, ax=axes[i])
    axes[i].set_title(f'{feature} vs House Price')

plt.tight_layout()
plt.show()
```

### 2. Data Preprocessing

Next, we prepare the data for model training:

```python
# Separate features and target variable
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Model Training and Evaluation

We will try several regression models and compare their performance:

```python
# Define evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.2f}")
    print(f"Test R²: {test_r2:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    return model, y_test_pred

# Linear Regression
print("Linear Regression Model:")
lr_model, lr_pred = evaluate_model(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test)
print("\n")

# Ridge Regression
print("Ridge Regression Model:")
ridge_model, ridge_pred = evaluate_model(Ridge(alpha=1.0), X_train_scaled, X_test_scaled, y_train, y_test)
print("\n")

# Lasso Regression
print("Lasso Regression Model:")
lasso_model, lasso_pred = evaluate_model(Lasso(alpha=0.1), X_train_scaled, X_test_scaled, y_train, y_test)
print("\n")

# Decision Tree Regression
print("Decision Tree Regression Model:")
dt_model, dt_pred = evaluate_model(DecisionTreeRegressor(max_depth=5), X_train, X_test, y_train, y_test)
print("\n")

# Random Forest Regression
print("Random Forest Regression Model:")
rf_model, rf_pred = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=10), X_train, X_test, y_train, y_test)
```

### 4. Feature Importance Analysis

Understand which features have the most impact on house prices:

```python
# Use Random Forest model for feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

### 5. Prediction Visualization

Visualize the model’s predicted values against the actual values:

```python
# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Random Forest Model: Actual vs Predicted House Price')
plt.tight_layout()
plt.show()

# Residual analysis
residuals = y_test - rf_pred
plt.figure(figsize=(10, 6))
plt.scatter(rf_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted House Price')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.tight_layout()
plt.show()
```

## Advanced Challenges

If you have completed the basic tasks, try the following advanced challenges:

1. **Feature Engineering**: Create new features, such as room-to-area ratios or interaction features.
2. **Hyperparameter Tuning**: Optimize model parameters using grid search or random search.
3. **Ensemble Methods**: Try stacking or voting ensemble methods to improve prediction performance.
4. **Non-linear Transformations**: Apply log or polynomial transformations to features or target variables.
5. **Cross-validation**: Implement k-fold cross-validation for more robust model evaluation.

## Summary and Reflection

Through this project, we learned how to build a house price prediction model, from data exploration to model evaluation. We found that factors like the number of rooms, the percentage of lower status population, and the pupil-teacher ratio significantly impact house prices.

In real-world applications, house price prediction models can help buyers assess reasonable prices, sellers set appropriate pricing strategies, and developers identify promising areas.

### Reflection Questions

1. Besides the features we used, what other factors might influence house prices? How could you obtain this data?
2. Which types of houses did our model perform poorly on? Why?
3. How could you apply this model to other cities? What factors would you need to consider?

<div class="practice-link">
  <a href="/en/projects/regression/sales-forecast.html" class="button">Next Project: Sales Forecasting</a>
</div>
