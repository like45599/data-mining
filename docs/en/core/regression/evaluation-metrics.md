# Regression Evaluation Metrics

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the calculation methods and meanings of common regression evaluation metrics</li>
      <li>Master the applicable scenarios and limitations of different evaluation metrics</li>
      <li>Learn how to use cross-validation to evaluate regression models</li>
      <li>Understand how to choose the appropriate evaluation metrics for model comparison</li>
    </ul>
  </div>
</div>

## Common Regression Evaluation Metrics

There are various metrics to evaluate the performance of regression models, each with its specific use and interpretation.

### Mean Squared Error (MSE)

Mean squared error is one of the most commonly used regression evaluation metrics, calculated as the average of the squared differences between predicted and actual values:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value
- $n$ is the number of samples

MSE penalizes larger errors more heavily, but its unit is the square of the target variable.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
```

  </div>
</div>

### Root Mean Squared Error (RMSE)

Root mean squared error is the square root of MSE, which makes its unit the same as the target variable, making it easier to interpret:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
```

  </div>
</div>

### Mean Absolute Error (MAE)

Mean absolute error calculates the average of the absolute differences between predicted and actual values:

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

MAE is less sensitive to outliers than MSE and is easier to interpret.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
```

  </div>
</div>

### R-Squared (R²)

R-Squared measures the proportion of the variance in the target variable explained by the model, and the range is typically between 0 and 1 (it can also be negative):

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where:
- $\bar{y}$ is the mean of the actual values

An R² value of 1 indicates perfect fit, while 0 means the model is no better than predicting the mean value.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import r2_score

# Calculate R²
r2 = r2_score(y_true, y_pred)
print(f"R-Squared (R²): {r2:.4f}")
```

  </div>
</div>

### Adjusted R-Squared

Adjusted R-Squared takes into account the number of features and penalizes the addition of irrelevant features:

$$\text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

Where:
- $n$ is the number of samples
- $p$ is the number of features

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

# Calculate Adjusted R²
adj_r2 = adjusted_r2_score(y_true, y_pred, X.shape[1])
print(f"Adjusted R-Squared (Adjusted R²): {adj_r2:.4f}")
```

  </div>
</div>

### Mean Absolute Percentage Error (MAPE)

Mean absolute percentage error calculates the average percentage difference between predicted and actual values:

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

MAPE provides a relative error measure, but it can have issues when the actual values are close to zero.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
```

  </div>
</div>

### Median Absolute Error (MedAE)

Median absolute error calculates the median of the absolute differences between predicted and actual values:

$$MedAE = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, ..., |y_n - \hat{y}_n|)$$

MedAE is more robust to outliers.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.metrics import median_absolute_error

# Calculate MedAE
medae = median_absolute_error(y_true, y_pred)
print(f"Median Absolute Error (MedAE): {medae:.4f}")
```

  </div>
</div>

## Choosing the Right Evaluation Metric

Different evaluation metrics are suitable for different scenarios:

1. **MSE/RMSE**: Use when larger errors need heavier penalties, sensitive to outliers
2. **MAE**: Use when all errors should be treated equally, less sensitive to outliers
3. **R²**: Use when comparing target variables of different scales
4. **Adjusted R²**: Use when comparing models with different numbers of features
5. **MAPE**: Use when relative error is more important, but the target variable should not be close to zero
6. **MedAE**: Use when there are outliers in the data

## Using Cross-Validation to Evaluate Models

Cross-validation trains and evaluates a model on different subsets of the data, providing a more reliable estimate of performance.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Create K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate cross-validation scores
mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# Display results
print("MSE Cross-Validation Scores:")
for i, mse in enumerate(mse_scores):
    print(f"Fold {i+1}: {mse:.4f}")
print(f"Average MSE: {mse_scores.mean():.4f}")
print(f"Standard Deviation: {mse_scores.std():.4f}")

print("\nR² Cross-Validation Scores:")
for i, r2 in enumerate(r2_scores):
    print(f"Fold {i+1}: {r2:.4f}")
print(f"Average R²: {r2_scores.mean():.4f}")
print(f"Standard Deviation: {r2_scores.std():.4f}")
```

  </div>
</div>

### Learning Curve

A learning curve shows how model performance changes with varying training set sizes, helping diagnose overfitting or underfitting.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Calculate mean and standard deviation
train_mean = -train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = -test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.grid()
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Train MSE")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validation MSE")
plt.xlabel("Training Sample Size")
plt.ylabel("MSE")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()
```

  </div>
</div>

## Comparing Multiple Models

In practice, you often need to compare the performance of multiple regression models.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

# Create models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'Neural Network': MLPRegressor(random_state=42)
}

# Evaluation metrics
metrics = {
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'R²': 'r2'
}

# Calculate cross-validation scores
results = {}
for name, model in models.items():
    model_results = {}
    for metric_name, metric in metrics.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=metric)
        if metric.startswith('neg_'):
            scores = -scores
        model_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    results[name] = model_results

# Display results
for model_name, model_results in results.items():
    print(f"\n{model_name}:")
    for metric_name, values in model_results.items():
        print(f"  {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")

# Visual comparison
plt.figure(figsize=(12, 8))
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.25

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# MSE
mse_means = [results[name]['MSE']['mean'] for name in model_names]
mse_stds = [results[name]['MSE']['std'] for name in model_names]
ax1.bar(x, mse_means, width, yerr=mse_stds, label='MSE', color='red', alpha=0.7)
ax1.set_ylabel('MSE')
ax1.set_title('MSE Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right')

# MAE
mae_means = [results[name]['MAE']['mean'] for name in model_names]
mae_stds = [results[name]['MAE']['std'] for name in model_names]
ax2.bar(x, mae_means, width, yerr=mae_stds, label='MAE', color='blue', alpha=0.7)
ax2.set_ylabel('MAE')
ax2.set_title('MAE Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right')

# R²
r2_means = [results[name]['R²']['mean'] for name in model_names]
r2_stds = [results[name]['R²']['std'] for name in model_names]
ax3.bar(x, r2_means, width, yerr=r2_stds, label='R²', color='green', alpha=0.7)
ax3.set_ylabel('R²')
ax3.set_title('R² Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, rotation=45, ha='right')

fig.tight_layout()
plt.show()
```

  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>Common Pitfalls
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Relying solely on one metric</strong>: Different metrics reflect different aspects of model performance</li>
      <li><strong>Neglecting cross-validation</strong>: A single training/test split can lead to high variance estimates</li>
      <li><strong>Over-interpreting R²</strong>: A high R² does not necessarily mean a model is strong at prediction</li>
      <li><strong>Ignoring domain knowledge</strong>: Consider business needs when choosing metrics</li>
    </ul>
  </div>
</div>

## Summary and Reflection

Regression evaluation metrics are important tools for selecting and optimizing regression models. Different metrics are suitable for different scenarios.

### Key Takeaways

- MSE and RMSE penalize larger errors more heavily, suitable for scenarios where outliers have a high impact
- MAE and MedAE are less sensitive to outliers, providing more robust evaluations
- R² and adjusted R² measure the proportion of variance explained by the model, useful for comparing models with different scales
- Cross-validation provides more reliable performance estimates, reducing the risk of overfitting
- Learning curves help diagnose overfitting or underfitting in models

### Reflection Questions

1. When should MSE be chosen over MAE as an evaluation metric?
2. Why can R² be negative, and what does this mean?
3. How can the most appropriate evaluation metric be selected based on business needs?

<div class="practice-link">
  <a href="/projects/regression.html" class="button">Go to Practice Project</a>
</div>
