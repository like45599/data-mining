# Nonlinear Regression Methods

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Key Takeaways
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the basic concepts and application scenarios of nonlinear regression</li>
      <li>Master common nonlinear regression methods such as polynomial regression and decision tree regression</li>
      <li>Learn how to choose the appropriate nonlinear regression model</li>
      <li>Understand the overfitting problem in nonlinear regression and its solutions</li>
    </ul>
  </div>
</div>

## Overview of Nonlinear Regression

Nonlinear regression is a statistical method used to establish a nonlinear relationship between independent and dependent variables. When the data exhibits obvious nonlinear characteristics, linear regression models often fail to capture the true relationship of the data, and nonlinear regression models are needed.

### Application Scenarios

Nonlinear regression is particularly useful in the following scenarios:

- **Biological growth curves**: Such as bacterial growth, population dynamics, etc., following exponential or sigmoid curves
- **Physical phenomena**: Such as radioactive decay, thermodynamic processes, etc.
- **Economics**: Such as consumer behavior, market saturation analysis
- **Drug response**: The nonlinear relationship between drug dosage and effect

### Differences from Linear Regression

| Feature         | Linear Regression           | Nonlinear Regression                   |
|-----------------|-----------------------------|----------------------------------------|
| Model Form      | $y = w_0 + w_1x_1 + ... + w_nx_n$ | Can be any function form, such as exponential, logarithmic, polynomial, etc. |
| Parameter Estimation | Usually has an analytical solution (Least Squares) | Usually requires iterative optimization algorithms |
| Interpretability | Strong, coefficients directly represent variable impacts | May be weaker, depends on the specific model |
| Computational Complexity | Low                        | High                                   |
| Overfitting Risk | Low                         | High                                   |

## Common Nonlinear Regression Methods

### 1. Polynomial Regression

Polynomial regression is an extension of linear regression, which captures nonlinear relationships by adding higher-order terms of the independent variable.

**Mathematical Expression**: $y = w_0 + w_1x + w_2x^2 + ... + w_nx^n$

**Advantages**:
- Simple to implement, can use the linear regression framework
- Works well for moderately complex nonlinear relationships

**Disadvantages**:
- High-degree polynomials can lead to overfitting
- Sensitive to outliers

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate nonlinear data
np.random.seed(0)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(100) * 0.1

# Create polynomial regression model
degrees = [1, 3, 5, 9]
plt.figure(figsize=(14, 5))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())
    
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression)
    ])
    pipeline.fit(X, y)
    
    # Predict
    X_test = np.linspace(0, 1, 100)[:, np.newaxis]
    plt.plot(X_test, pipeline.predict(X_test), label=f"Degree {degree}")
    plt.scatter(X, y, color='navy', s=30, marker='o', label="Training Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.title(f"{degree} Degree Polynomial")

plt.tight_layout()
plt.show()
```

  </div>
</div>

### 2. Decision Tree Regression

Decision tree regression divides the feature space into multiple regions and predicts with constant values within each region.

**Advantages**:
- Captures complex nonlinear relationships
- Does not require data normalization
- Insensitive to outliers

**Disadvantages**:
- Prone to overfitting
- Cannot extrapolate beyond the training data range

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.randn(80) * 0.1

# Fit decision tree
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = regr.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, color="darkorange", label="Data Points")
plt.plot(X_test, y_pred, color="cornflowerblue", label="Decision Tree Prediction", linewidth=2)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

  </div>
</div>

### 3. Support Vector Regression (SVR)

SVR applies Support Vector Machine techniques to regression problems by finding the maximum margin hyperplane while allowing some error.

**Advantages**:
- Effective for high-dimensional data
- Can handle complex nonlinear relationships with kernel tricks
- Robust to outliers

**Disadvantages**:
- High computational complexity
- Parameter tuning is difficult
- Not suitable for large-scale datasets

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Create SVR models
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=0.1, coef0=1)

# Fit models
svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=2,
                 label='{} Model'.format(kernel_label[ix]))
    axes[ix].scatter(X, y, color='darkorange', label='Data Points')
    axes[ix].set_xlabel('Feature')
    axes[ix].set_ylabel('Target')
    axes[ix].set_title('{} Kernel'.format(kernel_label[ix]))
    axes[ix].legend()
fig.tight_layout()
plt.show()
```

  </div>
</div>

### 4. Random Forest Regression

Random Forest regression is an ensemble learning method that builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting.

**Advantages**:
- High prediction accuracy
- Less prone to overfitting
- Can handle high-dimensional data
- Can evaluate feature importance

**Disadvantages**:
- High computational complexity
- Poor model interpretability
- Predictions can be inaccurate for extreme values

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.randn(100) * 0.1

# Create random forest regression model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = rf.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='Data Points')
plt.plot(X_test, y_pred, color='navy', label='Random Forest Prediction')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Random Forest Regression')
plt.legend()
plt.show()
```

  </div>
</div>

### 5. Neural Network Regression

Neural networks can learn complex nonlinear relationships, especially when dealing with large datasets.

**Advantages**:
- Can model extremely complex nonlinear relationships
- Suitable for large-scale datasets
- Can automatically learn feature representations

**Disadvantages**:
- Requires a large amount of data
- High computational resource demands
- Difficult parameter tuning
- Poor interpretability

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.randn(100) * 0.1

# Create neural network regression model
nn_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=2000,
    random_state=42
)

# Fit model
nn_reg.fit(X, y)

# Predict
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = nn_reg.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='Data Points')
plt.plot(X_test, y_pred, color='navy', label='Neural Network Prediction')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Neural Network Regression')
plt.legend()
plt.show()
```

  </div>
</div>

## Model Selection and Overfitting Solutions

### How to Choose the Right Nonlinear Regression Model

When choosing the appropriate nonlinear regression model, consider the following factors:

1. **Data Characteristics**: Understand the data distribution and potential types of nonlinear relationships
2. **Sample Size**: More complex models require more data
3. **Computational Resources**: Some models (like neural networks) require more computational resources
4. **Interpretability Needs**: Polynomial regression might be more appropriate if model interpretability is needed
5. **Prediction Accuracy Requirements**: Generally, complex models provide higher prediction accuracy

### Methods to Handle Overfitting

Nonlinear regression models are prone to overfitting. Here are some common methods to address it:

1. **Regularization**:
   - L1 Regularization (Lasso): Encourages some feature coefficients to be zero
   - L2 Regularization (Ridge): Shrinks all feature coefficients

2. **Cross-validation**: Use k-fold cross-validation to select the optimal model complexity

3. **Feature Selection**: Reduce irrelevant features to lower model complexity

4. **Early Stopping**: Stop training when performance on the validation set begins to degrade

5. **Ensemble Methods**: Combine predictions from multiple simpler models

<div class="code-example">
  <div class="code-example__title">Code Example: Regularized Polynomial Regression</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Generate data
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1
X = X.reshape(-1, 1)

# Create polynomial regression model with Ridge regularization
degrees = [1, 4, 15]
alphas = [0, 0.001, 1.0]

plt.figure(figsize=(14, 8))
for i, degree in enumerate(degrees):
    for j, alpha in enumerate(alphas):
        ax = plt.subplot(len(degrees), len(alphas), i * len(alphas) + j + 1)
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        model.fit(X, y)
        
        # Predict
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        plt.plot(X_test, model.predict(X_test), label=f"Model")
        plt.scatter(X, y, color='navy', s=30, marker='o', label="Training points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.title(f"degree={degree}, alpha={alpha}")
        
plt.tight_layout()
plt.show()
```

  </div>
</div>

## Practical Application Case

### Case: Predicting Housing Prices

In the real estate field, there is typically a nonlinear relationship between housing prices and several factors such as area, location, and age. Below is an example using nonlinear regression to predict housing prices:

<div class="code-example">
  <div class="code-example__title">Code Example: Housing Price Prediction</div>
  <div class="code-example__content">

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

# Train model
gbr.fit(X_train_scaled, y_train)

# Predict
y_pred = gbr.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R²: {r2:.4f}")

# Visualize feature importance
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(housing.feature_names)[sorted_idx])
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Visualize prediction results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.show()
```

  </div>
</div>

## Conclusion

Nonlinear regression is a powerful tool for modeling complex relationships in data. By understanding the characteristics of different models, applying appropriate regularization, and using cross-validation, we can build robust nonlinear regression models that provide high accuracy.

<BackToPath />

<div class="practice-link">
  <a href="/projects/prediction.html" class="button">Go to Practice Projects</a>
</div>
