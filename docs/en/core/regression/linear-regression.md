# Linear Regression

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Key Takeawaysss
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the basic principles and assumptions of linear regression</li>
      <li>Master the differences between simple linear regression and multiple linear regression</li>
      <li>Learn how to evaluate the performance of linear regression models</li>
      <li>Understand how regularization techniques improve linear regression</li>
    </ul>
  </div>
</div>

## Overview of Linear Regression

Linear regression is the most basic and widely used regression analysis method. It is used to establish a relationship model between a dependent variable (target) and one or more independent variables (features). Linear regression assumes a linear relationship between the features and the target.

### Simple Linear Regression

Simple linear regression involves one independent variable and one dependent variable. Its mathematical expression is:

$$y = w_0 + w_1x + \varepsilon$$

Where:
- $y$ is the dependent variable (target)
- $x$ is the independent variable (feature)
- $w_0$ is the intercept (bias term)
- $w_1$ is the slope (weight)
- $\varepsilon$ is the error term

The goal of simple linear regression is to find the best $w_0$ and $w_1$ values that minimize the error between predicted values and actual values.

### Multiple Linear Regression

Multiple linear regression involves multiple independent variables. Its mathematical expression is:

$$y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n + \varepsilon$$

Or in matrix form:

$$y = \mathbf{X}\mathbf{w} + \varepsilon$$

Where:
- $y$ is the dependent variable
- $x_1, x_2, ..., x_n$ are the independent variables
- $w_0, w_1, w_2, ..., w_n$ are the model parameters
- $\varepsilon$ is the error term
- $\mathbf{X}$ is the feature matrix
- $\mathbf{w}$ is the parameter vector

## Assumptions of Linear Regression

The linear regression model is based on the following assumptions:

1. **Linear relationship**: There is a linear relationship between independent and dependent variables
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: The error terms have constant variance
4. **Normality**: The error terms follow a normal distribution
5. **No multicollinearity**: The independent variables are not perfectly linearly correlated

When these assumptions are satisfied, the linear regression model can provide unbiased and efficient parameter estimates.

## Parameter Estimation Methods

### Least Squares Method

The least squares method is the most common parameter estimation technique. Its goal is to minimize the residual sum of squares (RSS):

$$RSS = (y - X\beta)^T(y - X\beta)$$

For simple linear regression, the analytical solution for least squares estimation is:

$$w_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

$$w_0 = \bar{y} - w_1\bar{x}$$

For multiple linear regression, the solution in matrix form is:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Gradient Descent

When the data size is large, calculating $(\mathbf{X}^T\mathbf{X})^{-1}$ can be computationally expensive. In such cases, gradient descent can be used to iteratively solve for the parameters:

1. Initialize parameters $\mathbf{w}$
2. Compute the gradient of the loss function with respect to the parameters
3. Update the parameters in the opposite direction of the gradient
4. Repeat steps 2 and 3 until convergence

The update rule is:

$$w_j := w_j - \alpha \frac{\partial}{\partial w_j} RSS$$

Where $\alpha$ is the learning rate.

<div class="code-example">
  <div class="code-example__title">Code Example: Simple Linear Regression</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Use sklearn's LinearRegression
model = LinearRegression()
model.fit(X, y)

# Get parameters
w0 = model.intercept_[0]
w1 = model.coef_[0][0]
print(f"Intercept (w0): {w0:.4f}")
print(f"Slope (w1): {w1:.4f}")

# Predict
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_pred, 'r-', linewidth=2, label=f'y = {w0:.2f} + {w1:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate model
y_pred_all = model.predict(X)
mse = mean_squared_error(y, y_pred_all)
r2 = r2_score(y, y_pred_all)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R²: {r2:.4f}")
```

  </div>
</div>

<div class="code-example">
  <div class="code-example__title">Code Example: Multiple Linear Regression</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Get parameters
w0 = model.intercept_
w = model.coef_
print(f"Intercept (w0): {w0:.4f}")
print("Feature weights (w):")
for i, feature_name in enumerate(housing.feature_names):
    print(f"  {feature_name}: {w[i]:.4f}")

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R²: {r2:.4f}")

# Visualize prediction results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()
```

  </div>
</div>

## Regularization Techniques

When there are many features or multicollinearity among the features, standard linear regression may lead to overfitting. Regularization techniques reduce the risk of overfitting by adding a penalty term to the loss function.

### Ridge Regression (L2 Regularization)

Ridge regression adds a penalty term proportional to the square of the coefficients to reduce model complexity:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha ||\mathbf{w}||_2^2$$

Where $\alpha$ is the regularization strength parameter. Ridge regression shrinks all coefficients, but does not set them to zero.

### Lasso Regression (L1 Regularization)

Lasso regression adds a penalty term proportional to the absolute value of the coefficients:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha ||\mathbf{w}||_1$$

An important feature of Lasso regression is that it can shrink some coefficients to exactly zero, thereby performing feature selection.

### Elastic Net

Elastic Net combines the penalty terms from Ridge and Lasso:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha_1 ||\mathbf{w}||_1 + \alpha_2 ||\mathbf{w}||_2^2$$

Elastic Net overcomes some of the limitations of Lasso when handling highly correlated features, while still retaining feature selection capabilities.

<div class="code-example">
  <div class="code-example__title">Code Example: Regularized Linear Regression</div>
  <div class="code-example__content">

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train models
ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)
elastic.fit(X_train_scaled, y_train)

# Predict
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_elastic = elastic.predict(X_test_scaled)

# Calculate MSE
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)

print(f"Ridge MSE: {mse_ridge:.4f}")
print(f"Lasso MSE: {mse_lasso:.4f}")
print(f"ElasticNet MSE: {mse_elastic:.4f}")

# Visualize coefficients
plt.figure(figsize=(12, 6))
plt.plot(ridge.coef_, 's-', label='Ridge')
plt.plot(lasso.coef_, 'o-', label='Lasso')
plt.plot(elastic.coef_, '^-', label='ElasticNet')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Coefficients with Different Regularization Methods')
plt.legend()
plt.grid(True)
plt.show()
```

  </div>
</div>

## Advantages and Disadvantages of Linear Regression

### Advantages

- **Simple and interpretable**: The model is simple and the parameters have clear interpretations.
- **Computationally efficient**: Especially for small to medium-sized datasets.
- **No hyperparameter tuning required**: Standard linear regression does not have hyperparameters to tune.
- **Good baseline model**: Provides a baseline for comparing more complex models.

### Disadvantages

- **Assumption limitations**: Assumes a linear relationship between features and the target.
- **Sensitive to outliers**: Outliers can significantly affect the model parameters.
- **Cannot capture non-linear relationships**: Performs poorly when the relationship between data is complex.
- **Multicollinearity issues**: Parameter estimates are unstable when features are highly correlated.

## Practical Application Case

### Case: Predicting Housing Prices

Predicting housing prices is a classic application of linear regression. Below is an example using the California housing dataset:

<div class="code-example">
  <div class="code-example__title">Code Example: California Housing Price Prediction</div>
  <div class="code-example__content">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Data exploration
print("Dataset Shape:", X.shape)
print("Feature Names:", housing.feature_names)
print("Feature Description:")
print(X.describe())

# Correlation analysis
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target
correlation = data.corr()
plt.figure(figsize=(12, 10))
plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation)), correlation.columns)
plt.title('Feature Correlation Heatmap')
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Get coefficients
coefficients = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': model.coef_
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)

# Visualize coefficients
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.grid(True)
plt.show()

# Predict and evaluate
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Visualize prediction results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()

# Residual analysis
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='r', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()
```

  </div>
</div>

## Conclusion

Linear regression is one of the most fundamental algorithms in data mining and machine learning. Although simple, it performs well in many practical applications. Understanding the principles and assumptions of linear regression, mastering parameter estimation methods and regularization techniques is crucial for building effective predictive models.

<BackToPath />

<div class="practice-link">
  <a href="/projects/prediction.html" class="button">Go to Practice Projects</a>
</div>
