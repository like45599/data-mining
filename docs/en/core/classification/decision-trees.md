# Decision Tree Algorithm

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Key Takeaways
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the basic principles and construction process of decision trees</li>
      <li>Master decision tree splitting criteria and pruning techniques</li>
      <li>Learn the application of decision trees in classification and regression problems</li>
      <li>Understand the advantages and disadvantages of decision trees and their improvement methods</li>
    </ul>
  </div>
</div>

## Basic Principles of Decision Trees

A decision tree is a tree-like structure used for classification and regression models. It divides the data into different subsets through a series of questions, ultimately providing a prediction.

### Structure of a Decision Tree

A decision tree consists of the following parts:

- **Root Node**: The starting node that contains all the samples.
- **Internal Nodes**: Represent feature test conditions.
- **Branches**: Represent the result of a test condition.
- **Leaf Nodes**: Represent the final classification or regression result.

<div class="visualization-container">
  <div class="visualization-title">Decision Tree Structure Example</div>
  <div class="visualization-content">
    <img src="/images/decision_tree_structure.svg" alt="Decision Tree Structure Example">
  </div>
  <div class="visualization-caption">
    Image: A simple decision tree example. Root and internal nodes represent feature tests, branches represent test results, and leaf nodes represent the final classification.
  </div>
</div>

### Decision Tree Construction Process

The construction of a decision tree is usually done in a top-down recursive manner. The main steps are:

1. **Select the Best Feature**: Use metrics like information gain or Gini impurity to select the best feature to split.
2. **Split the Dataset**: Divide the dataset into subsets based on the selected feature.
3. **Recursively Build Subtrees**: Repeat the above process for each subset.
4. **Set Stop Conditions**: Stop splitting when specific conditions are met (e.g., maximum depth or too few samples in a node).

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span> Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>The early version of the decision tree algorithm, CART (Classification and Regression Trees), was proposed by Leo Breiman et al. in 1984. The earlier ID3 algorithm was developed by Ross Quinlan in 1979. Decision trees are not only a powerful machine learning algorithm but also form the basis of ensemble methods like Random Forests and Gradient Boosting Trees.</p>
  </div>
</div>

## Splitting Criteria

The core of decision tree algorithms is how to choose the best feature and split point. Commonly used splitting criteria include:

### 1. Information Gain (ID3 Algorithm)

Information gain is based on the reduction in entropy, where entropy measures the uncertainty of a system:

$$\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

where $p_i$ is the proportion of samples in class $i$, and $c$ is the number of classes.

Information gain is calculated as:

$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$

where $A$ is the feature, and $S_v$ is the subset of samples where feature $A$ takes the value $v$.

### 2. Gain Ratio (C4.5 Algorithm)

To overcome the bias of information gain towards features with more values, C4.5 introduces gain ratio:

$$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$

where:

$$\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

### 3. Gini Impurity (CART Algorithm)

Gini impurity measures the impurity of a set, with lower values indicating purer sets:

$$\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$$

The Gini index is calculated as:

$$\text{Gini\_Index}(S, A) = \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Gini}(S_v)$$

The feature that minimizes the Gini index is chosen as the best splitting feature.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train decision tree model (using Gini impurity)
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_gini.fit(X_train, y_train)

# Create and train decision tree model (using information gain)
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_entropy.fit(X_train, y_train)

# Predict
y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

# Evaluate
gini_accuracy = dt_gini.score(X_test, y_test)
entropy_accuracy = dt_entropy.score(X_test, y_test)

print(f"Gini impurity decision tree accuracy: {gini_accuracy:.4f}")
print(f"Information gain decision tree accuracy: {entropy_accuracy:.4f}")

# Visualize the decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_gini, feature_names=iris.feature_names, 
               class_names=iris.target_names, filled=True)
plt.title("Gini impurity decision tree")
plt.show()
```

  </div>
</div>

## Pruning Decision Trees

Decision trees are prone to overfitting, and pruning is an important technique to prevent this.

### Pre-pruning

Pre-pruning stops the tree growth early during construction:

- Limit the maximum depth of the tree
- Limit the minimum number of samples per node
- Limit the minimum information gain for splitting

### Post-pruning

Post-pruning builds a complete tree and then prunes unimportant subtrees:

- Error rate-based pruning
- Cost complexity pruning (CART algorithm)

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter grid
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.02, 0.03, 0.04]  # Cost complexity parameter for post-pruning
}

# Create grid search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Execute grid search
grid_search.fit(X_train, y_train)

# Output best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use the best model for prediction
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")

# Visualize the best decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(best_model, feature_names=cancer.feature_names, 
               class_names=['Malignant', 'Benign'], filled=True, max_depth=3)
plt.title("Best decision tree (limit depth to 3)")
plt.show()
```

  </div>
</div>

## Feature Importance

Decision trees can calculate feature importance, helping to understand which features have the most impact on predictions:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import matplotlib.pyplot as plt

# Use the previously trained best model
feature_importance = best_model.feature_importances_

# Create a feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

# Visualize the top 15 important features
plt.figure(figsize=(12, 8))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Decision Tree Feature Importance')
plt.gca().invert_yaxis()  # Display from top to bottom
plt.show()
```

  </div>
</div>

## Decision Tree Regression

Decision trees can also be used for regression problems:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
boston = load_boston()
X, y = boston.data, boston.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train decision tree regression model
regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize prediction results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Decision Tree Regression: Actual vs Predicted')
plt.show()
```

  </div>
</div>

## Advantages and Disadvantages of Decision Trees

### Advantages

1. **Easy to understand and interpret**: Decision trees offer a clear, visual representation of the decision process.
2. **No need for feature scaling**: They are not sensitive to the scale of features.
3. **Handle both numerical and categorical features**: No need for special encoding.
4. **Handle multi-class problems**: Naturally supports multi-class classification.
5. **Can handle missing values**: Can learn patterns to handle missing data.

### Disadvantages

1. **Prone to overfitting**: Especially when the tree depth is large.
2. **Unstable**: Small changes in data can result in significant changes in the tree structure.
3. **Bias towards features with many values**: Tends to prefer features with more values.
4. **Difficult to learn certain relationships**: Such as XOR relationships.
5. **Limited predictive performance**: A single tree usually performs worse than ensemble methods.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span> Common Pitfalls
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Neglecting pruning</strong>: Failing to prune the tree properly can lead to overfitting.</li>
      <li><strong>Over-relying on a single tree</strong>: Not considering ensemble methods for complex problems.</li>
      <li><strong>Ignoring class imbalance</strong>: Not using the class_weight parameter to address imbalanced data.</li>
      <li><strong>Misinterpreting feature importance</strong>: Feature importance does not imply causality.</li>
    </ul>
  </div>
</div>

## Improvements and Extensions to Decision Trees

To overcome the limitations of decision trees, various improvements and extensions have been developed:

1. **Random Forest**: Builds multiple trees and averages/votes the results.
2. **Gradient Boosting Trees**: Builds trees sequentially using gradient boosting.
3. **XGBoost/LightGBM**: Efficient implementations of gradient boosting trees.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest accuracy: {accuracy:.4f}")

# Compare with single decision tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Single decision tree accuracy: {dt_accuracy:.4f}")
```

  </div>
</div>

## Summary and Reflection

Decision trees are an intuitive and powerful machine learning algorithm used for both classification and regression. While a single decision tree has certain limitations, it serves as the foundation for many advanced ensemble methods.

### Key Takeaways

- Decision trees build a tree structure by recursively splitting the data.
- Common splitting criteria include information gain, gain ratio, and Gini impurity.
- Pruning is crucial to prevent overfitting.
- Decision trees can calculate feature importance, offering model interpretability.
- Ensemble methods like Random Forest can overcome the limitations of single trees.

### Reflection Questions

1. When should you choose a decision tree over other classification/regression algorithms?
2. How can you balance the complexity and predictive performance of a decision tree?
3. Why do Random Forests typically perform better than a single decision tree?

<BackToPath />

<div class="practice-link">
  <a href="/projects/classification.html" class="button">Go to Practice Projects</a>
</div>
