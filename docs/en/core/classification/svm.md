# Support Vector Machine (SVM)

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the basic principles and mathematical foundation of SVM</li>
      <li>Master the difference between linear and nonlinear SVM</li>
      <li>Learn the role of kernel functions and how to choose them</li>
      <li>Practice applying SVM in classification problems</li>
    </ul>
  </div>
</div>

## Basic Principles of SVM

Support Vector Machine (SVM) is a powerful supervised learning algorithm widely used in classification and regression problems. The core idea of SVM is to find an optimal hyperplane that can maximize the margin between different classes of data points.

### Linearly Separable Case

In the simplest case of binary classification with linearly separable data, SVM tries to find a hyperplane that:

1. Correctly classifies all training samples
2. Maximizes the distance (margin) to the nearest training sample point

<div class="visualization-container">
  <div class="visualization-title">Linear SVM Principle</div>
  <div class="visualization-content">
    <img src="/images/svm_linear.svg" alt="Linear SVM Principle Diagram">
  </div>
  <div class="visualization-caption">
    Figure: Decision boundary and support vectors of Linear SVM. Red and blue points represent two different classes, the black solid line is the decision boundary, the dashed lines represent the margin boundaries, and the circled points are support vectors.
  </div>
</div>

### Mathematical Expression

The optimization problem of linear SVM can be represented as:

$$\min_{w, b} \frac{1}{2} ||w||^2$$

$$\text{s.t. } y_i(w^T x_i + b) \geq 1, \forall i=1,\ldots,n$$

Where:
- $w$ is the normal vector, which determines the direction of the hyperplane
- $b$ is the bias term, which determines the position of the hyperplane
- $x_i$ is the feature vector
- $y_i$ is the class label (+1 or -1)
- The constraint ensures all samples are correctly classified and the margin is at least 1

### Support Vectors

Support vectors are the data points that are closest to the decision boundary, and they satisfy:

$$y_i(w^T x_i + b) = 1$$

These points are crucial for determining the decision boundary, while other points do not influence the model. This is an important feature of SVM, making it efficient even in high-dimensional spaces.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>The SVM algorithm was first proposed by Vladimir Vapnik and Alexey Chervonenkis in 1963, but it became widely popular in the 1990s with the introduction of kernel methods. The theoretical foundation of SVM comes from the VC dimension theory in statistical learning theory, which provides theoretical guarantees for the model's generalization ability.</p>
  </div>
</div>

## Soft Margin SVM

### Handling Linearly Inseparable Data

In real-world applications, data is often not perfectly linearly separable, and may contain noise or outliers. Soft margin SVM introduces slack variables, allowing some samples to violate the constraints:

$$\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0, \forall i=1,\ldots,n$$

Where:
- $\xi_i$ is the slack variable, representing the degree of violation for the $i$th sample
- $C$ is the penalty parameter, controlling the trade-off between maximizing the margin and minimizing misclassification

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Generate example data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_train, y_train)

# Prediction
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize decision boundary
def plot_decision_boundary(model, X, y):
    # Set grid range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict class labels for the grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and sample points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    
    # Mark support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')
    
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(svm, X, y)
```

  </div>
</div>

### Impact of the C Parameter

The parameter C controls the regularization strength, affecting the model complexity:

- **Large C value**: Emphasizes correct classification of each sample, which may lead to overfitting
- **Small C value**: Allows more misclassifications but aims for a larger margin, typically resulting in better generalization

<div class="visualization-container">
  <div class="visualization-title">Impact of C Parameter on SVM</div>
  <div class="visualization-content">
    <img src="/images/svm_c_parameter.svg" alt="Impact of C Parameter on SVM">
  </div>
  <div class="visualization-caption">
    Figure: Impact of different C values on the SVM decision boundary. Left: C=0.1, Middle: C=1, Right: C=10.
  </div>
</div>

## Kernel Methods

### Handling Nonlinear Problems

When data is not linearly separable in the original feature space, SVM uses kernel tricks to map the data to a higher-dimensional feature space, where it may become linearly separable.

The kernel function $K(x_i, x_j)$ computes the inner product of two samples in the higher-dimensional feature space without explicitly computing the mapping:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Where $\phi$ is the mapping function from the original feature space to the higher-dimensional space.

### Common Kernel Functions

1. **Linear Kernel**: $K(x_i, x_j) = x_i^T x_j$
   - Suitable for linearly separable data
   - Computationally efficient

2. **Polynomial Kernel**: $K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$
   - Suitable for data with clear polynomial relationships
   - Parameters: $\gamma$ (scaling), $r$ (bias), $d$ (degree)

3. **Radial Basis Function (RBF) Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
   - Most commonly used nonlinear kernel
   - Suitable for a variety of complex data
   - Parameter: $\gamma$ (controls the radius of influence)

4. **Sigmoid Kernel**: $K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$
   - Derived from neural networks
   - Parameters: $\gamma$ (scaling), $r$ (bias)

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

# Generate nonlinear data
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Create SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
plt.figure(figsize=(16, 4))

for i, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, gamma=2)
    svm.fit(X, y)
    
    # Visualize
    plt.subplot(1, 4, i+1)
    
    # Set grid range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict class labels for the grid points
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and sample points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f'Kernel: {kernel}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

  </div>
</div>

### Kernel Function Selection Guide

- **Linear Kernel**: For large datasets, high-dimensional features, linearly separable data
- **RBF Kernel**: For medium-sized datasets, low-dimensional features, nonlinear relationships
- **Polynomial Kernel**: When there is a polynomial relationship between features
- **Sigmoid Kernel**: For neural network-related problems

## Practical Applications of SVM

### 1. Data Preprocessing

SVM is highly sensitive to feature scaling, and preprocessing is crucial:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Create a pipeline for preprocessing and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))  # SVM model
])

# Fit the pipeline to the data
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)
```

  </div>
</div>

### 2. Hyperparameter Tuning

Use grid search to find the best parameters:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'poly', 'sigmoid']
}

# Create grid search
grid_search = GridSearchCV(
    pipeline,  # Using previously defined pipeline
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all available CPUs
)

# Execute grid search
grid_search.fit(X_train, y_train)

# Output the best parameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Predict with the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

  </div>
</div>

### 3. Handling Imbalanced Data

For imbalanced datasets, class weights can be used:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
# Use 'balanced' to automatically adjust class weights
svm = SVC(kernel='rbf', class_weight='balanced')

# Or manually specify class weights
class_weights = {0: 1.0, 1: 5.0}  # Assuming class 1 is the minority class
svm = SVC(kernel='rbf', class_weight=class_weights)
```

  </div>
</div>

### 4. SVM for Multi-Class Classification

Although SVM is inherently a binary classifier, it can handle multi-class problems using these strategies:

- **One-vs-One (OvO)**: Train a classifier for each pair of classes, requiring $\frac{n(n-1)}{2}$ classifiers
- **One-vs-Rest (OvR)**: Train a classifier for each class (that class vs all others), requiring $n$ classifiers

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load multi-class dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# One-vs-One strategy (default)
svm_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
svm_ovo.fit(X_train, y_train)

# One-vs-Rest strategy
svm_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
svm_ovr.fit(X_train, y_train)

# Evaluate
y_pred_ovo = svm_ovo.predict(X_test)
y_pred_ovr = svm_ovr.predict(X_test)

print("One-vs-One Strategy Evaluation Report:")
print(classification_report(y_test, y_pred_ovo, target_names=iris.target_names))

print("One-vs-Rest Strategy Evaluation Report:")
print(classification_report(y_test, y_pred_ovr, target_names=iris.target_names))
```

  </div>
</div>

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>Common Pitfalls
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Neglecting Feature Scaling</strong>: SVM is very sensitive to feature scales, and unscaled features may severely degrade performance</li>
      <li><strong>Blindly Using RBF Kernel</strong>: While the RBF kernel is powerful, for large-scale linearly separable data, a linear kernel may be more efficient</li>
      <li><strong>Over-Tuning Parameters</strong>: Over-optimizing C and gamma can lead to overfitting</li>
      <li><strong>Ignoring Class Imbalance</strong>: Failing to set class weights in imbalanced data can lead to bias towards the majority class</li>
    </ul>
  </div>
</div>

## Summary and Reflection

SVM is a powerful classification algorithm that, through margin maximization and kernel tricks, can effectively handle both linear and nonlinear classification problems.

### Key Takeaways

- SVM seeks to find the decision boundary with the largest margin
- Support vectors are the key points that determine the boundary
- Soft margin SVM handles noise and outliers through the C parameter
- Kernel methods enable SVM to deal with nonlinear problems
- Feature scaling is critical to SVM performance

### Reflection Questions

1. What are the advantages and disadvantages of SVM compared to logistic regression?
2. In what cases should SVM be chosen over other classification algorithms?
3. How can SVM's computational efficiency be addressed with large datasets?

<BackToPath />

<div class="practice-link">
  <a href="/projects/classification.html" class="button">Go to Practice Project</a>
</div>
