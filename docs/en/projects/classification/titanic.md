# Titanic Survival Prediction

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Beginner level</li>
      <li><strong>Type</strong>: Binary classification</li>
      <!-- <li><strong>Estimated time</strong>: 4-6 hours</li> -->
      <li><strong>Skills</strong>: Data cleaning, feature engineering, classification algorithms, model evaluation</li>
      <li><strong>Related knowledge module</strong>: <a href="/core/classification/svm.html">Classification algorithms</a></li>
    </ul>
  </div>
</div>

## Project Background

On April 15, 1912, the luxury passenger liner Titanic, famously known as the "unsinkable ship," collided with an iceberg and sank during its maiden voyage. Of the 2,224 passengers and crew, 1,502 lost their lives. This tragedy shocked the world and led to improvements in maritime safety regulations.

In this project, we will analyze the passenger data from the Titanic and attempt to answer the question, "Which types of people were more likely to survive?" By building a predictive model, we can identify key factors that influenced survival rates.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span> Did you know?
  </div>
  <div class="knowledge-card__content">
    <p>The sinking of the Titanic demonstrated the impact of social class in disasters. The "women and children first" policy and the different locations of cabins led to significant differences in survival rates. First-class passengers had a survival rate of about 62%, second-class passengers had a survival rate of 41%, and third-class passengers had a survival rate of only 25%.</p>
  </div>
</div>

## Dataset Introduction

The dataset used in this project contains information about 891 passengers aboard the Titanic. Each passenger's record includes the following features:

- **PassengerId**: Passenger ID
- **Survived**: Survival status (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = First class, 2 = Second class, 3 = Third class)
- **Name**: Passenger's name
- **Sex**: Gender
- **Age**: Age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Embarkation port (C = Cherbourg, Q = Queenstown, S = Southampton)

There are some missing values in the dataset, especially in the Age, Cabin, and Embarked features. Handling these missing values will be an important part of the data preprocessing.

## Project Goals

1. Explore and analyze Titanic passenger data.
2. Handle missing values and prepare features.
3. Build a classification model to predict passenger survival.
4. Evaluate model performance and interpret results.
5. Submit prediction results.

## Implementation Steps

### Step 1: Data Exploration and Visualization

First, we need to load the data and conduct an initial exploration to understand the basic features and distribution of the data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Load the data
train_data = pd.read_csv('titanic_train.csv')

# View basic information about the data
print(train_data.shape)
train_data.head()

# Check for missing values
train_data.isnull().sum()

# Basic statistical description
train_data.describe()

# Overview of survival rate
train_data['Survived'].value_counts(normalize=True)

# Visualize the relationship between different features and survival rate
plt.figure(figsize=(12, 5))

plt.subplot(131)
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Gender')

plt.subplot(132)
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Passenger Class')

plt.subplot(133)
sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.title('Survival Rate by Embarkation Port')

plt.tight_layout()
plt.show()
```

### Step 2: Data Preprocessing

Next, we need to handle the missing values and convert categorical features into formats that are usable by the model.

```python
# Handle missing age data
age_median = train_data['Age'].median()
train_data['Age'].fillna(age_median, inplace=True)

# Handle missing embarkation port data
embarked_mode = train_data['Embarked'].mode()[0]
train_data['Embarked'].fillna(embarked_mode, inplace=True)

# Create new feature: Family size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# Create new feature: Traveling alone
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

# Extract titles from names
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Map titles to categories
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Don": "Royalty",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Sir": "Royalty",
    "Capt": "Officer",
    "Ms": "Mrs"
}
train_data['Title'] = train_data['Title'].map(title_mapping)

# Select features to use
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']

# Perform one-hot encoding for categorical features
train_encoded = pd.get_dummies(train_data[features])

# Prepare feature matrix and target variable
X = train_encoded
y = train_data['Survived']
```

### Step 3: Model Building and Evaluation

Now, we can build classification models and evaluate their performance.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_val)
dt_accuracy = accuracy_score(y_val, dt_pred)
print(f"Decision Tree accuracy: {dt_accuracy:.4f}")

# Random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_pred)
print(f"Random Forest accuracy: {rf_accuracy:.4f}")

# SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_val)
svm_accuracy = accuracy_score(y_val, svm_pred)
print(f"SVM accuracy: {svm_accuracy:.4f}")

# Cross-validation
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# View confusion matrix and classification report for the best model
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_val, rf_pred))
print("\nRandom Forest Classification Report:")
print(classification_report(y_val, rf_pred))

# Feature importance
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
```

### Step 4: Model Optimization

We can further improve the model's performance through hyperparameter tuning.

```python
from sklearn.model_selection import GridSearchCV

# Random forest parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use the best model
best_rf_model = grid_search.best_estimator_
```

### Step 5: Predicting on the Test Set

Finally, we use the best model to make predictions on the test set.

```python
# Load the test data
test_data = pd.read_csv('titanic_test.csv')

# Preprocess the test data (repeat the preprocessing steps above)

# Predict with the best model
test_predictions = best_rf_model.predict(test_encoded)

# Create a submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)
```

## Results Analysis

Through our analysis and modeling, we found the following factors significantly influenced Titanic passengers' survival rates:

1. **Gender**: Women had a significantly higher survival rate than men.
2. **Passenger Class**: First-class passengers had a higher survival rate than second and third-class passengers.
3. **Age**: Children had a higher survival rate than adults.
4. **Family Size**: Passengers from medium-sized families had a higher survival rate.
5. **Title**: Titles reflecting social status were associated with survival rates.

These findings align with historical records, reflecting the "women and children first" policy and the impact of social class on survival chances.

## Advanced Challenges

If you've completed the basic tasks, you can try the following advanced challenges:

1. **Feature Engineering**: Try creating more meaningful features, such as fare range, age groups, etc.
2. **Model Ensemble**: Use voting or stacking methods to combine predictions from multiple models.
3. **Survival Probability Analysis**: Instead of predicting survival, analyze survival probabilities for different feature combinations.
4. **Visualization**: Create more advanced visualizations, such as decision tree visualizations or survival heatmaps.
5. **Missing Value Handling**: Try more complex methods for handling missing values, such as imputation based on similar passengers' features.

## Conclusion and Reflection

Through this project, we learned how to handle a complete classification problem, from data exploration to model deployment. The Titanic dataset, although small, contains common challenges in data mining, such as handling missing values, feature engineering, and model selection.

In practical applications, such analysis can help us understand the key factors influencing a particular outcome, leading to more effective strategies and policies. For example, in disaster management, similar analyses can help identify high-risk groups and prioritize resource allocation.

### Reflective Questions

1. Does our model exhibit any biases? How can we ensure fairness in predictions?
2. How can we apply this analysis approach to predict other disasters or events?
3. Besides accuracy, what other metrics can we use to evaluate model effectiveness?

<div class="practice-link">
  <a href="/projects/classification/spam-filter.html" class="button">Next item: Spam filter</a>
</div> 