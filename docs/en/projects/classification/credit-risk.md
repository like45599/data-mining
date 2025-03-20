# Credit Risk Assessment

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Advanced</li>
      <li><strong>Type</strong>: Classification</li>
      <!-- <li><strong>Estimated Time</strong>: 6-8 hours</li> -->
      <li><strong>Skill Points</strong>: Handling imbalanced data, feature engineering, model interpretation, evaluation metric selection</li>
      <li><strong>Related Knowledge Module</strong>: <a href="/core/classification/svm.html">Classification Algorithms</a></li>
    </ul>
  </div>
</div>

## Project Background

Credit risk assessment is one of the core businesses of financial institutions, used to determine whether a borrower is capable of repaying a loan on time. Accurate credit risk assessment can help financial institutions reduce non-performing loans while providing financial services to more qualified borrowers.

Traditional credit scoring models mainly rely on factors such as a borrower's credit history, income level, and debt ratio. With the development of big data and machine learning technologies, modern credit risk assessment can integrate data from additional dimensions—including transaction behavior, social networks, and psychological characteristics—to build more comprehensive and accurate risk prediction models.

In this project, we will use the German credit dataset to build a credit risk assessment model that predicts whether a borrower will default.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>The concept of credit scoring was first proposed by Fair Isaac Corporation in the 1950s and has since evolved into the widely used FICO scoring system. Modern credit scoring systems typically quantify a borrower's creditworthiness with scores ranging from 300 to 850, with higher scores indicating lower credit risk.</p>
  </div>
</div>

## Dataset Introduction

The German credit dataset contains information on 1000 borrowers, each with 20 features, including:

- **Personal Information**: Age, gender, marital status, etc.
- **Financial Status**: Income, savings, number of existing credits, etc.
- **Employment Information**: Employment type, years of employment, etc.
- **Housing Information**: Housing type, duration of residence, etc.
- **Credit History**: Past credit records, loan purpose, etc.

The target variable is binary: good customer (0) or bad customer (1), where a bad customer indicates a risk of default.

An important characteristic of the dataset is its class imbalance—good customers account for 70% while bad customers account for 30%. This imbalance reflects the real-world distribution of credit risk, but also poses challenges for model training.

## Project Objectives

1. Build a classification model that can accurately predict a borrower's credit risk  
2. Address the class imbalance in the dataset  
3. Identify the key factors influencing credit risk  
4. Evaluate the model's performance using various metrics  
5. Provide model interpretation and business recommendations  

## Implementation Steps

### 1. Data Exploration and Preprocessing

First, we need to understand the basic characteristics and quality of the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ['status', 'duration', 'credit_history', 'purpose', 'amount', 
                'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 
                'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 
                'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target']
df = pd.read_csv(url, sep=' ', header=None, names=column_names)

# Convert the target variable to 0 and 1 (the original data uses 1 and 2)
df['target'] = df['target'].map({1: 0, 2: 1})

# View basic information about the data
print(df.info())
print(df.describe())

# Check class distribution
print(df['target'].value_counts(normalize=True))

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Credit Risk Distribution')
plt.xlabel('Risk Class (0 = Good Customer, 1 = Bad Customer)')
plt.ylabel('Count')
plt.show()

# Explore the relationship between numerical features and the target variable
numerical_features = ['duration', 'amount', 'age', 'existing_credits']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.boxplot(x='target', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'{feature} vs Credit Risk')
    axes[i].set_xlabel('Risk Class (0 = Good Customer, 1 = Bad Customer)')

plt.tight_layout()
plt.show()

# Explore the relationship between categorical features and the target variable
categorical_features = ['credit_history', 'purpose', 'personal_status_sex', 'savings']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    cross_tab = pd.crosstab(df[feature], df['target'], normalize='index')
    cross_tab.plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(f'{feature} vs Credit Risk')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Proportion')
    axes[i].legend(['Good Customer', 'Bad Customer'])

plt.tight_layout()
plt.show()
```

Next, we need to process the categorical and numerical features in the data:

```python
# Identify numerical and categorical features
numerical_features = ['duration', 'amount', 'installment_rate', 'present_residence', 
                      'age', 'existing_credits', 'num_dependents']
categorical_features = ['status', 'credit_history', 'purpose', 'savings', 
                        'employment_duration', 'personal_status_sex', 'other_debtors', 
                        'property', 'other_installment_plans', 'housing', 'job', 
                        'telephone', 'foreign_worker']

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Check the shape of the processed features
print(f"Processed training set shape: {X_train_processed.shape}")
```

### 2. Handling Class Imbalance

Credit risk data typically exhibits class imbalance issues, which can be addressed using various methods:

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# Check original class distribution
print(f"Original training set class distribution: {Counter(y_train)}")

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
print(f"Training set class distribution after SMOTE: {Counter(y_train_smote)}")

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_processed, y_train)
print(f"Training set class distribution after undersampling: {Counter(y_train_rus)}")

# Apply combined sampling (SMOTE + undersampling)
over = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversample the minority class to 50% of the majority class
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Undersample the majority class to 80% of the minority class
combined_sampling = ImbPipeline(steps=[('over', over), ('under', under)])
X_train_combined, y_train_combined = combined_sampling.fit_resample(X_train_processed, y_train)
print(f"Training set class distribution after combined sampling: {Counter(y_train_combined)}")
```

### 3. Model Training and Evaluation

We will experiment with multiple classification algorithms and use evaluation metrics suitable for imbalanced data:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score

# Define evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Print evaluation results
    print(f"\n{model_name} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.show()
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.show()
    
    return model, accuracy, precision, recall, f1, auc

# Train and evaluate multiple models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
}

results = []
for name, model in models.items():
    model_result = evaluate_model(model, X_train_smote, y_train_smote, X_test_processed, y_test, name)
    results.append((name,) + model_result[1:])

# Compare the performance of different models
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
print("\nModel Performance Comparison:")
print(results_df)

# Visualize the comparison
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4. Feature Importance Analysis

Understand which features are most important for predicting credit risk:

```python
# Use the feature importances from the Random Forest model
rf_model = models['Random Forest']
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualize the top 15 important features
plt.figure(figsize=(12, 8))
plt.title('Feature Importance')
plt.bar(range(15), importances[indices[:15]], align='center')
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=90)
plt.tight_layout()
plt.show()

# Use SHAP values for a more detailed explanation
import shap

# Choose a model for explanation (e.g., Random Forest)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_processed)

# Visualize the SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values[1], X_test_processed, feature_names=feature_names)
```

### 5. Model Optimization

Optimize the best model using cross-validation and grid search:

```python
from sklearn.model_selection import GridSearchCV

# Assume Random Forest is the best model
best_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use grid search to find the best parameters
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # Using F1 score as the optimization metric
    n_jobs=-1
)

grid_search.fit(X_train_smote, y_train_smote)

# Print the best parameters
print("Best Parameters:")
print(grid_search.best_params_)

# Evaluate the model with the best parameters
best_rf = grid_search.best_estimator_
evaluate_model(best_rf, X_train_smote, y_train_smote, X_test_processed, y_test, "Optimized Random Forest")
```

### 6. Threshold Optimization

In credit risk assessment, the costs of misclassification are asymmetric. We can adjust the classification threshold to balance precision and recall:

```python
# Get prediction probabilities
y_prob = best_rf.predict_proba(X_test_processed)[:, 1]

# Calculate precision and recall for different thresholds
thresholds = np.arange(0, 1, 0.01)
precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_threshold))
    recall_scores.append(recall_score(y_test, y_pred_threshold))
    f1_scores.append(f1_score(y_test, y_pred_threshold))

# Visualize performance under different thresholds
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Model Performance at Different Thresholds')
plt.legend()
plt.grid(True)
plt.show()

# Find the threshold with the highest F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Highest F1 Score at Best Threshold: {max(f1_scores):.4f}")

# Make final predictions using the best threshold
y_pred_final = (y_prob >= best_threshold).astype(int)
print("\nFinal Evaluation with the Best Threshold:")
print(classification_report(y_test, y_pred_final))
```

### 7. Business Interpretation and Recommendations

Finally, we need to translate the model results into business insights:

```python
# Calculate rejection rate and bad debt rate
def calculate_business_metrics(y_true, y_pred, y_prob, threshold):
    # Make predictions using the given threshold
    y_pred_threshold = (y_prob >= threshold).astype(int)
    
    # Calculate rejection rate (the proportion predicted as bad customers)
    rejection_rate = np.mean(y_pred_threshold)
    
    # Calculate bad debt rate (the proportion of actual bad customers among those predicted as good)
    approved_indices = y_pred_threshold == 0
    if np.sum(approved_indices) > 0:
        bad_debt_rate = np.mean(y_true[approved_indices] == 1)
    else:
        bad_debt_rate = 0
    
    return rejection_rate, bad_debt_rate

# Calculate business metrics for different thresholds
business_metrics = []
for threshold in thresholds:
    rejection_rate, bad_debt_rate = calculate_business_metrics(y_test, y_pred, y_prob, threshold)
    business_metrics.append((threshold, rejection_rate, bad_debt_rate))

business_df = pd.DataFrame(business_metrics, columns=['Threshold', 'Rejection Rate', 'Bad Debt Rate'])

# Visualize business metrics
plt.figure(figsize=(10, 6))
plt.plot(business_df['Threshold'], business_df['Rejection Rate'], label='Rejection Rate')
plt.plot(business_df['Threshold'], business_df['Bad Debt Rate'], label='Bad Debt Rate')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('Business Metrics at Different Thresholds')
plt.legend()
plt.grid(True)
plt.show()

# Select a threshold based on business objectives
# For example, if we want the bad debt rate to be no more than 5%
target_bad_debt = 0.05
valid_thresholds = business_df[business_df['Bad Debt Rate'] <= target_bad_debt]
if not valid_thresholds.empty:
    business_threshold = valid_thresholds.iloc[np.argmin(valid_thresholds['Rejection Rate'])]['Threshold']
    print(f"To keep the bad debt rate below 5%, it is recommended to use a threshold of: {business_threshold:.2f}")
    print(f"Rejection Rate at this threshold: {valid_thresholds.iloc[np.argmin(valid_thresholds['Rejection Rate'])]['Rejection Rate']:.2f}")
    print(f"Bad Debt Rate at this threshold: {valid_thresholds.iloc[np.argmin(valid_thresholds['Rejection Rate'])]['Bad Debt Rate']:.2f}")
else:
    print("No threshold found that meets the target bad debt rate.")
```

## Advanced challenge

If you have already completed the basic tasks, you can try the following advanced challenges:

1. **Cost-sensitive learning** : Consider the business costs of different types of errors and build cost-sensitive models
2. **Feature Engineering Advanced** : Create interactive features, polynomial features or features based on domain knowledge
3. **Advanced Model interpretation** : Use tools such as LIME or SHAP to provide interpretation of individual predictions
4. **Fairness analysis** : Evaluate the model's performance on different demographic groups to detect and mitigate potential bias
5. **Deployment considerations** : Design model monitoring and update strategies to deal with concept drift

## Summary and Reflections

Through this project, we learned how to build a credit risk assessment model, address class imbalance issues, and translate model results into business decisions. Credit risk assessment is a complex task that requires balancing multiple factors, including prediction accuracy, business objectives, and ethical considerations.

In practical applications, credit risk models need to be updated and monitored regularly to adapt to the ever-changing economic environment and customer behavior. Additionally, the fairness and transparency of the model are important considerations to ensure that lending decisions do not unfairly impact specific groups.

### Questions for Reflection

1. How can we balance the trade-off between prediction performance and interpretability in credit risk assessment?  
2. How do different sampling methods affect the model's ability to predict the minority class (high-risk customers)?  
3. How can the credit risk model be integrated with business processes to serve as an effective decision support tool?

<div class="practice-link">
  <a href="/projects/clustering/customer-segmentation.html" class="button">Next Module: Customer Segmentation Project</a>
</div>
