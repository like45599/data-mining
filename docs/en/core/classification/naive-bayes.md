# Naive Bayes Algorithm

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand Bayes' theorem and its application in classification</li>
      <li>Master the basic principles of the Naive Bayes algorithm</li>
      <li>Learn about different types of Naive Bayes models</li>
      <li>Practice applying Naive Bayes to text classification</li>
    </ul>
  </div>
</div>

## Basics of Bayes' Theorem

The Naive Bayes algorithm is based on Bayes' theorem, a mathematical formula that describes conditional probabilities.

### Bayes' Theorem

Bayes' theorem is expressed as:

$$
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
$$

where:
- \( P(A|B) \) is the conditional probability of \( A \) given that \( B \) has occurred (posterior probability)
- \( P(B|A) \) is the conditional probability of \( B \) given that \( A \) has occurred (likelihood)
- \( P(A) \) is the probability of \( A \) (prior probability)
- \( P(B) \) is the probability of \( B \) (marginal probability)

### Application in Classification Problems

In classification, we want to compute the probability that a sample belongs to class \( y \) given the features \( X \):

$$
P(y|X) = \frac{P(X|y) \times P(y)}{P(X)}
$$

where:
- \( P(y|X) \) is the probability of class \( y \) given the features \( X \) (the target we want to estimate)
- \( P(X|y) \) is the probability of observing features \( X \) given class \( y \)
- \( P(y) \) is the prior probability of class \( y \)
- \( P(X) \) is the probability of the features \( X \) (a constant for all classes)

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Bayes' theorem is named after the British mathematician Thomas Bayes (1702-1761). Interestingly, Bayes himself never published the theorem—it was compiled and published posthumously by Richard Price. Bayes' theorem is not only fundamental in machine learning but also finds wide applications in statistics, medical diagnosis, legal reasoning, and more.</p>
  </div>
</div>

## Principles of the Naive Bayes Algorithm

### The "Naive" Assumption

The reason for calling the algorithm "naive" is because it makes a simplifying assumption: **all features are independent of each other**. This means:

$$
P(X|y) = P(x_1|y) \times P(x_2|y) \times \ldots \times P(x_n|y)
$$

where \( x_1, x_2, \ldots, x_n \) are the individual features of the feature vector \( X \).

Although this assumption rarely holds true in practice, Naive Bayes often performs well on many real-world problems.

### Classification Decision

The Naive Bayes classifier selects the class with the highest posterior probability:

$$
\hat{y} = \arg\max_y P(y|X) = \arg\max_y \frac{P(X|y)P(y)}{P(X)}
$$

Since \( P(X) \) is the same for all classes, this simplifies to:

$$
\hat{y} = \arg\max_y P(X|y)P(y) = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)
$$

<div class="visualization-container">
  <div class="visualization-title">Naive Bayes Classification Process</div>
  <div class="visualization-content">
    <img src="/images/naive_bayes_process.svg" alt="Naive Bayes Classification Process">
  </div>
  <div class="visualization-caption">
    Figure: The Naive Bayes classification process. Prior and conditional probabilities are learned from the training data, then used to compute the posterior probability for a new sample, with the class having the highest posterior chosen as the prediction.
  </div>
</div>

## Variants of Naive Bayes

Depending on the assumed distribution of the features, there are several main variants of Naive Bayes:

### 1. Gaussian Naive Bayes

Assumes that the features follow a Gaussian (normal) distribution, making it suitable for continuous features:

$$
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)
$$

where \( \mu_y \) and \( \sigma_y^2 \) are the mean and variance of feature \( x_i \) for class \( y \), respectively.

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

  </div>
</div>

### 2. Multinomial Naive Bayes

Assumes that the features are discrete and follow a multinomial distribution, making it suitable for text classification or count data:

$$
P(x_i|y) = \frac{n_{yi} + \alpha}{n_y + \alpha n}
$$

where:
- \( n_{yi} \) is the count of feature \( i \) in class \( y \)
- \( n_y \) is the total count of all features in class \( y \)
- \( \alpha \) is the smoothing parameter (Laplace smoothing)
- \( n \) is the total number of features

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example text data
texts = [
    'I love this movie', 'This movie is great', 'The acting was amazing',
    'I hated this film', 'Terrible movie', 'The worst film I have seen',
    'The plot was interesting', 'I enjoyed the story', 'Great characters'
]
labels = [1, 1, 1, 0, 0, 0, 1, 1, 1]  # 1 = positive review, 0 = negative review

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Create and train the Multinomial Naive Bayes model
mnb = MultinomialNB(alpha=1.0)  # alpha is the Laplace smoothing parameter
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display feature importance
feature_names = vectorizer.get_feature_names_out()
for class_idx in range(len(mnb.classes_)):
    top_features = sorted(zip(mnb.feature_log_prob_[class_idx], feature_names), reverse=True)[:5]
    print(f"Top 5 important words for class {mnb.classes_[class_idx]}: {[word for _, word in top_features]}")
```

  </div>
</div>

### 3. Bernoulli Naive Bayes

Assumes that the features are binary (0 or 1) and follow a Bernoulli distribution, making it suitable for document classification with binary features:

$$
P(x_i|y) = P(i|y)^{x_i} \times (1-P(i|y))^{(1-x_i)}
$$

where \( P(i|y) \) is the probability of feature \( i \) being present in class \( y \).

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Use the same example data as for Multinomial Naive Bayes,
# but with binary features (indicating word presence rather than counts)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Create and train the Bernoulli Naive Bayes model
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)

# Make predictions
y_pred = bnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

  </div>
</div>

## Application of Naive Bayes in Text Classification

Naive Bayes is a classic algorithm for text classification (such as spam filtering). Below is a complete example of spam email classification:

<div class="code-example">
  <div class="code-example__title">Code Example</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load data (example; replace with actual data in practice)
# Assume the data format consists of email content and labels (1 = spam, 0 = non-spam)
emails = [
    "Get rich quick! Guaranteed money in just one week.",
    "Meeting scheduled for tomorrow at 10 AM.",
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "Please review the quarterly report before Friday.",
    "URGENT: Your account has been compromised. Verify now!",
    "Reminder: Team lunch at noon today.",
    "Free vacation! Limited time offer. Act now!",
    "The project deadline has been extended to next Monday."
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42
)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display important features
tfidf = pipeline.named_steps['tfidf']
nb = pipeline.named_steps['classifier']
feature_names = tfidf.get_feature_names_out()

# Get the top probability words for the spam class
spam_idx = np.where(nb.classes_ == 1)[0][0]
top_spam_features = sorted(zip(nb.feature_log_prob_[spam_idx], feature_names), reverse=True)[:10]
print("\nTop words in spam emails:")
for prob, word in top_spam_features:
    print(f"{word}: {np.exp(prob):.4f}")

# Test new emails
new_emails = [
    "Congratulations! You've been selected for a free cruise.",
    "Please submit your timesheet by end of day."
]
predictions = pipeline.predict(new_emails)
for email, pred in zip(new_emails, predictions):
    print(f"\nEmail: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Non-Spam'}")
```

  </div>
</div>

## Advantages and Disadvantages of Naive Bayes

### Advantages

1. **High Computational Efficiency**: Fast training and prediction, making it suitable for large datasets.
2. **Effective with Small Datasets**: Performs well even with a limited number of training samples.
3. **Handles High-Dimensional Data Well**: Particularly effective for text classification and other sparse, high-dimensional data.
4. **Easy to Implement and Understand**: The algorithm is simple and intuitive, which makes it easy to interpret.

### Disadvantages

1. **Feature Independence Assumption**: In reality, features are often correlated, which may require feature selection or dimensionality reduction.
2. **Zero Probability Problem**: Smoothing techniques must be used to handle features that did not appear in the training data.
3. **Insensitive to Feature Weights**: It cannot directly account for situations where one feature is more important than another.
4. **Less Accurate for Numerical Data Modeling**: The Gaussian assumption may not always accurately reflect the actual distribution.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>Common Misconceptions
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Ignoring the Independence Assumption</strong>: Failing to perform feature selection or dimensionality reduction when features are highly correlated.</li>
      <li><strong>Neglecting Smoothing</strong>: Not setting an appropriate alpha value to address the zero probability issue.</li>
      <li><strong>Incorrect Variant Selection</strong>: Using Multinomial Naive Bayes for continuous data or Gaussian Naive Bayes for text data.</li>
      <li><strong>Overreliance on Probability Outputs</strong>: Naive Bayes probability estimates are often not very accurate and should not be overly trusted.</li>
    </ul>
  </div>
</div>

## Summary and Reflections

Naive Bayes is a simple yet powerful classification algorithm, especially suitable for high-dimensional data problems like text classification. Although its assumptions rarely hold perfectly in practice, its simplicity and computational efficiency make it a popular choice in many applications.

### Key Takeaways

- Naive Bayes is based on Bayes' theorem and the assumption of feature independence.
- Its main variants include Gaussian, Multinomial, and Bernoulli Naive Bayes.
- It performs well in tasks such as text classification.
- Smoothing techniques are necessary to address the zero probability problem.

### Reflection Questions

1. Why does Naive Bayes perform well even when the feature independence assumption does not hold?
2. In what situations should one choose Naive Bayes over other classification algorithms?
3. How can Naive Bayes be improved to better handle dependencies between features?

<div class="practice-link">
  <a href="/projects/classification.html" class="button">Proceed to Practice Projects</a>
</div>
