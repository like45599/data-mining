
# Spam Filter

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Intermediate</li>
      <li><strong>Type</strong>: Text Classification</li>
      <!-- <li><strong>Estimated Time</strong>: 5-7 hours</li> -->
      <li><strong>Skills</strong>: Text Preprocessing, Feature Extraction, Naive Bayes Classification, Model Evaluation</li>
      <li><strong>Relevant Knowledge Module</strong>: <a href="/core/classification/svm.html">Classification Algorithms</a></li>
    </ul>
  </div>
</div>

## Project Background

Spam is a common issue in email systems, with billions of spam emails being sent daily, making up a large portion of global email traffic. These emails not only waste time and resources but may also contain malicious links or fraudulent content, posing security risks to users.

Automated spam filtering systems use machine learning algorithms to differentiate between normal and spam emails. These systems analyze the content of the email, sender information, and other features to learn to identify spam patterns.

In this project, we will build a spam filter based on the Naive Bayes algorithm, learning how to process text data and apply probabilistic classification methods.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Naive Bayes was one of the first machine learning algorithms used for spam filtering and is still widely used today. Despite its simplicity, it performs excellently in text classification tasks, especially when training data is limited. Modern spam filtering systems often combine multiple algorithms, but Naive Bayes remains an important component.</p>
  </div>
</div>

## Dataset Introduction

The dataset used in this project contains approximately 5,000 emails labeled as either "spam" or "ham" (normal emails). Each email includes the following information:

- **Content**: The full text of the email
- **Subject**: The subject line of the email
- **Sender**: The sender's email address
- **Date**: The date and time the email was sent
- **Label**: Labeled as "spam" or "ham"

The dataset has been preprocessed to remove sensitive information but retains the typical characteristics of spam emails.

## Project Goals

1. Implement text data preprocessing and feature extraction
2. Build a Naive Bayes-based spam classifier
3. Evaluate model performance and optimize parameters
4. Analyze the decision-making process of the model and identify which features are most important for classification
5. Build a simple spam filtering system

## Implementation Steps

### Step 1: Data Loading and Exploration

First, we load the data and perform preliminary exploration to understand the basic information about the data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load data
df = pd.read_csv('email_dataset.csv')

# View basic data information
print(df.info())
print(df.head())

# View label distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df)
plt.title('Email Type Distribution')
plt.show()

# View email length distribution
df['content_length'] = df['content'].apply(len)
df['subject_length'] = df['subject'].apply(len)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='content_length', hue='label', bins=50, kde=True)
plt.title('Email Content Length Distribution')
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='subject_length', hue='label', bins=50, kde=True)
plt.title('Email Subject Length Distribution')
plt.tight_layout()
plt.show()

# View common sender domains
df['sender_domain'] = df['sender'].apply(lambda x: x.split('@')[-1] if '@' in x else 'unknown')
top_domains = df.groupby(['sender_domain', 'label']).size().unstack().fillna(0)
top_domains['total'] = top_domains['spam'] + top_domains['ham']
top_domains = top_domains.sort_values('total', ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_domains[['spam', 'ham']].plot(kind='bar', stacked=True)
plt.title('Most Common Sender Domains')
plt.ylabel('Email Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 2: Text Preprocessing

Next, we preprocess the email content, which involves cleaning the text, removing stopwords, and performing stemming.

```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Apply preprocessing
df['cleaned_content'] = df['content'].apply(preprocess_text)
df['cleaned_subject'] = df['subject'].apply(preprocess_text)

# View preprocessed text examples
print("Original Text:")
print(df['content'].iloc[0])
print("\nPreprocessed:")
print(df['cleaned_content'].iloc[0])
```

### Step 3: Feature Extraction

Now, we extract features from the text using the TF-IDF vectorization method.

```python
# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Combine subject and content
df['text'] = df['cleaned_subject'] + ' ' + df['cleaned_content']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Use TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# View feature dimensions
print(f"Feature Dimensions: {X_train_tfidf.shape}")

# View some feature names
feature_names = vectorizer.get_feature_names_out()
print(f"Some Feature Names: {feature_names[:20]}")
```

### Step 4: Build Naive Bayes Classifier

Now, we use the Naive Bayes algorithm to build the spam classifier.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Train Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```

### Step 5: Model Optimization and Feature Analysis

Next, we optimize model parameters and analyze important features.

```python
from sklearn.model_selection import GridSearchCV

# Parameter optimization
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Output best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Use best model
best_nb = grid_search.best_estimator_
y_pred_best = best_nb.predict(X_test_tfidf)

# Evaluate optimized model
print("\nOptimized Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Analyze important features
def get_most_informative_features(vectorizer, classifier, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs_with_fnames = sorted(zip(classifier.coef_[0], feature_names))
    top_negative = coefs_with_fnames[:n]
    top_positive = coefs_with_fnames[:-(n+1):-1]
    return top_positive, top_negative

# Get important features for spam and ham
top_spam_features, top_ham_features = get_most_informative_features(vectorizer, best_nb)

# Visualize important features
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
y_pos = np.arange(len(top_spam_features))
plt.barh(y_pos, [x[0] for x in top_spam_features], align='center')
plt.yticks(y_pos, [x[1] for x in top_spam_features])
plt.title('Important Features for Spam')

plt.subplot(1, 2, 2)
y_pos = np.arange(len(top_ham_features))
plt.barh(y_pos, [abs(x[0]) for x in top_ham_features], align='center')
plt.yticks(y_pos, [x[1] for x in top_ham_features])
plt.title('Important Features for Ham')

plt.tight_layout()
plt.show()
```

### Step 6: Build a Simple Spam Filtering System

Finally, we build a simple spam filtering system that can classify new emails.

```python
def predict_email(email_content, email_subject='', threshold=0.5):
    # Preprocess
    cleaned_content = preprocess_text(email_content)
    cleaned_subject = preprocess_text(email_subject) if email_subject else ''
    combined_text = cleaned_subject + ' ' + cleaned_content
    
    # Vectorize
    email_tfidf = vectorizer.transform([combined_text])
    
    # Predict probability
    spam_prob = best_nb.predict_proba(email_tfidf)[0, 1]
    
    # Decide based on threshold
    is_spam = spam_prob > threshold
    
    return {
        'is_spam': bool(is_spam),
        'spam_probability': float(spam_prob),
        'prediction': 'Spam' if is_spam else 'Ham'
    }

# Test the system
test_emails = [
    {
        'subject': 'Meeting tomorrow',
        'content': 'Hi team, just a reminder that we have a meeting scheduled for tomorrow at 10am. Please prepare your weekly reports.'
    },
    {
        'subject': 'URGENT: Your account has been compromised',
        'content': 'Dear valued customer, your account has been compromised. Click here to verify your information and claim your $1000 reward immediately!'
    },
    {
        'subject': 'Free Viagra and Cialis',
        'content': 'Best prices on the market! Buy now and get 90% discount on all products. Limited time offer!'
    }
]

for i, email in enumerate(test_emails):
    result = predict_email(email['content'], email['subject'])
    print(f"Email {i+1}:")
    print(f"Subject: {email['subject']}")
    print(f"Content: {email['content'][:100]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Spam Probability: {result['spam_probability']:.4f}")
    print("-" * 80)
```

## Results Analysis

By implementing this project, we successfully built a spam filtering system that can effectively distinguish between normal and spam emails. The model achieved about 95% accuracy on the test set, indicating the effectiveness of our approach.

Analysis of important features showed that spam emails often contain words like "free," "offer," "money," and "discount," while normal emails contain more personal and work-related terms. This matches our intuition and verifies that the model has learned meaningful patterns.

Parameter optimization showed that adjusting the smoothing parameter of Naive Bayes can further improve model performance. The final system provides spam probability for new emails, allowing users to adjust the filtering threshold based on their needs.

## Advanced Challenges

If you've completed the basic tasks, try the following advanced challenges:

1. **Advanced Feature Engineering**: Experiment with n-gram features, part-of-speech tagging, or entity recognition.
2. **Model Comparison**: Compare Naive Bayes with SVM, random forests, and other classification algorithms.
3. **Online Learning**: Implement a system that continuously learns from user feedback.
4. **Multilingual Support**: Extend the system to support spam detection in multiple languages.
5. **Deployment**: Deploy the model as a web application or email client plugin.

## Summary and Reflection

Through this project, we learned how to process text data and apply the Naive Bayes algorithm to build a spam filter. Text classification is a fundamental task in natural language processing, and mastering these skills can be applied to sentiment analysis, topic classification, and other scenarios.

In practice, spam filtering systems need to be regularly updated to cope with new spam patterns. Spam senders also constantly adjust their strategies to evade filters, creating an "arms race." Therefore, real-world systems typically combine multiple techniques and regularly update the model.

### Reflection Questions

1. How do you balance precision and recall in spam filtering? What is the cost of misclassification?
2. Naive Bayes assumes that features are independent, but words in text are clearly not independent. Why does Naive Bayes still perform well in text classification?
3. How do you handle evasion techniques used by spam senders, such as deliberate misspellings or using images instead of text?

<div class="practice-link">
  <a href="/projects/classification/credit-risk.html" class="button">Next Project: Credit Risk Assessment</a>
</div>
