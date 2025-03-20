# 垃圾邮件过滤器

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：中级</li>
      <li><strong>类型</strong>：文本分类</li>
      <!-- <li><strong>预计时间</strong>：5-7小时</li> -->
      <li><strong>技能点</strong>：文本预处理、特征提取、朴素贝叶斯分类、模型评估</li>
      <li><strong>对应知识模块</strong>：<a href="/core/classification/svm.html">分类算法</a></li>
    </ul>
  </div>
</div>

## 项目背景

垃圾邮件是电子邮件系统中的一个普遍问题，每天有数十亿封垃圾邮件被发送，占全球电子邮件流量的很大一部分。这些邮件不仅浪费时间和资源，还可能包含恶意链接或欺诈内容，对用户造成安全风险。

自动垃圾邮件过滤系统使用机器学习算法来区分正常邮件和垃圾邮件，是现代电子邮件服务的核心组件。这些系统通过分析邮件内容、发件人信息和其他特征，学习识别垃圾邮件的模式。

在这个项目中，我们将构建一个基于朴素贝叶斯算法的垃圾邮件过滤器，学习如何处理文本数据并应用概率分类方法。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>朴素贝叶斯是最早用于垃圾邮件过滤的机器学习算法之一，至今仍被广泛使用。尽管算法简单，但在文本分类任务中表现出色，特别是在训练数据有限的情况下。现代垃圾邮件过滤系统通常结合多种算法，但朴素贝叶斯仍是其中的重要组成部分。</p>
  </div>
</div>

## 数据集介绍

本项目使用的数据集包含约5,000封标记为"垃圾邮件"或"正常邮件"的电子邮件。每封邮件包含以下信息：

- **邮件内容**：邮件的完整文本
- **主题**：邮件的主题行
- **发件人**：发件人的电子邮件地址
- **日期**：邮件发送的日期和时间
- **标签**：标记为"spam"（垃圾邮件）或"ham"（正常邮件）

数据集已经过预处理，移除了敏感信息，但保留了垃圾邮件的典型特征。

## 项目目标

1. 实现文本数据的预处理和特征提取
2. 构建基于朴素贝叶斯的垃圾邮件分类器
3. 评估模型性能并优化参数
4. 分析模型的决策过程，了解哪些特征对分类最重要
5. 构建一个简单的垃圾邮件过滤系统

## 实施步骤

### 步骤1：数据加载与探索

首先，我们加载数据并进行初步探索，了解数据的基本情况。

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

# 下载必要的NLTK资源
nltk.download('stopwords')
nltk.download('punkt')

# 加载数据
df = pd.read_csv('email_dataset.csv')

# 查看数据基本信息
print(df.info())
print(df.head())

# 查看标签分布
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df)
plt.title('邮件类型分布')
plt.show()

# 查看邮件长度分布
df['content_length'] = df['content'].apply(len)
df['subject_length'] = df['subject'].apply(len)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='content_length', hue='label', bins=50, kde=True)
plt.title('邮件内容长度分布')
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='subject_length', hue='label', bins=50, kde=True)
plt.title('邮件主题长度分布')
plt.tight_layout()
plt.show()

# 查看常见发件人域名
df['sender_domain'] = df['sender'].apply(lambda x: x.split('@')[-1] if '@' in x else 'unknown')
top_domains = df.groupby(['sender_domain', 'label']).size().unstack().fillna(0)
top_domains['total'] = top_domains['spam'] + top_domains['ham']
top_domains = top_domains.sort_values('total', ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_domains[['spam', 'ham']].plot(kind='bar', stacked=True)
plt.title('常见发件人域名')
plt.ylabel('邮件数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 步骤2：文本预处理

接下来，我们对邮件内容进行预处理，包括清洗文本、去除停用词和词干提取。

```python
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 移除标点符号
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # 移除数字
    text = re.sub(r'\d+', ' ', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 分词
    words = nltk.word_tokenize(text)
    
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 词干提取
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# 应用预处理
df['cleaned_content'] = df['content'].apply(preprocess_text)
df['cleaned_subject'] = df['subject'].apply(preprocess_text)

# 查看预处理后的文本示例
print("原始文本:")
print(df['content'].iloc[0])
print("\n预处理后:")
print(df['cleaned_content'].iloc[0])
```

### 步骤3：特征提取

现在，我们使用TF-IDF向量化方法从文本中提取特征。

```python
# 将标签转换为数值
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 组合主题和内容
df['text'] = df['cleaned_subject'] + ' ' + df['cleaned_content']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 使用TF-IDF向量化文本
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 查看特征维度
print(f"特征维度: {X_train_tfidf.shape}")

# 查看部分特征名称
feature_names = vectorizer.get_feature_names_out()
print(f"部分特征名称: {feature_names[:20]}")
```

### 步骤4：构建朴素贝叶斯分类器

现在，我们使用朴素贝叶斯算法构建垃圾邮件分类器。

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 训练朴素贝叶斯模型
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# 在测试集上预测
y_pred = nb_classifier.predict(X_test_tfidf)

# 评估模型性能
print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()
```

### 步骤5：模型优化与特征分析

接下来，我们优化模型参数并分析重要特征。

```python
from sklearn.model_selection import GridSearchCV

# 参数优化
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)

# 使用最佳参数的模型
best_nb = grid_search.best_estimator_
y_pred_best = best_nb.predict(X_test_tfidf)

# 评估优化后的模型
print("\n优化后的模型性能:")
print("准确率:", accuracy_score(y_test, y_pred_best))
print("\n分类报告:")
print(classification_report(y_test, y_pred_best))

# 分析重要特征
def get_most_informative_features(vectorizer, classifier, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs_with_fnames = sorted(zip(classifier.coef_[0], feature_names))
    top_negative = coefs_with_fnames[:n]
    top_positive = coefs_with_fnames[:-(n+1):-1]
    return top_positive, top_negative

# 获取垃圾邮件和正常邮件的重要特征
top_spam_features, top_ham_features = get_most_informative_features(vectorizer, best_nb)

# 可视化重要特征
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
y_pos = np.arange(len(top_spam_features))
plt.barh(y_pos, [x[0] for x in top_spam_features], align='center')
plt.yticks(y_pos, [x[1] for x in top_spam_features])
plt.title('垃圾邮件重要特征')

plt.subplot(1, 2, 2)
y_pos = np.arange(len(top_ham_features))
plt.barh(y_pos, [abs(x[0]) for x in top_ham_features], align='center')
plt.yticks(y_pos, [x[1] for x in top_ham_features])
plt.title('正常邮件重要特征')

plt.tight_layout()
plt.show()
```

### 步骤6：构建简单的垃圾邮件过滤系统

最后，我们构建一个简单的垃圾邮件过滤系统，可以对新邮件进行分类。

```python
def predict_email(email_content, email_subject='', threshold=0.5):
    # 预处理
    cleaned_content = preprocess_text(email_content)
    cleaned_subject = preprocess_text(email_subject) if email_subject else ''
    combined_text = cleaned_subject + ' ' + cleaned_content
    
    # 向量化
    email_tfidf = vectorizer.transform([combined_text])
    
    # 预测概率
    spam_prob = best_nb.predict_proba(email_tfidf)[0, 1]
    
    # 根据阈值判断
    is_spam = spam_prob > threshold
    
    return {
        'is_spam': bool(is_spam),
        'spam_probability': float(spam_prob),
        'prediction': 'Spam' if is_spam else 'Ham'
    }

# 测试系统
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

## 结果分析

通过实施这个项目，我们成功构建了一个垃圾邮件过滤系统，能够有效区分正常邮件和垃圾邮件。模型在测试集上取得了约95%的准确率，表明我们的方法是有效的。

分析重要特征发现，垃圾邮件通常包含"free"、"offer"、"money"、"discount"等词汇，而正常邮件则包含更多的个人交流和工作相关词汇。这与我们的直觉相符，也验证了模型学习到了有意义的模式。

参数优化显示，适当调整朴素贝叶斯的平滑参数可以进一步提高模型性能。最终的系统能够为新邮件提供垃圾邮件概率，使用户可以根据自己的需求调整过滤阈值。

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **特征工程进阶**：尝试使用n-gram特征、词性标注或实体识别等高级文本特征
2. **模型比较**：比较朴素贝叶斯与SVM、随机森林等其他分类算法的性能
3. **在线学习**：实现一个能够从用户反馈中不断学习的系统
4. **多语言支持**：扩展系统以支持多种语言的垃圾邮件检测
5. **部署应用**：将模型部署为Web应用或邮件客户端插件

## 小结与反思

通过这个项目，我们学习了如何处理文本数据并应用朴素贝叶斯算法构建垃圾邮件过滤器。文本分类是自然语言处理的基础任务，掌握这些技能可以应用于情感分析、主题分类等多种场景。

在实际应用中，垃圾邮件过滤系统需要不断更新以应对新的垃圾邮件模式。垃圾邮件发送者也在不断调整策略以逃避过滤，这形成了一种"军备竞赛"。因此，实际系统通常结合多种技术，并定期更新模型。

### 思考问题

1. 如何平衡垃圾邮件过滤的精确性和召回率？错误分类的成本是什么？
2. 朴素贝叶斯算法假设特征之间相互独立，但文本中的词汇显然不是独立的。为什么朴素贝叶斯在文本分类中仍然表现良好？
3. 如何处理垃圾邮件发送者使用的规避技术，如故意拼写错误、使用图片而非文本等？

<div class="practice-link">
  <a href="/projects/classification/credit-risk.html" class="button">下一个项目：信用风险评估</a>
</div> 