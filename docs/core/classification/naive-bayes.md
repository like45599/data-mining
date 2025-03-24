# 朴素贝叶斯算法

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解贝叶斯定理及其在分类中的应用</li>
      <li>掌握朴素贝叶斯算法的基本原理</li>
      <li>学习不同类型的朴素贝叶斯模型</li>
      <li>实践朴素贝叶斯在文本分类中的应用</li>
    </ul>
  </div>
</div>

## 贝叶斯定理基础

朴素贝叶斯算法基于贝叶斯定理，这是一个描述事件条件概率的数学公式。

### 贝叶斯定理

贝叶斯定理表示为：

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

其中：
- $P(A|B)$ 是已知B发生后A发生的条件概率（后验概率）
- $P(B|A)$ 是已知A发生后B发生的条件概率（似然概率）
- $P(A)$ 是A发生的概率（先验概率）
- $P(B)$ 是B发生的概率（边缘概率）

### 在分类问题中的应用

在分类问题中，我们希望计算给定特征$X$时，样本属于类别$y$的概率$P(y|X)$：

$$P(y|X) = \frac{P(X|y) \times P(y)}{P(X)}$$

其中：
- $P(y|X)$ 是给定特征X时类别为y的概率（我们要求的目标）
- $P(X|y)$ 是类别为y时观察到特征X的概率
- $P(y)$ 是类别y的先验概率
- $P(X)$ 是特征X出现的概率（对所有类别来说是常数）

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>贝叶斯定理以英国数学家托马斯·贝叶斯(Thomas Bayes, 1702-1761)命名。有趣的是，贝叶斯本人并未发表这个定理，它是在他去世后由理查德·普莱斯(Richard Price)整理并发表的。贝叶斯定理不仅是机器学习的基础，也在统计学、医学诊断、法律推理等多个领域有广泛应用。</p>
  </div>
</div>

## 朴素贝叶斯算法原理

### "朴素"假设

朴素贝叶斯之所以"朴素"，是因为它做了一个简化假设：**所有特征之间相互独立**。这意味着：

$$P(X|y) = P(x_1|y) \times P(x_2|y) \times ... \times P(x_n|y)$$

其中$x_1, x_2, ..., x_n$是特征向量$X$的各个特征。

这个假设虽然在实际中很少完全成立，但朴素贝叶斯在许多实际问题中仍然表现良好。

### 分类决策

朴素贝叶斯分类器选择具有最大后验概率的类别：

$$\hat{y} = \arg\max_y P(y|X) = \arg\max_y \frac{P(X|y)P(y)}{P(X)}$$

由于$P(X)$对所有类别都相同，可以简化为：

$$\hat{y} = \arg\max_y P(X|y)P(y) = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)$$

<div class="visualization-container">
  <div class="visualization-title">朴素贝叶斯分类过程</div>
  <div class="visualization-content">
    <img src="/images/naive_bayes_process.svg" alt="朴素贝叶斯分类过程">
  </div>
  <div class="visualization-caption">
    图: 朴素贝叶斯分类过程。从训练数据中学习先验概率和条件概率，然后用这些概率计算新样本的后验概率，选择后验概率最大的类别作为预测结果。
  </div>
</div>

## 朴素贝叶斯的变体

根据特征的分布假设，朴素贝叶斯有几种主要变体：

### 1. 高斯朴素贝叶斯

假设特征服从高斯分布，适用于连续型特征：

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

其中$\mu_y$和$\sigma_y^2$分别是类别$y$下特征$x_i$的均值和方差。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练高斯朴素贝叶斯模型
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 预测
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
```

  </div>
</div>

### 2. 多项式朴素贝叶斯

假设特征是离散的，服从多项式分布，适用于文本分类等计数数据：

$$P(x_i|y) = \frac{n_{yi} + \alpha}{n_y + \alpha n}$$

其中：
- $n_{yi}$ 是类别$y$中特征$i$的出现次数
- $n_y$ 是类别$y$中所有特征的出现次数总和
- $\alpha$ 是平滑参数（拉普拉斯平滑）
- $n$ 是特征的总数

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 示例文本数据
texts = [
    'I love this movie', 'This movie is great', 'The acting was amazing',
    'I hated this film', 'Terrible movie', 'The worst film I have seen',
    'The plot was interesting', 'I enjoyed the story', 'Great characters'
]
labels = [1, 1, 1, 0, 0, 0, 1, 1, 1]  # 1=正面评价, 0=负面评价

# 文本转换为特征向量
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 创建并训练多项式朴素贝叶斯模型
mnb = MultinomialNB(alpha=1.0)  # alpha是拉普拉斯平滑参数
mnb.fit(X_train, y_train)

# 预测
y_pred = mnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 查看特征重要性
feature_names = vectorizer.get_feature_names_out()
for class_idx in range(len(mnb.classes_)):
    top_features = sorted(zip(mnb.feature_log_prob_[class_idx], feature_names), reverse=True)[:5]
    print(f"类别 {mnb.classes_[class_idx]} 的前5个重要词: {[word for _, word in top_features]}")
```

  </div>
</div>

### 3. 伯努利朴素贝叶斯

假设特征是二元的（0或1），服从伯努利分布，适用于文档分类等二元特征：

$$P(x_i|y) = P(i|y)^{x_i} \times (1-P(i|y))^{(1-x_i)}$$

其中$P(i|y)$是类别$y$中特征$i$出现的概率。

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 使用与多项式朴素贝叶斯相同的示例数据
# 但使用二元特征（词是否出现，而非出现次数）
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 创建并训练伯努利朴素贝叶斯模型
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)

# 预测
y_pred = bnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
```

  </div>
</div>

## 朴素贝叶斯在文本分类中的应用

朴素贝叶斯是文本分类（如垃圾邮件过滤）的经典算法，下面是一个完整的垃圾邮件分类示例：

<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据（示例，实际应用中需要替换为真实数据）
# 假设数据格式为：邮件内容和标签（1=垃圾邮件，0=正常邮件）
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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42
)

# 创建处理流水线
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('classifier', MultinomialNB(alpha=0.1))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)

# 评估
print(classification_report(y_test, y_pred, target_names=['正常邮件', '垃圾邮件']))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 查看重要特征
tfidf = pipeline.named_steps['tfidf']
nb = pipeline.named_steps['classifier']
feature_names = tfidf.get_feature_names_out()

# 获取垃圾邮件类别的高概率词
spam_idx = np.where(nb.classes_ == 1)[0][0]
top_spam_features = sorted(zip(nb.feature_log_prob_[spam_idx], feature_names), reverse=True)[:10]
print("\n垃圾邮件中的高概率词:")
for prob, word in top_spam_features:
    print(f"{word}: {np.exp(prob):.4f}")

# 测试新邮件
new_emails = [
    "Congratulations! You've been selected for a free cruise.",
    "Please submit your timesheet by end of day."
]
predictions = pipeline.predict(new_emails)
for email, pred in zip(new_emails, predictions):
    print(f"\n邮件: {email}")
    print(f"预测: {'垃圾邮件' if pred == 1 else '正常邮件'}")
```

  </div>
</div>

## 朴素贝叶斯的优缺点

### 优点

1. **计算效率高**：训练和预测速度快，适合大规模数据集
2. **对小数据集效果好**：即使训练样本较少也能获得不错的性能
3. **处理高维数据能力强**：特别适合文本分类等高维稀疏数据
4. **易于实现和理解**：算法简单直观，容易解释

### 缺点

1. **特征独立性假设**：现实中特征往往不是完全独立的
2. **零概率问题**：需要使用平滑技术处理训练集中未出现的特征
3. **对特征权重不敏感**：无法直接处理一个特征比另一个更重要的情况
4. **对数值特征的建模不够精确**：高斯假设可能不符合实际分布

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>忽略特征独立性假设</strong>：在特征高度相关的情况下未进行特征选择或降维</li>
      <li><strong>忽略平滑处理</strong>：未设置适当的alpha值处理零概率问题</li>
      <li><strong>错误选择变体</strong>：对连续数据使用多项式朴素贝叶斯或对文本数据使用高斯朴素贝叶斯</li>
      <li><strong>过度依赖概率输出</strong>：朴素贝叶斯的概率估计通常不够准确，不应过度依赖其概率值</li>
    </ul>
  </div>
</div>

## 小结与思考

朴素贝叶斯是一种简单而强大的分类算法，特别适合文本分类等高维数据问题。尽管其假设在实际中很少完全成立，但由于其简单性和计算效率，仍然是许多应用的首选算法。

### 关键要点回顾

- 朴素贝叶斯基于贝叶斯定理和特征独立性假设
- 主要变体包括高斯、多项式和伯努利朴素贝叶斯
- 在文本分类等任务中表现出色
- 需要使用平滑技术处理零概率问题

### 思考问题

1. 为什么朴素贝叶斯在特征独立性假设不成立的情况下仍能表现良好？
2. 在什么情况下应该选择朴素贝叶斯而非其他分类算法？
3. 如何改进朴素贝叶斯以处理特征间的依赖关系？

<BackToPath />

<div class="practice-link">
  <a href="/projects/classification.html" class="button">前往实践项目</a>
</div> 