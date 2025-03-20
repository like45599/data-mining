<template>
  <div class="features-container">
    <div class="feature-card" v-for="(feature, index) in features" :key="index">
      <div class="feature-icon">
        <i :class="feature.icon"></i>
      </div>
      <h3>{{ feature.title }}</h3>
      <p>{{ feature.description }}</p>
      <a :href="feature.link" class="feature-link">了解更多</a>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      features: [
        {
          title: '数据预处理',
          description: '学习如何清洗、转换和准备数据，为后续分析奠定基础。',
          icon: 'icon-data',
          link: '/core/preprocessing/data-presentation.html'
        },
        {
          title: '分类算法',
          description: '掌握SVM、朴素贝叶斯和决策树等主要分类方法。',
          icon: 'icon-classify',
          link: '/core/classification/svm.html'
        },
        {
          title: '聚类分析',
          description: '探索K-Means等无监督学习方法，发现数据中的自然分组。',
          icon: 'icon-cluster',
          link: '/core/clustering/kmeans.html'
        },
        {
          title: '预测与回归',
          description: '学习如何使用线性和非线性回归方法进行预测分析。',
          icon: 'icon-predict',
          link: '/core/regression/linear-regression.html'
        }
      ]
    }
  }
}
</script>

<style scoped>
.features-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
  margin: 2rem 0;
}

.feature-card {
  background-color: var(--bg-color);
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: var(--card-shadow);
  transition: transform 0.3s, box-shadow 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.feature-icon {
  font-size: 2.5rem;
  color: var(--theme-color);
  margin-bottom: 1rem;
}

.feature-card h3 {
  margin-top: 0;
  font-size: 1.4rem;
  color: var(--text-color);
}

.feature-link {
  display: inline-block;
  margin-top: 1rem;
  color: var(--theme-color);
  text-decoration: none;
  font-weight: 500;
}

.feature-link:hover {
  text-decoration: underline;
}
</style> 