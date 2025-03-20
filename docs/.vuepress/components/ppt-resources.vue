<template>
  <div class="resource-container">
    <div class="resource-grid">
      <div v-for="(ppt, index) in pptList" :key="index" class="resource-card">
        <div class="resource-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" class="icon-ppt">
            <path d="M256 0v128h128L256 0zM224 128L224 0H48C21.49 0 0 21.49 0 48v416C0 490.5 21.49 512 48 512h288c26.51 0 48-21.49 48-48V160h-127.1C238.3 160 224 145.7 224 128z"/>
          </svg>
        </div>
        <div class="resource-info">
          <h3>{{ ppt.title }}</h3>
          <p>{{ ppt.description }}</p>
          <a :href="ppt.link" class="download-btn" download>
            下载
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const pptList = ref([
  {
    title: 'L1 数据表示与预处理',
    description: '介绍数据类型、特征工程、数据清洗等基础知识，包含pandas实战示例',
    link: encodeURI('/resources/ppt/L1 Data presentation.pptx'),
    keywords: ['数据类型', '特征工程', 'pandas', '数据清洗']
  },
  {
    title: 'L2 缺失值处理',
    description: '详解各种缺失值处理策略，包含实际案例分析和最佳实践指南',
    link: encodeURI('/resources/ppt/L2 Restoring missing values.pptx'),
    keywords: ['缺失值类型', '填充策略', '删除策略', '高级处理方法']
  },
  {
    title: 'L3 分类与SVM',
    description: 'SVM分类器原理与应用',
    link: encodeURI('/resources/ppt/L3 Classification. SVM.pptx')
  },
  {
    title: 'L4 垃圾邮件过滤',
    description: '基于朴素贝叶斯的文本分类',
    link: encodeURI('/resources/ppt/L4 Spam filters. Naive Baies.pptx')
  },
  {
    title: 'L5 决策树',
    description: '决策树算法与应用',
    link: encodeURI('/resources/ppt/L5 Decision trees.pptx')
  },
  {
    title: 'L6 聚类分析',
    description: 'K-Means聚类算法',
    link: encodeURI('/resources/ppt/L6  Clusterization. K-Means.pptx')
  },
  {
    title: 'L7 预测与回归',
    description: '回归分析方法',
    link: encodeURI('/resources/ppt/L7 Prediction. Regression.pptx')
  }
])
</script>

<style scoped>
.resource-container {
  padding: 2rem;
}

.resource-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2rem;
}

.resource-card {
  border: 1px solid #eaecef;
  border-radius: 8px;
  padding: 1.5rem;
  transition: all 0.3s ease;
  display: flex;
  align-items: start;
  gap: 1rem;
}

.resource-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.resource-icon {
  width: 40px;
  height: 40px;
  flex-shrink: 0;
}

.icon-ppt {
  fill: #D04423;
}

.resource-info {
  flex-grow: 1;
}

.resource-info h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
}

.resource-info p {
  margin: 0 0 1rem 0;
  color: #666;
  font-size: 0.9rem;
}

.download-btn {
  display: inline-block;
  padding: 0.5rem 1rem;
  background-color: #3eaf7c;
  color: white;
  text-decoration: none;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.download-btn:hover {
  background-color: #2c8f6c;
}

@media (max-width: 719px) {
  .resource-grid {
    grid-template-columns: 1fr;
  }
  
  .resource-container {
    padding: 1rem;
  }
}
</style> 