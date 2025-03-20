<template>
  <div class="resource-container">
    <div class="resource-grid">
      <div v-for="(doc, index) in docList" :key="index" class="resource-card">
        <div class="resource-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" class="icon-doc">
            <path d="M365.3 93.38l-74.63-74.64C278.6 6.742 262.3 0 245.4 0H64C28.65 0 0 28.65 0 64l.0065 384c0 35.34 28.65 64 64 64H320c35.2 0 64-28.8 64-64V138.6C384 121.7 377.3 105.4 365.3 93.38zM336 448c0 8.836-7.164 16-16 16H64.02c-8.838 0-16-7.164-16-16L48 64.13c0-8.836 7.164-16 16-16h160L224 128c0 17.67 14.33 32 32 32h79.1V448z"/>
          </svg>
        </div>
        <div class="resource-info">
          <h3>{{ doc.title }}</h3>
          <p>{{ doc.description }}</p>
          <a :href="doc.link" class="download-btn" download>
            下载
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const docList = ref([
  {
    title: '数据预处理实践任务作业',
    description: '包含数据清洗和特征工程的实践任务',
    link: encodeURI('/resources/docs/П1.docx')
  },
  {
    title: 'SVM分类器实现任务作业',
    description: '基于Python的SVM分类器实现',
    link: encodeURI('/resources/docs/П2.docx')
  },
  {
    title: '朴素贝叶斯实现任务',
    description: '垃圾邮件分类器的实现指南',
    link: encodeURI('/resources/docs/П3.docx')
  },
  {
    title: '决策树模型实践',
    description: '使用决策树解决分类问题',
    link: encodeURI('/resources/docs/П4.docx')
  },
  {
    title: 'K-Means聚类分析',
    description: '客户分群案例实践',
    link: encodeURI('/resources/docs/П5.docx')
  },
  {
    title: '回归分析实践',
    description: '预测模型的构建与评估',
    link: encodeURI('/resources/docs/П6.docx')
  },
  {
    title: '综合项目实践',
    description: '数据挖掘方法的综合应用',
    link: encodeURI('/resources/docs/П7.docx')
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

.icon-doc {
  fill: #2B579A;
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