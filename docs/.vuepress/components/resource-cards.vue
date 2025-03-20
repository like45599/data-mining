<template>
  <div class="resource-cards">
    <div class="resource-card" v-for="(resource, index) in resources" :key="index">
      <div class="resource-card__type" :class="resource.type">{{ resource.typeText }}</div>
      <h3 class="resource-card__title">{{ resource.title }}</h3>
      <p class="resource-card__description">{{ resource.description }}</p>
      <a :href="resource.link" class="resource-card__link">查看详情</a>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      resources: [
        {
          type: 'tutorial',
          typeText: '教程',
          title: '数据预处理实战指南',
          description: '学习如何处理缺失值、异常值和不一致数据，为分析做好准备。',
          link: '/core/preprocessing/data-presentation.html'
        },
        {
          type: 'project',
          typeText: '项目',
          title: '泰坦尼克号生存预测',
          description: '通过分类算法预测乘客生存概率，实践特征工程和模型评估。',
          link: '/projects/classification/titanic.html'
        },
        {
          type: 'tool',
          typeText: '工具',
          title: '数据挖掘工具集',
          description: '精选的数据挖掘工具和库，帮助你高效完成分析任务。',
          link: '/resources/tools.html'
        }
      ]
    }
  }
}
</script>

<style scoped>
.resource-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.resource-card {
  background-color: var(--bg-color);
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: var(--card-shadow);
  position: relative;
  overflow: hidden;
}

.resource-card__type {
  position: absolute;
  top: 0;
  right: 0;
  padding: 0.3rem 0.8rem;
  font-size: 0.8rem;
  color: white;
  border-bottom-left-radius: 8px;
}

.resource-card__type.tutorial {
  background-color: #1890ff;
}

.resource-card__type.project {
  background-color: #52c41a;
}

.resource-card__type.tool {
  background-color: #722ed1;
}

.resource-card__title {
  margin-top: 0.5rem;
  font-size: 1.2rem;
}

.resource-card__description {
  color: #666;
  margin-bottom: 1rem;
}

.resource-card__link {
  color: var(--theme-color);
  text-decoration: none;
  font-weight: 500;
}

.resource-card__link:hover {
  text-decoration: underline;
}
</style> 