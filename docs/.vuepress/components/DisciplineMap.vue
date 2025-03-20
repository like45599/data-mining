<template>
  <div class="discipline-map">
    <div class="discipline-center">
      <div class="discipline-icon">🔍</div>
      <div class="discipline-name">{{ isEnglish ? 'Data Mining' : '数据挖掘' }}</div>
    </div>
    
    <div class="discipline-connections">
      <div class="discipline-item" v-for="(discipline, index) in disciplines" :key="index">
        <div class="discipline-icon">{{ discipline.icon }}</div>
        <div class="discipline-name">{{ discipline.name }}</div>
        <div class="discipline-desc">{{ discipline.description }}</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DisciplineMap',
  computed: {
    isEnglish() {
      return this.$lang === 'en-US';
    },
    
    disciplines() {
      if (this.isEnglish) {
        return [
          {
            icon: '📊',
            name: 'Statistics',
            description: 'Provides theoretical foundation and methodology for data analysis'
          },
          {
            icon: '🧠',
            name: 'Machine Learning',
            description: 'Provides algorithms and models for automated learning and prediction'
          },
          {
            icon: '💾',
            name: 'Database Technology',
            description: 'Provides methods for data storage, querying and management'
          },
          {
            icon: '📈',
            name: 'Business Intelligence',
            description: 'Applies data mining results to support business decisions'
          },
          {
            icon: '🔮',
            name: 'Artificial Intelligence',
            description: 'Provides intelligent systems and complex problem-solving methods'
          },
          {
            icon: '🖥️',
            name: 'Computer Science',
            description: 'Provides algorithm design and computational foundations'
          }
        ];
      } else {
        return [
          {
            icon: '📊',
            name: '统计学',
            description: '提供数据分析的理论基础和方法论'
          },
          {
            icon: '🧠',
            name: '机器学习',
            description: '提供自动化学习和预测的算法和模型'
          },
          {
            icon: '💾',
            name: '数据库技术',
            description: '提供数据存储、查询和管理的方法'
          },
          {
            icon: '📈',
            name: '商业智能',
            description: '应用数据挖掘结果支持业务决策'
          },
          {
            icon: '🔮',
            name: '人工智能',
            description: '提供智能系统和复杂问题求解方法'
          },
          {
            icon: '🖥️',
            name: '计算机科学',
            description: '提供算法设计和计算基础'
          }
        ];
      }
    }
  }
}
</script>

<style scoped>
.discipline-map {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 30px;
  margin: 20px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.discipline-center {
  background-color: #3498db;
  color: white;
  width: 150px;
  height: 150px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin: 0 auto 40px auto;
  box-shadow: 0 4px 10px rgba(52,152,219,0.3);
}

.discipline-center .discipline-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.discipline-center .discipline-name {
  font-weight: bold;
  font-size: 18px;
}

.discipline-connections {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 20px;
}

.discipline-item {
  background-color: white;
  border-left: 4px solid #3498db;
  border-radius: 5px;
  padding: 15px;
  width: calc(33% - 20px);
  min-width: 200px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: transform 0.3s, box-shadow 0.3s;
}

.discipline-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.discipline-item .discipline-icon {
  font-size: 24px;
  margin-bottom: 10px;
}

.discipline-item .discipline-name {
  font-weight: bold;
  margin-bottom: 8px;
  color: #2c3e50;
}

.discipline-item .discipline-desc {
  font-size: 14px;
  color: #7f8c8d;
  line-height: 1.4;
}

@media (max-width: 768px) {
  .discipline-item {
    width: 100%;
  }
}
</style> 