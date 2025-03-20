<template>
  <div class="learning-path">
    <div class="learning-path__timeline">
      <div 
        v-for="(step, index) in steps" 
        :key="index"
        class="learning-path__step"
        :class="{ 'active': activeStep === index, 'completed': index < activeStep }"
        @click="setActiveStep(index)"
      >
        <div class="learning-path__step-connector" v-if="index > 0"></div>
        <div class="learning-path__step-icon">
          <span class="learning-path__step-number">{{ index + 1 }}</span>
        </div>
        <div class="learning-path__step-content">
          <h3 class="learning-path__step-title">{{ step.title }}</h3>
          <p class="learning-path__step-description">{{ step.description }}</p>
        </div>
      </div>
    </div>
    
    <div class="learning-path__details" v-if="steps[activeStep]">
      <div class="learning-path__details-header">
        <h3>{{ steps[activeStep].title }} - {{ isEnglish ? 'Details' : '详细内容' }}</h3>
      </div>
      <div class="learning-path__details-body">
        <div class="learning-path__topics">
          <h4>{{ isEnglish ? 'Core Topics' : '核心主题' }}</h4>
          <ul>
            <li v-for="(topic, topicIndex) in steps[activeStep].topics" :key="topicIndex">
              <a :href="topic.link">{{ topic.title }}</a>
            </li>
          </ul>
        </div>
        <div class="learning-path__skills">
          <h4>{{ isEnglish ? 'Key Skills' : '关键技能' }}</h4>
          <div class="learning-path__skill-tags">
            <span 
              v-for="(skill, skillIndex) in steps[activeStep].skills" 
              :key="skillIndex"
              class="learning-path__skill-tag"
            >
              {{ skill }}
            </span>
          </div>
        </div>
      </div>
      <div class="learning-path__action" v-if="steps[activeStep].action">
        <a :href="steps[activeStep].action.link" class="learning-path__action-button">
          {{ steps[activeStep].action.text }}
        </a>
      </div>
    </div>
    
    <div class="learning-path__progress">
      <div class="learning-path__progress-bar">
        <div 
          class="learning-path__progress-fill" 
          :style="{ width: `${(activeStep / (steps.length - 1)) * 100}%` }"
        ></div>
      </div>
      <div class="learning-path__progress-text">
        完成度: {{ Math.round((activeStep / (steps.length - 1)) * 100) }}%
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'LearningPathVisualization',
  data() {
    return {
      activeStep: 0,
      isEnglish: this.$lang === 'en-US'
    };
  },
  computed: {
    steps() {
      if (this.isEnglish) {
        return [
          {
            title: 'Data Mining Basics',
            description: 'Understand the fundamental concepts and processes of data mining',
            topics: [
              { title: 'What is Data Mining', link: '/en/overview/definition.html' },
              { title: 'Data Mining Process', link: '/en/overview/process.html' },
              { title: 'Applications', link: '/en/overview/applications.html' }
            ],
            skills: [
              'Understanding data mining terminology',
              'Identifying suitable problems for data mining',
              'Recognizing the data mining process steps'
            ],
            action: {
              text: 'Start Learning Basics',
              link: '/en/overview/definition.html'
            }
          },
          {
            title: 'Data Preprocessing',
            description: 'Learn how to clean, transform and prepare data for analysis',
            topics: [
              { title: 'Data Representation', link: '/en/core/preprocessing/data-presentation.html' },
              { title: 'Missing Values', link: '/en/core/preprocessing/missing-values.html' },
              { title: 'Feature Engineering', link: '/en/core/preprocessing/feature-engineering.html' }
            ],
            skills: [
              'Data cleaning techniques',
              'Handling missing values',
              'Feature selection and transformation',
              'Data normalization and standardization'
            ],
            action: {
              text: 'Learn Data Preprocessing',
              link: '/en/core/preprocessing/data-presentation.html'
            }
          },
          {
            title: 'Classification Algorithms',
            description: 'Master various methods for predicting categorical outcomes',
            topics: [
              { title: 'Decision Trees', link: '/en/core/classification/decision-trees.html' },
              { title: 'Support Vector Machines', link: '/en/core/classification/svm.html' },
              { title: 'Naive Bayes', link: '/en/core/classification/naive-bayes.html' }
            ],
            skills: [
              'Building classification models',
              'Model evaluation and validation',
              'Hyperparameter tuning',
              'Ensemble methods'
            ],
            action: {
              text: 'Learn Classification Algorithms',
              link: '/en/core/classification/svm.html'
            }
          },
          {
            title: 'Clustering Analysis',
            description: 'Explore unsupervised learning methods to discover natural groupings',
            topics: [
              { title: 'K-Means Clustering', link: '/en/core/clustering/kmeans.html' },
              { title: 'Hierarchical Clustering', link: '/en/core/clustering/hierarchical.html' },
              { title: 'Evaluation Methods', link: '/en/core/clustering/evaluation.html' }
            ],
            skills: [
              'Identifying appropriate clustering algorithms',
              'Determining optimal number of clusters',
              'Interpreting clustering results',
              'Visualizing high-dimensional clusters'
            ],
            action: {
              text: 'Learn Clustering Analysis',
              link: '/en/core/clustering/kmeans.html'
            }
          },
          {
            title: 'Prediction and Regression',
            description: 'Learn techniques for predicting continuous values and time series',
            topics: [
              { title: 'Linear Regression', link: '/en/core/regression/linear-regression.html' },
              { title: 'Non-linear Models', link: '/en/core/regression/nonlinear-regression.html' },
              { title: 'Model Evaluation', link: '/en/core/regression/evaluation-metrics.html' }
            ],
            skills: [
              'Building regression models',
              'Feature selection for regression',
              'Time series forecasting',
              'Regression model evaluation'
            ],
            action: {
              text: 'Learn Prediction and Regression',
              link: '/en/core/regression/linear-regression.html'
            }
          },
          {
            title: 'Practice Project',
            description: 'Apply learned knowledge to real-world projects',
            topics: [
              { title: 'Titanic Survival Prediction', link: '/en/projects/classification/titanic.html' },
              { title: 'Customer Segmentation Analysis', link: '/en/projects/clustering/customer-segmentation.html' },
              { title: 'House Price Prediction', link: '/en/projects/regression/house-price.html' }
            ],
            skills: ['Project Practice', 'Comprehensive Application', 'Result Interpretation'],
            action: {
              text: 'Start Practice Project',
              link: '/en/projects/'
            }
          }
        ];
      } else {
        return [
          {
            title: '数据挖掘基础',
            description: '理解数据挖掘的基本概念和流程',
            topics: [
              { title: '什么是数据挖掘', link: '/overview/definition.html' },
              { title: '数据挖掘流程', link: '/overview/process.html' },
              { title: '应用场景', link: '/overview/applications.html' }
            ],
            skills: [
              '理解数据挖掘术语',
              '识别适合数据挖掘的问题',
              '认识数据挖掘流程步骤'
            ],
            action: {
              text: '开始学习基础知识',
              link: '/overview/definition.html'
            }
          },
          {
            title: '数据预处理',
            description: '学习如何清洗、转换和准备数据以进行分析',
            topics: [
              { title: '数据表示', link: '/core/preprocessing/data-presentation.html' },
              { title: '缺失值处理', link: '/core/preprocessing/missing-values.html' },
              { title: '特征工程', link: '/core/preprocessing/feature-engineering.html' }
            ],
            skills: [
              '数据清洗技术',
              '处理缺失值',
              '特征选择与转换',
              '数据归一化和标准化'
            ],
            action: {
              text: '学习数据预处理',
              link: '/core/preprocessing/data-presentation.html'
            }
          },
          {
            title: '分类算法',
            description: '掌握预测分类结果的各种方法',
            topics: [
              { title: '决策树', link: '/core/classification/decision-trees.html' },
              { title: '支持向量机', link: '/core/classification/svm.html' },
              { title: '朴素贝叶斯', link: '/core/classification/naive-bayes.html' }
            ],
            skills: [
              '构建分类模型',
              '模型评估和验证',
              '超参数调优',
              '集成方法'
            ],
            action: {
              text: '学习分类算法',
              link: '/core/classification/svm.html'
            }
          },
          {
            title: '聚类分析',
            description: '探索无监督学习方法以发现自然分组',
            topics: [
              { title: 'K-均值聚类', link: '/core/clustering/kmeans.html' },
              { title: '层次聚类', link: '/core/clustering/hierarchical.html' },
              { title: '评估方法', link: '/core/clustering/evaluation.html' }
            ],
            skills: [
              '识别适当的聚类算法',
              '确定最佳聚类数量',
              '解释聚类结果',
              '可视化高维聚类'
            ],
            action: {
              text: '学习聚类分析',
              link: '/core/clustering/kmeans.html'
            }
          },
          {
            title: '预测与回归',
            description: '学习预测连续值和时间序列的技术',
            topics: [
              { title: '线性回归', link: '/core/regression/linear-regression.html' },
              { title: '非线性模型', link: '/core/regression/nonlinear-regression.html' },
              { title: '模型评估', link: '/core/regression/evaluation-metrics.html' }
            ],
            skills: [
              '构建回归模型',
              '回归的特征选择',
              '时间序列预测',
              '回归模型评估'
            ],
            action: {
              text: '学习预测与回归',
              link: '/core/regression/linear-regression.html'
            }
          },
          {
            title: '实践项目',
            description: '通过实际项目巩固所学知识，提升实战能力',
            topics: [
              { title: '泰坦尼克号生存预测', link: '/projects/classification/titanic.html' },
              { title: '客户分群分析', link: '/projects/clustering/customer-segmentation.html' },
              { title: '房价预测', link: '/projects/regression/house-price.html' }
            ],
            skills: ['项目实战', '综合应用', '结果解释'],
            action: {
              text: '开始实践项目',
              link: '/projects/'
            }
          }
        ];
      }
    }
  },
  methods: {
    setActiveStep(index) {
      this.activeStep = index;
    }
  }
};
</script>

<style lang="scss" scoped>
.learning-path {
  margin: 2rem 0;
  
  &__timeline {
    display: flex;
    justify-content: space-between;
    position: relative;
    margin-bottom: 2rem;
    padding: 0 1rem;
    
    &:before {
      content: '';
      position: absolute;
      top: 2rem;
      left: 2rem;
      right: 2rem;
      height: 2px;
      background-color: #e0e0e0;
      z-index: 1;
    }
  }
  
  &__step {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: calc(100% / 6);
    cursor: pointer;
    
    &-icon {
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 50%;
      background-color: #f0f0f0;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 0.8rem;
      transition: all 0.3s ease;
      border: 2px solid #e0e0e0;
    }
    
    &-number {
      font-size: 1rem;
      font-weight: bold;
      color: #666;
    }
    
    &-content {
      text-align: center;
      max-width: 120px;
    }
    
    &-title {
      margin: 0;
      font-size: 0.9rem;
      color: #333;
      transition: color 0.3s ease;
    }
    
    &-description {
      display: none;
    }
    
    &.active {
      .learning-path__step-icon {
        background-color: var(--theme-color);
        border-color: var(--theme-color);
        transform: scale(1.1);
      }
      
      .learning-path__step-number {
        color: white;
      }
      
      .learning-path__step-title {
        color: var(--theme-color);
        font-weight: bold;
      }
    }
    
    &.completed {
      .learning-path__step-icon {
        background-color: #e6f7ff;
        border-color: var(--theme-color);
      }
      
      .learning-path__step-number {
        color: var(--theme-color);
      }
    }
  }
  
  &__details {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid var(--theme-color);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    
    &-header {
      margin-bottom: 1rem;
      
      h3 {
        margin: 0;
        color: var(--theme-color);
      }
    }
    
    &-body {
      display: flex;
      gap: 2rem;
      
      @media (max-width: 768px) {
        flex-direction: column;
        gap: 1rem;
      }
    }
  }
  
  &__topics, &__skills {
    flex: 1;
    
    h4 {
      margin-top: 0;
      margin-bottom: 0.8rem;
      font-size: 1rem;
      color: #333;
    }
    
    ul {
      margin: 0;
      padding-left: 1.5rem;
      
      li {
        margin-bottom: 0.5rem;
        
        a {
          color: var(--theme-color);
          text-decoration: none;
          
          &:hover {
            text-decoration: underline;
          }
        }
      }
    }
  }
  
  &__skill-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  &__skill-tag {
    background-color: rgba(24, 144, 255, 0.1);
    color: var(--theme-color);
    padding: 0.3rem 0.8rem;
    border-radius: 16px;
    font-size: 0.9rem;
  }
  
  &__action {
    margin-top: 1.5rem;
    text-align: right;
  }
  
  &__action-button {
    display: inline-block;
    background-color: var(--theme-color);
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 4px;
    text-decoration: none;
    font-weight: 500;
    transition: background-color 0.3s;
    
    &:hover {
      background-color: var(--theme-color-light);
    }
  }
  
  &__progress {
    margin-top: 2rem;
    padding: 0 1rem;
  }
  
  &__progress-bar {
    height: 6px;
    background-color: #f0f0f0;
    border-radius: 3px;
    overflow: hidden;
  }
  
  &__progress-fill {
    height: 100%;
    background-color: var(--theme-color);
    transition: width 0.3s ease;
  }
  
  &__progress-text {
    margin-top: 0.5rem;
    text-align: right;
    font-size: 0.9rem;
    color: #666;
  }
}

@media (max-width: 768px) {
  .learning-path {
    &__timeline {
      flex-direction: column;
      gap: 1rem;
      
      &:before {
        top: 0;
        bottom: 0;
        left: 1.25rem;
        right: auto;
        width: 2px;
        height: auto;
      }
    }
    
    &__step {
      width: 100%;
      flex-direction: row;
      align-items: flex-start;
      
      &-icon {
        margin-right: 1rem;
        margin-bottom: 0;
      }
      
      &-content {
        text-align: left;
        max-width: none;
      }
      
      &-title {
        font-size: 1rem;
      }
      
      &-description {
        display: block;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #666;
      }
    }
  }
}
</style>