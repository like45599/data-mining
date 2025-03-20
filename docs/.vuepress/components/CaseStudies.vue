<template>
  <div class="case-studies">
    <div class="case-study" v-for="(caseStudy, index) in caseStudies" :key="index">
      <div class="case-header">
        <div class="case-icon">{{ caseStudy.icon }}</div>
        <div class="case-title">{{ caseStudy.title }}</div>
      </div>
      
      <div class="case-content">
        <div class="case-section">
          <div class="section-title">{{ isEnglish ? 'Business Background' : '业务背景' }}</div>
          <div class="section-content">
            {{ caseStudy.background }}
          </div>
        </div>
        
        <div class="case-section">
          <div class="section-title">{{ isEnglish ? 'Data Mining Methods' : '数据挖掘方法' }}</div>
          <div class="section-content">
            <ul class="method-list">
              <li v-for="(method, methodIndex) in caseStudy.methods" :key="methodIndex">
                <span class="method-name">{{ method.name }}</span>：{{ method.description }}
              </li>
            </ul>
          </div>
        </div>
        
        <div class="case-section">
          <div class="section-title">{{ isEnglish ? 'Results & Benefits' : '结果与收益' }}</div>
          <div class="section-content">
            <div class="result-item" v-for="(result, resultIndex) in caseStudy.results" :key="resultIndex">
              <div class="result-value">{{ result.value }}</div>
              <div class="result-label">{{ result.label }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  computed: {
    isEnglish() {
      return this.$lang === 'en-US';
    },
    caseStudies() {
      if (this.isEnglish) {
        return [
          {
            icon: '📱',
            title: 'Telecom Customer Churn Prediction',
            background: 'The customer churn rate in the telecom industry is typically between 15-25%, and customer acquisition costs are much higher than retention costs. Predicting which customers are likely to churn can help companies take targeted measures to improve customer retention.',
            methods: [
              {
                name: 'Data Collection',
                description: 'Customer profiles, call records, billing information, customer service interactions'
              },
              {
                name: 'Feature Engineering',
                description: 'Call frequency, bill amount changes, contract term, number of complaints'
              },
              {
                name: 'Model Selection',
                description: 'Logistic regression, random forest, gradient boosting trees'
              },
              {
                name: 'Model Evaluation',
                description: 'AUC, precision, recall, F1 score'
              }
            ],
            results: [
              {
                value: '25%',
                label: 'Reduction in customer churn'
              },
              {
                value: '$1.8M',
                label: 'Annual cost savings'
              },
              {
                value: '85%',
                label: 'Prediction accuracy'
              }
            ]
          },
          {
            icon: '🛒',
            title: 'Retail Market Basket Analysis',
            background: 'Understanding which products are frequently purchased together can help retailers optimize product placement, promotions, and inventory management. Market basket analysis uses association rule mining to discover these patterns.',
            methods: [
              {
                name: 'Data Collection',
                description: 'Transaction records, product categories, time of purchase'
              },
              {
                name: 'Data Preprocessing',
                description: 'Transaction formatting, product categorization, time period segmentation'
              },
              {
                name: 'Association Rule Mining',
                description: 'Apriori algorithm, FP-Growth, support and confidence calculation'
              },
              {
                name: 'Rule Evaluation',
                description: 'Lift, conviction, leverage metrics'
              }
            ],
            results: [
              {
                value: '18%',
                label: 'Increase in cross-selling'
              },
              {
                value: '12%',
                label: 'Improvement in inventory turnover'
              },
              {
                value: '200+',
                label: 'Actionable product associations discovered'
              }
            ]
          }
        ];
      } else {
        return [
          {
            icon: '📱',
            title: '电信客户流失预测',
            background: '电信行业的客户流失率通常在15-25%之间，客户获取成本远高于保留成本。预测哪些客户可能流失，可以帮助公司采取针对性措施提高客户保留率。',
            methods: [
              {
                name: '数据收集',
                description: '客户资料、通话记录、账单信息、客服互动'
              },
              {
                name: '特征工程',
                description: '通话频率、账单金额变化、合约期限、投诉次数'
              },
              {
                name: '模型选择',
                description: '逻辑回归、随机森林、梯度提升树'
              },
              {
                name: '模型评估',
                description: 'AUC、精确率、召回率、F1分数'
              }
            ],
            results: [
              {
                value: '25%',
                label: '客户流失率降低'
              },
              {
                value: '180万',
                label: '年度成本节约'
              },
              {
                value: '85%',
                label: '预测准确率'
              }
            ]
          },
          {
            icon: '🛒',
            title: '零售购物篮分析',
            background: '了解哪些产品经常一起购买可以帮助零售商优化产品布局、促销活动和库存管理。购物篮分析使用关联规则挖掘来发现这些模式。',
            methods: [
              {
                name: '数据收集',
                description: '交易记录、产品类别、购买时间'
              },
              {
                name: '数据预处理',
                description: '交易格式化、产品分类、时间段划分'
              },
              {
                name: '关联规则挖掘',
                description: 'Apriori算法、FP-Growth、支持度和置信度计算'
              },
              {
                name: '规则评估',
                description: '提升度、确信度、杠杆率指标'
              }
            ],
            results: [
              {
                value: '18%',
                label: '交叉销售增长'
              },
              {
                value: '12%',
                label: '库存周转率提升'
              },
              {
                value: '200+',
                label: '发现可行的产品关联'
              }
            ]
          }
        ];
      }
    }
  }
};
</script>

<style scoped>
.case-studies {
  display: flex;
  flex-direction: column;
  gap: 30px;
  margin: 20px 0;
}

.case-study {
  background-color: white;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  overflow: hidden;
}

.case-header {
  background-color: #3498db;
  color: white;
  padding: 15px 20px;
  display: flex;
  align-items: center;
}

.case-icon {
  font-size: 24px;
  margin-right: 15px;
}

.case-title {
  font-size: 18px;
  font-weight: bold;
}

.case-content {
  padding: 20px;
}

.case-section {
  margin-bottom: 20px;
}

.section-title {
  font-weight: bold;
  color: #2c3e50;
  margin-bottom: 10px;
  font-size: 16px;
}

.method-list {
  list-style-type: none;
  padding: 0;
  margin: 0;
}

.method-list li {
  margin-bottom: 8px;
}

.method-name {
  font-weight: bold;
  color: #3498db;
}

.result-item {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  text-align: center;
  min-width: 100px;
}

.result-value {
  font-size: 24px;
  font-weight: bold;
  color: #3498db;
  margin-bottom: 5px;
}

.result-label {
  font-size: 14px;
  color: #7f8c8d;
}

.section-content {
  display: flex;
  gap: 15px;
}

.section-content:not(.result-item) {
  display: block;
}

@media (max-width: 768px) {
  .section-content {
    flex-direction: column;
  }
}
</style> 