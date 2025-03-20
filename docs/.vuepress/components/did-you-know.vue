<template>
  <div class="did-you-know">
    <div class="did-you-know__header">
      <span class="did-you-know__icon">💡</span>
      <h3>你知道吗？</h3>
    </div>
    <div class="did-you-know__content">
      <p>{{ facts[currentFactIndex] }}</p>
    </div>
    <div class="did-you-know__footer">
      <button @click="prevFact" class="did-you-know__button">上一条</button>
      <span class="did-you-know__counter">{{ currentFactIndex + 1 }}/{{ facts.length }}</span>
      <button @click="nextFact" class="did-you-know__button">下一条</button>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    category: {
      type: String,
      default: 'general'
    }
  },
  data() {
    return {
      currentFactIndex: 0,
      factsByCategory: {
        general: [
          "数据挖掘一词最早出现在1990年代初，但其基本概念可以追溯到更早的统计分析和模式识别研究。",
          "Netflix曾举办一个著名的竞赛，悬赏100万美元寻找能够提高其推荐系统准确率的算法。",
          "决策树算法的历史可以追溯到1960年代，最早用于社会学研究。",
          "K-Means算法虽然简单，但在50多年后的今天仍然是最常用的聚类算法之一。",
          "支持向量机(SVM)的理论基础来自于1960年代的统计学习理论，但直到1990年代才真正流行起来。"
        ],
        preprocessing: [
          "数据科学家通常花费60-80%的时间在数据清洗和预处理上。",
          "在大型数据项目中，良好的数据预处理可以将模型性能提高20%以上。",
          "缺失值处理方法的选择可能比模型选择对最终结果影响更大。",
          "特征工程被认为是数据科学中最重要的技能之一，往往比算法选择更能提升模型性能。"
        ],
        classification: [
          "朴素贝叶斯算法基于18世纪数学家托马斯·贝叶斯的工作，但直到计算机时代才广泛应用于分类问题。",
          "垃圾邮件过滤是最早成功应用机器学习的领域之一，大大减少了垃圾邮件的数量。",
          "决策树是少数几个既可用于分类又可用于回归的算法。",
          "支持向量机在高维空间中特别有效，这使其成为文本分类和基因分析的理想选择。"
        ],
        clustering: [
          "K-Means算法最早由Stuart Lloyd在1957年提出，但直到1982年才正式发表。",
          "聚类分析在市场细分中的应用可以追溯到1970年代。",
          "确定最佳聚类数量是聚类分析中最具挑战性的问题之一。",
          "层次聚类算法可以追溯到生物分类学，最初用于构建物种分类系统。"
        ],
        regression: [
          "线性回归是最古老的统计技术之一，可以追溯到19世纪初。",
          "最小二乘法由德国数学家高斯和法国数学家勒让德独立发明。",
          "神经网络的概念最早出现在1940年代，但直到近年来计算能力提升才真正流行。",
          "梯度提升树是许多数据科学竞赛中最常用的算法，因其强大的预测能力。"
        ]
      },
      factTranslations: {
        "数据挖掘一词最早出现在1990年代初，但其基本概念可以追溯到更早的统计分析和模式识别研究。": 
          "The term 'data mining' first appeared in the early 1990s, but its basic concepts can be traced back to earlier statistical analysis and pattern recognition research.",
        // 其他翻译...
      }
    }
  },
  computed: {
    facts() {
      const lang = this.$lang;
      if (lang === 'en-US') {
        return this.factsByCategory[this.category].map(fact => this.factTranslations[fact] || fact);
      }
      return this.factsByCategory[this.category];
    }
  },
  methods: {
    nextFact() {
      this.currentFactIndex = (this.currentFactIndex + 1) % this.facts.length;
    },
    prevFact() {
      this.currentFactIndex = (this.currentFactIndex - 1 + this.facts.length) % this.facts.length;
    }
  }
}
</script>

<style scoped>
.did-you-know {
  background-color: #f8f9fa;
  border-left: 4px solid var(--theme-color);
  border-radius: 4px;
  padding: 1.5rem;
  margin: 2rem 0;
}

.did-you-know__header {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
}

.did-you-know__icon {
  font-size: 1.5rem;
  margin-right: 0.5rem;
}

.did-you-know__header h3 {
  margin: 0;
  color: var(--theme-color);
}

.did-you-know__content {
  min-height: 80px;
}

.did-you-know__footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
}

.did-you-know__button {
  background-color: var(--theme-color);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  transition: background-color 0.3s;
}

.did-you-know__button:hover {
  background-color: var(--theme-color-light);
}

.did-you-know__counter {
  color: #666;
}
</style> 