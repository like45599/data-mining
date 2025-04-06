import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrInterpolate } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  props: {
    category: {
      type: String,
      default: "general"
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
        general: {
          "数据挖掘一词最早出现在1990年代初，但其基本概念可以追溯到更早的统计分析和模式识别研究。": "The term 'data mining' first appeared in the early 1990s, but its basic concepts can be traced back to earlier statistical analysis and pattern recognition research.",
          "Netflix曾举办一个著名的竞赛，悬赏100万美元寻找能够提高其推荐系统准确率的算法。": "Netflix once held a famous competition offering $1 million to find an algorithm that could improve their recommendation system accuracy.",
          "决策树算法的历史可以追溯到1960年代，最早用于社会学研究。": "The history of decision tree algorithms can be traced back to the 1960s, when they were first used in sociological research.",
          "K-Means算法虽然简单，但在50多年后的今天仍然是最常用的聚类算法之一。": "Despite its simplicity, K-Means remains one of the most commonly used clustering algorithms even after 50 years.",
          "支持向量机(SVM)的理论基础来自于1960年代的统计学习理论，但直到1990年代才真正流行起来。": "Support Vector Machines (SVM) are based on statistical learning theory from the 1960s, but didn't become popular until the 1990s."
        },
        preprocessing: {
          "数据科学家通常花费60-80%的时间在数据清洗和预处理上。": "Data scientists typically spend 60-80% of their time on data cleaning and preprocessing.",
          "在大型数据项目中，良好的数据预处理可以将模型性能提高20%以上。": "In large data projects, good preprocessing can improve model performance by more than 20%.",
          "缺失值处理方法的选择可能比模型选择对最终结果影响更大。": "The choice of missing value handling method can have a greater impact on final results than model selection.",
          "特征工程被认为是数据科学中最重要的技能之一，往往比算法选择更能提升模型性能。": "Feature engineering is considered one of the most important skills in data science, often improving model performance more than algorithm selection."
        }
        // ... 其他类别的翻译
      }
    };
  },
  computed: {
    headerText() {
      return this.$lang === "en-US" ? "Did You Know?" : "你知道吗？";
    },
    prevButtonText() {
      return this.$lang === "en-US" ? "Previous" : "上一条";
    },
    nextButtonText() {
      return this.$lang === "en-US" ? "Next" : "下一条";
    },
    facts() {
      const lang = this.$lang;
      if (lang === "en-US" && this.factTranslations[this.category]) {
        const translations = this.factTranslations[this.category];
        return this.factsByCategory[this.category].map((fact) => translations[fact] || fact);
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
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "did-you-know" }, _attrs))} data-v-16f1c3a1><div class="did-you-know__header" data-v-16f1c3a1><span class="did-you-know__icon" data-v-16f1c3a1>💡</span><h3 data-v-16f1c3a1>${ssrInterpolate($options.headerText)}</h3></div><div class="did-you-know__content" data-v-16f1c3a1><p data-v-16f1c3a1>${ssrInterpolate($options.facts[$data.currentFactIndex])}</p></div><div class="did-you-know__footer" data-v-16f1c3a1><button class="did-you-know__button" data-v-16f1c3a1>${ssrInterpolate($options.prevButtonText)}</button><span class="did-you-know__counter" data-v-16f1c3a1>${ssrInterpolate($data.currentFactIndex + 1)}/${ssrInterpolate($options.facts.length)}</span><button class="did-you-know__button" data-v-16f1c3a1>${ssrInterpolate($options.nextButtonText)}</button></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/did-you-know.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const didYouKnow = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-16f1c3a1"], ["__file", "did-you-know.vue"]]);
export {
  didYouKnow as default
};
