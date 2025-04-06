import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderList, ssrRenderClass, ssrInterpolate, ssrRenderAttr, ssrRenderStyle } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  name: "LearningPathVisualization",
  data() {
    return {
      activeStep: 0,
      isEnglish: this.$lang === "en-US"
    };
  },
  computed: {
    steps() {
      if (this.isEnglish) {
        return [
          {
            title: "Data Mining Basics",
            description: "Understand the fundamental concepts and processes of data mining",
            topics: [
              { title: "What is Data Mining", link: "/en/overview/definition.html" },
              { title: "Data Mining Process", link: "/en/overview/process.html" },
              { title: "Applications", link: "/en/overview/applications.html" }
            ],
            skills: [
              "Understanding data mining terminology",
              "Identifying suitable problems for data mining",
              "Recognizing the data mining process steps"
            ],
            action: {
              text: "Start Learning Basics",
              link: "/en/overview/definition.html"
            }
          },
          {
            title: "Data Preprocessing",
            description: "Learn how to clean, transform and prepare data for analysis",
            topics: [
              { title: "Data Representation", link: "/en/core/preprocessing/data-presentation.html" },
              { title: "Missing Values", link: "/en/core/preprocessing/missing-values.html" },
              { title: "Feature Engineering", link: "/en/core/preprocessing/feature-engineering.html" }
            ],
            skills: [
              "Data cleaning techniques",
              "Handling missing values",
              "Feature selection and transformation",
              "Data normalization and standardization"
            ],
            action: {
              text: "Learn Data Preprocessing",
              link: "/en/core/preprocessing/data-presentation.html"
            }
          },
          {
            title: "Classification Algorithms",
            description: "Master various methods for predicting categorical outcomes",
            topics: [
              { title: "Decision Trees", link: "/en/core/classification/decision-trees.html" },
              { title: "Support Vector Machines", link: "/en/core/classification/svm.html" },
              { title: "Naive Bayes", link: "/en/core/classification/naive-bayes.html" }
            ],
            skills: [
              "Building classification models",
              "Model evaluation and validation",
              "Hyperparameter tuning",
              "Ensemble methods"
            ],
            action: {
              text: "Learn Classification Algorithms",
              link: "/en/core/classification/svm.html"
            }
          },
          {
            title: "Clustering Analysis",
            description: "Explore unsupervised learning methods to discover natural groupings",
            topics: [
              { title: "K-Means Clustering", link: "/en/core/clustering/kmeans.html" },
              { title: "Hierarchical Clustering", link: "/en/core/clustering/hierarchical.html" },
              { title: "Evaluation Methods", link: "/en/core/clustering/evaluation.html" }
            ],
            skills: [
              "Identifying appropriate clustering algorithms",
              "Determining optimal number of clusters",
              "Interpreting clustering results",
              "Visualizing high-dimensional clusters"
            ],
            action: {
              text: "Learn Clustering Analysis",
              link: "/en/core/clustering/kmeans.html"
            }
          },
          {
            title: "Prediction and Regression",
            description: "Learn techniques for predicting continuous values and time series",
            topics: [
              { title: "Linear Regression", link: "/en/core/regression/linear-regression.html" },
              { title: "Non-linear Models", link: "/en/core/regression/nonlinear-regression.html" },
              { title: "Model Evaluation", link: "/en/core/regression/evaluation-metrics.html" }
            ],
            skills: [
              "Building regression models",
              "Feature selection for regression",
              "Time series forecasting",
              "Regression model evaluation"
            ],
            action: {
              text: "Learn Prediction and Regression",
              link: "/en/core/regression/linear-regression.html"
            }
          },
          {
            title: "Practice Project",
            description: "Apply learned knowledge to real-world projects",
            topics: [
              { title: "Titanic Survival Prediction", link: "/en/projects/classification/titanic.html" },
              { title: "Customer Segmentation Analysis", link: "/en/projects/clustering/customer-segmentation.html" },
              { title: "House Price Prediction", link: "/en/projects/regression/house-price.html" }
            ],
            skills: ["Project Practice", "Comprehensive Application", "Result Interpretation"],
            action: {
              text: "Start Practice Project",
              link: "/en/projects/"
            }
          }
        ];
      } else {
        return [
          {
            title: "数据挖掘基础",
            description: "理解数据挖掘的基本概念和流程",
            topics: [
              { title: "什么是数据挖掘", link: "/overview/definition.html" },
              { title: "数据挖掘流程", link: "/overview/process.html" },
              { title: "应用场景", link: "/overview/applications.html" }
            ],
            skills: [
              "理解数据挖掘术语",
              "识别适合数据挖掘的问题",
              "认识数据挖掘流程步骤"
            ],
            action: {
              text: "开始学习基础知识",
              link: "/overview/definition.html"
            }
          },
          {
            title: "数据预处理",
            description: "学习如何清洗、转换和准备数据以进行分析",
            topics: [
              { title: "数据表示", link: "/core/preprocessing/data-presentation.html" },
              { title: "缺失值处理", link: "/core/preprocessing/missing-values.html" },
              { title: "特征工程", link: "/core/preprocessing/feature-engineering.html" }
            ],
            skills: [
              "数据清洗技术",
              "处理缺失值",
              "特征选择与转换",
              "数据归一化和标准化"
            ],
            action: {
              text: "学习数据预处理",
              link: "/core/preprocessing/data-presentation.html"
            }
          },
          {
            title: "分类算法",
            description: "掌握预测分类结果的各种方法",
            topics: [
              { title: "决策树", link: "/core/classification/decision-trees.html" },
              { title: "支持向量机", link: "/core/classification/svm.html" },
              { title: "朴素贝叶斯", link: "/core/classification/naive-bayes.html" }
            ],
            skills: [
              "构建分类模型",
              "模型评估和验证",
              "超参数调优",
              "集成方法"
            ],
            action: {
              text: "学习分类算法",
              link: "/core/classification/svm.html"
            }
          },
          {
            title: "聚类分析",
            description: "探索无监督学习方法以发现自然分组",
            topics: [
              { title: "K-均值聚类", link: "/core/clustering/kmeans.html" },
              { title: "层次聚类", link: "/core/clustering/hierarchical.html" },
              { title: "评估方法", link: "/core/clustering/evaluation.html" }
            ],
            skills: [
              "识别适当的聚类算法",
              "确定最佳聚类数量",
              "解释聚类结果",
              "可视化高维聚类"
            ],
            action: {
              text: "学习聚类分析",
              link: "/core/clustering/kmeans.html"
            }
          },
          {
            title: "预测与回归",
            description: "学习预测连续值和时间序列的技术",
            topics: [
              { title: "线性回归", link: "/core/regression/linear-regression.html" },
              { title: "非线性模型", link: "/core/regression/nonlinear-regression.html" },
              { title: "模型评估", link: "/core/regression/evaluation-metrics.html" }
            ],
            skills: [
              "构建回归模型",
              "回归的特征选择",
              "时间序列预测",
              "回归模型评估"
            ],
            action: {
              text: "学习预测与回归",
              link: "/core/regression/linear-regression.html"
            }
          },
          {
            title: "实践项目",
            description: "通过实际项目巩固所学知识，提升实战能力",
            topics: [
              { title: "泰坦尼克号生存预测", link: "/projects/classification/titanic.html" },
              { title: "客户分群分析", link: "/projects/clustering/customer-segmentation.html" },
              { title: "房价预测", link: "/projects/regression/house-price.html" }
            ],
            skills: ["项目实战", "综合应用", "结果解释"],
            action: {
              text: "开始实践项目",
              link: "/projects/"
            }
          }
        ];
      }
    }
  },
  methods: {
    setActiveStep(index) {
      this.activeStep = index;
    },
    setFromLearningPath() {
      localStorage.setItem("fromLearningPath", "true");
    }
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "learning-path" }, _attrs))} data-v-7b9626ca><div class="learning-path__timeline" data-v-7b9626ca><!--[-->`);
  ssrRenderList($options.steps, (step, index) => {
    _push(`<div class="${ssrRenderClass([{ "active": $data.activeStep === index, "completed": index < $data.activeStep }, "learning-path__step"])}" data-v-7b9626ca>`);
    if (index > 0) {
      _push(`<div class="learning-path__step-connector" data-v-7b9626ca></div>`);
    } else {
      _push(`<!---->`);
    }
    _push(`<div class="learning-path__step-icon" data-v-7b9626ca><span class="learning-path__step-number" data-v-7b9626ca>${ssrInterpolate(index + 1)}</span></div><div class="learning-path__step-content" data-v-7b9626ca><h3 class="learning-path__step-title" data-v-7b9626ca>${ssrInterpolate(step.title)}</h3><p class="learning-path__step-description" data-v-7b9626ca>${ssrInterpolate(step.description)}</p></div></div>`);
  });
  _push(`<!--]--></div>`);
  if ($options.steps[$data.activeStep]) {
    _push(`<div class="learning-path__details" data-v-7b9626ca><div class="learning-path__details-header" data-v-7b9626ca><h3 data-v-7b9626ca>${ssrInterpolate($options.steps[$data.activeStep].title)} - ${ssrInterpolate($data.isEnglish ? "Details" : "详细内容")}</h3></div><div class="learning-path__details-body" data-v-7b9626ca><div class="learning-path__topics" data-v-7b9626ca><h4 data-v-7b9626ca>${ssrInterpolate($data.isEnglish ? "Core Topics" : "核心主题")}</h4><ul data-v-7b9626ca><!--[-->`);
    ssrRenderList($options.steps[$data.activeStep].topics, (topic, topicIndex) => {
      _push(`<li data-v-7b9626ca><a${ssrRenderAttr("href", topic.link)} data-v-7b9626ca>${ssrInterpolate(topic.title)}</a></li>`);
    });
    _push(`<!--]--></ul></div><div class="learning-path__skills" data-v-7b9626ca><h4 data-v-7b9626ca>${ssrInterpolate($data.isEnglish ? "Key Skills" : "关键技能")}</h4><div class="learning-path__skill-tags" data-v-7b9626ca><!--[-->`);
    ssrRenderList($options.steps[$data.activeStep].skills, (skill, skillIndex) => {
      _push(`<span class="learning-path__skill-tag" data-v-7b9626ca>${ssrInterpolate(skill)}</span>`);
    });
    _push(`<!--]--></div></div></div>`);
    if ($options.steps[$data.activeStep].action) {
      _push(`<div class="learning-path__action" data-v-7b9626ca><a${ssrRenderAttr("href", $options.steps[$data.activeStep].action.link)} class="learning-path__action-button" data-v-7b9626ca>${ssrInterpolate($options.steps[$data.activeStep].action.text)}</a></div>`);
    } else {
      _push(`<!---->`);
    }
    _push(`</div>`);
  } else {
    _push(`<!---->`);
  }
  _push(`<div class="learning-path__progress" data-v-7b9626ca><div class="learning-path__progress-bar" data-v-7b9626ca><div class="learning-path__progress-fill" style="${ssrRenderStyle({ width: `${$data.activeStep / ($options.steps.length - 1) * 100}%` })}" data-v-7b9626ca></div></div><div class="learning-path__progress-text" data-v-7b9626ca>${ssrInterpolate($data.isEnglish ? "Completion" : "完成度")}: ${ssrInterpolate(Math.round($data.activeStep / ($options.steps.length - 1) * 100))}% </div></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/learning-path-visualization.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const learningPathVisualization = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-7b9626ca"], ["__file", "learning-path-visualization.vue"]]);
export {
  learningPathVisualization as default
};
