import { shallowRef, defineComponent, computed, h as h$1, toRef, ref, onMounted, defineAsyncComponent, watch, reactive, inject, onUnmounted, TransitionGroup, Transition, getCurrentInstance, camelize, capitalize, toValue, provide, useSSRContext, mergeProps, createSlots, withCtx, renderSlot, createVNode, createBlock, createCommentVNode, openBlock, toDisplayString, toRefs, withDirectives, Fragment, renderList, vShow, resolveComponent, useTemplateRef, createTextVNode, nextTick, onBeforeUnmount, createSSRApp, customRef } from "vue";
import { isLinkWithProtocol, resolveLocalePath, isString, dedupeHead, splitPath, normalizeRoutePath, isLinkHttp, removeLeadingSlash, isLinkExternal, ensureEndingSlash, isPlainObject, removeEndingSlash } from "@vuepress/shared";
import { useRouter, useRoute, createRouter, createMemoryHistory, START_LOCATION } from "vue-router";
import { useElementSize, useWindowSize, useWindowScroll, useEventListener, useToggle, useStorage, usePreferredDark, watchImmediate, provideLocal, injectLocal } from "@vueuse/core";
import { setupDevtoolsPlugin } from "@vue/devtools-api";
import { ssrRenderAttrs, ssrRenderSlot, ssrInterpolate, ssrRenderSlotInner, ssrRenderList, ssrRenderComponent, ssrRenderClass, ssrRenderAttr, ssrRenderStyle } from "vue/server-renderer";
const redirects$1 = JSON.parse("{}");
const routes$1 = Object.fromEntries([
  ["/", { loader: () => import(
    /* webpackChunkName: "index.html" */
    "./assets/index.html-DW1HSo1T.js"
  ), meta: { "title": "" } }],
  ["/about/", { loader: () => import(
    /* webpackChunkName: "about_index.html" */
    "./assets/index.html-BO2Yyu5L.js"
  ), meta: { "title": "关于我们" } }],
  ["/en/", { loader: () => import(
    /* webpackChunkName: "en_index.html" */
    "./assets/index.html-BSU8EovR.js"
  ), meta: { "title": "" } }],
  ["/guide/classification-svm.html", { loader: () => import(
    /* webpackChunkName: "guide_classification-svm.html" */
    "./assets/classification-svm.html-De6JtdcD.js"
  ), meta: { "title": "支持向量机分类" } }],
  ["/guide/clustering.html", { loader: () => import(
    /* webpackChunkName: "guide_clustering.html" */
    "./assets/clustering.html-Zd49li5r.js"
  ), meta: { "title": "聚类分析" } }],
  ["/guide/data-presentation.html", { loader: () => import(
    /* webpackChunkName: "guide_data-presentation.html" */
    "./assets/data-presentation.html-DOK3pryq.js"
  ), meta: { "title": "数据表示方法" } }],
  ["/guide/decision-trees.html", { loader: () => import(
    /* webpackChunkName: "guide_decision-trees.html" */
    "./assets/decision-trees.html-BWS4u_lD.js"
  ), meta: { "title": "决策树" } }],
  ["/guide/missing-values.html", { loader: () => import(
    /* webpackChunkName: "guide_missing-values.html" */
    "./assets/missing-values.html-BO7FSCL8.js"
  ), meta: { "title": "缺失值处理" } }],
  ["/guide/prediction.html", { loader: () => import(
    /* webpackChunkName: "guide_prediction.html" */
    "./assets/prediction.html-HxzBSqxZ.js"
  ), meta: { "title": "预测与回归分析" } }],
  ["/guide/spam-filters.html", { loader: () => import(
    /* webpackChunkName: "guide_spam-filters.html" */
    "./assets/spam-filters.html-CcIykEYA.js"
  ), meta: { "title": "垃圾邮件过滤" } }],
  ["/learning-path/advanced.html", { loader: () => import(
    /* webpackChunkName: "learning-path_advanced.html" */
    "./assets/advanced.html-D0ufszPD.js"
  ), meta: { "title": "进阶学习指南" } }],
  ["/learning-path/beginner.html", { loader: () => import(
    /* webpackChunkName: "learning-path_beginner.html" */
    "./assets/beginner.html-Bl2pir-h.js"
  ), meta: { "title": "初学者指南" } }],
  ["/learning-path/practical.html", { loader: () => import(
    /* webpackChunkName: "learning-path_practical.html" */
    "./assets/practical.html-DP_ztHJk.js"
  ), meta: { "title": "实践应用指南" } }],
  ["/learning-path/", { loader: () => import(
    /* webpackChunkName: "learning-path_index.html" */
    "./assets/index.html-DWEkRx0r.js"
  ), meta: { "title": "学习路径" } }],
  ["/overview/applications.html", { loader: () => import(
    /* webpackChunkName: "overview_applications.html" */
    "./assets/applications.html-DIQxPQVm.js"
  ), meta: { "title": "数据挖掘应用领域" } }],
  ["/overview/definition.html", { loader: () => import(
    /* webpackChunkName: "overview_definition.html" */
    "./assets/definition.html-B2LziOGh.js"
  ), meta: { "title": "数据挖掘的定义与基本概念" } }],
  ["/overview/process.html", { loader: () => import(
    /* webpackChunkName: "overview_process.html" */
    "./assets/process.html-CPb94x-A.js"
  ), meta: { "title": "数据挖掘过程" } }],
  ["/overview/", { loader: () => import(
    /* webpackChunkName: "overview_index.html" */
    "./assets/index.html-BVyj4Mp7.js"
  ), meta: { "title": "数据挖掘概述" } }],
  ["/overview/tools.html", { loader: () => import(
    /* webpackChunkName: "overview_tools.html" */
    "./assets/tools.html-tbgu92Zg.js"
  ), meta: { "title": "数据挖掘工具与平台" } }],
  ["/core/classification/comparison.html", { loader: () => import(
    /* webpackChunkName: "core_classification_comparison.html" */
    "./assets/comparison.html-BntrtIHA.js"
  ), meta: { "title": "分类算法比较" } }],
  ["/core/classification/decision-trees.html", { loader: () => import(
    /* webpackChunkName: "core_classification_decision-trees.html" */
    "./assets/decision-trees.html-ZNa5gQ1R.js"
  ), meta: { "title": "决策树算法" } }],
  ["/core/classification/naive-bayes.html", { loader: () => import(
    /* webpackChunkName: "core_classification_naive-bayes.html" */
    "./assets/naive-bayes.html-CEAirp4k.js"
  ), meta: { "title": "朴素贝叶斯算法" } }],
  ["/core/classification/svm.html", { loader: () => import(
    /* webpackChunkName: "core_classification_svm.html" */
    "./assets/svm.html-huDN65i-.js"
  ), meta: { "title": "支持向量机(SVM)" } }],
  ["/core/clustering/applications.html", { loader: () => import(
    /* webpackChunkName: "core_clustering_applications.html" */
    "./assets/applications.html-Bb7iNXEq.js"
  ), meta: { "title": "聚类实际应用案例" } }],
  ["/core/clustering/evaluation.html", { loader: () => import(
    /* webpackChunkName: "core_clustering_evaluation.html" */
    "./assets/evaluation.html-Bcg3gi4s.js"
  ), meta: { "title": "聚类评价指标" } }],
  ["/core/clustering/kmeans.html", { loader: () => import(
    /* webpackChunkName: "core_clustering_kmeans.html" */
    "./assets/kmeans.html-4oTgV_0b.js"
  ), meta: { "title": "K-Means聚类算法" } }],
  ["/core/preprocessing/data-presentation.html", { loader: () => import(
    /* webpackChunkName: "core_preprocessing_data-presentation.html" */
    "./assets/data-presentation.html-iM9pSSxm.js"
  ), meta: { "title": "数据表示方法" } }],
  ["/core/preprocessing/missing-values.html", { loader: () => import(
    /* webpackChunkName: "core_preprocessing_missing-values.html" */
    "./assets/missing-values.html-gIrYlU9c.js"
  ), meta: { "title": "缺失值处理" } }],
  ["/practice/", { loader: () => import(
    /* webpackChunkName: "practice_index.html" */
    "./assets/index.html-DnIRthAA.js"
  ), meta: { "title": "实践指南" } }],
  ["/resources/", { loader: () => import(
    /* webpackChunkName: "resources_index.html" */
    "./assets/index.html-CZdwAAcK.js"
  ), meta: { "title": "课程资源" } }],
  ["/projects/classification.html", { loader: () => import(
    /* webpackChunkName: "projects_classification.html" */
    "./assets/classification.html-Bxk4VL6T.js"
  ), meta: { "title": "分类算法项目" } }],
  ["/projects/clustering.html", { loader: () => import(
    /* webpackChunkName: "projects_clustering.html" */
    "./assets/clustering.html-DyyCrocs.js"
  ), meta: { "title": "聚类分析项目" } }],
  ["/projects/prediction.html", { loader: () => import(
    /* webpackChunkName: "projects_prediction.html" */
    "./assets/prediction.html-QNpz4xOm.js"
  ), meta: { "title": "预测与回归项目" } }],
  ["/projects/preprocessing.html", { loader: () => import(
    /* webpackChunkName: "projects_preprocessing.html" */
    "./assets/preprocessing.html-Bfr3F8sk.js"
  ), meta: { "title": "数据预处理项目" } }],
  ["/projects/", { loader: () => import(
    /* webpackChunkName: "projects_index.html" */
    "./assets/index.html-BF7SPEmD.js"
  ), meta: { "title": "数据挖掘实践项目" } }],
  ["/projects/titanic.html", { loader: () => import(
    /* webpackChunkName: "projects_titanic.html" */
    "./assets/titanic.html-DQ7XH7_h.js"
  ), meta: { "title": "泰坦尼克号生存预测" } }],
  ["/core/regression/case-studies.html", { loader: () => import(
    /* webpackChunkName: "core_regression_case-studies.html" */
    "./assets/case-studies.html-D9e2k-31.js"
  ), meta: { "title": "回归实践案例" } }],
  ["/core/regression/evaluation-metrics.html", { loader: () => import(
    /* webpackChunkName: "core_regression_evaluation-metrics.html" */
    "./assets/evaluation-metrics.html-DKJQIqya.js"
  ), meta: { "title": "回归评估指标" } }],
  ["/core/regression/linear-regression.html", { loader: () => import(
    /* webpackChunkName: "core_regression_linear-regression.html" */
    "./assets/linear-regression.html-CkoAmwkI.js"
  ), meta: { "title": "线性回归" } }],
  ["/core/regression/model-selection.html", { loader: () => import(
    /* webpackChunkName: "core_regression_model-selection.html" */
    "./assets/model-selection.html-B5qmFGEJ.js"
  ), meta: { "title": "回归模型选择" } }],
  ["/core/regression/nonlinear-regression.html", { loader: () => import(
    /* webpackChunkName: "core_regression_nonlinear-regression.html" */
    "./assets/nonlinear-regression.html-D38QLNsQ.js"
  ), meta: { "title": "非线性回归方法" } }],
  ["/core/regression/summary.html", { loader: () => import(
    /* webpackChunkName: "core_regression_summary.html" */
    "./assets/summary.html-Cll9ZFJw.js"
  ), meta: { "title": "回归分析总结" } }],
  ["/en/about/", { loader: () => import(
    /* webpackChunkName: "en_about_index.html" */
    "./assets/index.html-CtPOTjG2.js"
  ), meta: { "title": "About Us" } }],
  ["/en/learning-path/advanced.html", { loader: () => import(
    /* webpackChunkName: "en_learning-path_advanced.html" */
    "./assets/advanced.html-BnB6x5Nk.js"
  ), meta: { "title": "Advanced Learning Guide" } }],
  ["/en/learning-path/beginner.html", { loader: () => import(
    /* webpackChunkName: "en_learning-path_beginner.html" */
    "./assets/beginner.html-Dou7ccry.js"
  ), meta: { "title": "Beginner's Guide" } }],
  ["/en/learning-path/practical.html", { loader: () => import(
    /* webpackChunkName: "en_learning-path_practical.html" */
    "./assets/practical.html-CCjep1GX.js"
  ), meta: { "title": "Practical Application Guide" } }],
  ["/en/learning-path/", { loader: () => import(
    /* webpackChunkName: "en_learning-path_index.html" */
    "./assets/index.html-Cem_dHHf.js"
  ), meta: { "title": "Learning Path" } }],
  ["/en/overview/applications.html", { loader: () => import(
    /* webpackChunkName: "en_overview_applications.html" */
    "./assets/applications.html-xWKzdCgo.js"
  ), meta: { "title": "Data Mining Application Areas" } }],
  ["/en/overview/definition.html", { loader: () => import(
    /* webpackChunkName: "en_overview_definition.html" */
    "./assets/definition.html-DQMzverF.js"
  ), meta: { "title": "Definition and Basic Concepts of Data Mining" } }],
  ["/en/overview/process.html", { loader: () => import(
    /* webpackChunkName: "en_overview_process.html" */
    "./assets/process.html-BhzWUslO.js"
  ), meta: { "title": "Data Mining Process" } }],
  ["/en/overview/", { loader: () => import(
    /* webpackChunkName: "en_overview_index.html" */
    "./assets/index.html-COMOTblF.js"
  ), meta: { "title": "Data Mining Overview" } }],
  ["/en/overview/tools.html", { loader: () => import(
    /* webpackChunkName: "en_overview_tools.html" */
    "./assets/tools.html-B-tnopUb.js"
  ), meta: { "title": "Data Mining Tools and Platforms" } }],
  ["/en/projects/classification.html", { loader: () => import(
    /* webpackChunkName: "en_projects_classification.html" */
    "./assets/classification.html-BcRDOVC5.js"
  ), meta: { "title": "Classification Algorithm Projects" } }],
  ["/en/projects/clustering.html", { loader: () => import(
    /* webpackChunkName: "en_projects_clustering.html" */
    "./assets/clustering.html-CBBiv-qG.js"
  ), meta: { "title": "Clustering Analysis Project" } }],
  ["/en/projects/prediction.html", { loader: () => import(
    /* webpackChunkName: "en_projects_prediction.html" */
    "./assets/prediction.html-CJGOWsZP.js"
  ), meta: { "title": "Prediction and Regression Projects" } }],
  ["/en/projects/preprocessing.html", { loader: () => import(
    /* webpackChunkName: "en_projects_preprocessing.html" */
    "./assets/preprocessing.html-BuAVKXe_.js"
  ), meta: { "title": "Data Preprocessing Projects" } }],
  ["/en/projects/", { loader: () => import(
    /* webpackChunkName: "en_projects_index.html" */
    "./assets/index.html-79TorSmr.js"
  ), meta: { "title": "Data Mining Practical Projects" } }],
  ["/projects/classification/credit-risk.html", { loader: () => import(
    /* webpackChunkName: "projects_classification_credit-risk.html" */
    "./assets/credit-risk.html-McoL66U_.js"
  ), meta: { "title": "信用风险评估" } }],
  ["/projects/classification/spam-filter.html", { loader: () => import(
    /* webpackChunkName: "projects_classification_spam-filter.html" */
    "./assets/spam-filter.html-Do2lnz9u.js"
  ), meta: { "title": "垃圾邮件过滤器" } }],
  ["/projects/classification/titanic.html", { loader: () => import(
    /* webpackChunkName: "projects_classification_titanic.html" */
    "./assets/titanic.html-CF-D1A2U.js"
  ), meta: { "title": "泰坦尼克号生存预测" } }],
  ["/projects/clustering/customer-segmentation.html", { loader: () => import(
    /* webpackChunkName: "projects_clustering_customer-segmentation.html" */
    "./assets/customer-segmentation.html-CK5ecHa8.js"
  ), meta: { "title": "客户分群分析" } }],
  ["/projects/clustering/image-segmentation.html", { loader: () => import(
    /* webpackChunkName: "projects_clustering_image-segmentation.html" */
    "./assets/image-segmentation.html-DZamwIiu.js"
  ), meta: { "title": "图像颜色分割" } }],
  ["/projects/preprocessing/ecommerce-data.html", { loader: () => import(
    /* webpackChunkName: "projects_preprocessing_ecommerce-data.html" */
    "./assets/ecommerce-data.html-DTekIlZQ.js"
  ), meta: { "title": "电商用户数据清洗与分析" } }],
  ["/projects/preprocessing/medical-missing-values.html", { loader: () => import(
    /* webpackChunkName: "projects_preprocessing_medical-missing-values.html" */
    "./assets/medical-missing-values.html-Bb-OJuZB.js"
  ), meta: { "title": "医疗数据缺失值处理" } }],
  ["/projects/regression/anomaly-detection.html", { loader: () => import(
    /* webpackChunkName: "projects_regression_anomaly-detection.html" */
    "./assets/anomaly-detection.html-CmfBsgZg.js"
  ), meta: { "title": "异常检测与预测" } }],
  ["/projects/regression/house-price.html", { loader: () => import(
    /* webpackChunkName: "projects_regression_house-price.html" */
    "./assets/house-price.html-BxewlKW4.js"
  ), meta: { "title": "房价预测模型" } }],
  ["/projects/regression/sales-forecast.html", { loader: () => import(
    /* webpackChunkName: "projects_regression_sales-forecast.html" */
    "./assets/sales-forecast.html-BRzhidPH.js"
  ), meta: { "title": "销售额预测" } }],
  ["/en/core/classification/comparison.html", { loader: () => import(
    /* webpackChunkName: "en_core_classification_comparison.html" */
    "./assets/comparison.html-CG0tFAGO.js"
  ), meta: { "title": "Classification Algorithm Comparison" } }],
  ["/en/core/classification/decision-trees.html", { loader: () => import(
    /* webpackChunkName: "en_core_classification_decision-trees.html" */
    "./assets/decision-trees.html-BBqR7ZYB.js"
  ), meta: { "title": "Decision Tree Algorithm" } }],
  ["/en/core/classification/naive-bayes.html", { loader: () => import(
    /* webpackChunkName: "en_core_classification_naive-bayes.html" */
    "./assets/naive-bayes.html-BpI73D7a.js"
  ), meta: { "title": "Naive Bayes Algorithm" } }],
  ["/en/core/classification/svm.html", { loader: () => import(
    /* webpackChunkName: "en_core_classification_svm.html" */
    "./assets/svm.html-DRjR7Ou-.js"
  ), meta: { "title": "Support Vector Machine (SVM)" } }],
  ["/en/core/clustering/applications.html", { loader: () => import(
    /* webpackChunkName: "en_core_clustering_applications.html" */
    "./assets/applications.html-Bw_03uoA.js"
  ), meta: { "title": "Clustering Practical Application Cases" } }],
  ["/en/core/clustering/evaluation.html", { loader: () => import(
    /* webpackChunkName: "en_core_clustering_evaluation.html" */
    "./assets/evaluation.html-CgF05sQ4.js"
  ), meta: { "title": "Clustering Evaluation Metrics" } }],
  ["/en/core/clustering/kmeans.html", { loader: () => import(
    /* webpackChunkName: "en_core_clustering_kmeans.html" */
    "./assets/kmeans.html-DfNu8OiV.js"
  ), meta: { "title": "K-Means Clustering Algorithm" } }],
  ["/en/core/preprocessing/data-presentation.html", { loader: () => import(
    /* webpackChunkName: "en_core_preprocessing_data-presentation.html" */
    "./assets/data-presentation.html-CUWRCCt-.js"
  ), meta: { "title": "Data Representation Methods" } }],
  ["/en/core/preprocessing/missing-values.html", { loader: () => import(
    /* webpackChunkName: "en_core_preprocessing_missing-values.html" */
    "./assets/missing-values.html-BMIN56Ax.js"
  ), meta: { "title": "Missing Value Handling" } }],
  ["/en/core/regression/evaluation-metrics.html", { loader: () => import(
    /* webpackChunkName: "en_core_regression_evaluation-metrics.html" */
    "./assets/evaluation-metrics.html--miuq2Lj.js"
  ), meta: { "title": "Regression Evaluation Metrics" } }],
  ["/en/core/regression/linear-regression.html", { loader: () => import(
    /* webpackChunkName: "en_core_regression_linear-regression.html" */
    "./assets/linear-regression.html-BhNThp5o.js"
  ), meta: { "title": "Linear Regression" } }],
  ["/en/core/regression/nonlinear-regression.html", { loader: () => import(
    /* webpackChunkName: "en_core_regression_nonlinear-regression.html" */
    "./assets/nonlinear-regression.html-C9MO9YEK.js"
  ), meta: { "title": "Nonlinear Regression Methods" } }],
  ["/en/projects/classification/credit-risk.html", { loader: () => import(
    /* webpackChunkName: "en_projects_classification_credit-risk.html" */
    "./assets/credit-risk.html-DR_hIkA4.js"
  ), meta: { "title": "Credit Risk Assessment" } }],
  ["/en/projects/classification/spam-filter.html", { loader: () => import(
    /* webpackChunkName: "en_projects_classification_spam-filter.html" */
    "./assets/spam-filter.html-9IiBziVX.js"
  ), meta: { "title": "Spam Filter" } }],
  ["/en/projects/classification/titanic.html", { loader: () => import(
    /* webpackChunkName: "en_projects_classification_titanic.html" */
    "./assets/titanic.html-Bxb7UV7f.js"
  ), meta: { "title": "Titanic Survival Prediction" } }],
  ["/en/projects/clustering/customer-segmentation.html", { loader: () => import(
    /* webpackChunkName: "en_projects_clustering_customer-segmentation.html" */
    "./assets/customer-segmentation.html-BOhaE3Ot.js"
  ), meta: { "title": "Customer Segmentation Analysis" } }],
  ["/en/projects/clustering/image-segmentation.html", { loader: () => import(
    /* webpackChunkName: "en_projects_clustering_image-segmentation.html" */
    "./assets/image-segmentation.html-BzUzF-Qs.js"
  ), meta: { "title": "Image Color Segmentation" } }],
  ["/en/projects/preprocessing/ecommerce-data.html", { loader: () => import(
    /* webpackChunkName: "en_projects_preprocessing_ecommerce-data.html" */
    "./assets/ecommerce-data.html-BIwjeKM9.js"
  ), meta: { "title": "E-commerce User Data Cleaning and Analysis" } }],
  ["/en/projects/preprocessing/medical-missing-values.html", { loader: () => import(
    /* webpackChunkName: "en_projects_preprocessing_medical-missing-values.html" */
    "./assets/medical-missing-values.html-CVE2SP4W.js"
  ), meta: { "title": "Handling Missing Values in Medical Data" } }],
  ["/en/projects/regression/anomaly-detection.html", { loader: () => import(
    /* webpackChunkName: "en_projects_regression_anomaly-detection.html" */
    "./assets/anomaly-detection.html-Hb0aBVzk.js"
  ), meta: { "title": "Anomaly Detection and Prediction" } }],
  ["/en/projects/regression/house-price.html", { loader: () => import(
    /* webpackChunkName: "en_projects_regression_house-price.html" */
    "./assets/house-price.html-CjBV3nuE.js"
  ), meta: { "title": "House Price Prediction Model" } }],
  ["/en/projects/regression/sales-forecast.html", { loader: () => import(
    /* webpackChunkName: "en_projects_regression_sales-forecast.html" */
    "./assets/sales-forecast.html-CXsv7xeB.js"
  ), meta: { "title": "Sales Forecasting" } }],
  ["/404.html", { loader: () => import(
    /* webpackChunkName: "404.html" */
    "./assets/404.html-uEynKHQR.js"
  ), meta: { "title": "" } }]
]);
var clientDataSymbol = Symbol(
  ""
);
var useClientData = () => {
  const clientData = inject(clientDataSymbol);
  if (!clientData) {
    throw new Error("useClientData() is called without provider.");
  }
  return clientData;
};
var usePageComponent = () => useClientData().pageComponent;
var usePageData = () => useClientData().pageData;
var usePageFrontmatter = () => useClientData().pageFrontmatter;
var usePageHead = () => useClientData().pageHead;
var usePageLang = () => useClientData().pageLang;
var usePageLayout = () => useClientData().pageLayout;
var useRouteLocale = () => useClientData().routeLocale;
var useRoutePath = () => useClientData().routePath;
var useRoutes = () => useClientData().routes;
var useSiteData = () => useClientData().siteData;
var contentUpdatedCallbacks = shallowRef([]);
var onContentUpdated = (fn) => {
  contentUpdatedCallbacks.value.push(fn);
  onUnmounted(() => {
    contentUpdatedCallbacks.value = contentUpdatedCallbacks.value.filter(
      (f) => f !== fn
    );
  });
};
var redirects = shallowRef(redirects$1);
var routes = shallowRef(routes$1);
var resolveRoutePath = (pathname, currentPath) => {
  const normalizedRoutePath = normalizeRoutePath(pathname, currentPath);
  if (routes.value[normalizedRoutePath]) return normalizedRoutePath;
  const encodedRoutePath = encodeURI(normalizedRoutePath);
  if (routes.value[encodedRoutePath]) {
    return encodedRoutePath;
  }
  const redirectedRoutePath = redirects.value[normalizedRoutePath] || redirects.value[encodedRoutePath];
  if (redirectedRoutePath) {
    return redirectedRoutePath;
  }
  return normalizedRoutePath;
};
var resolveRoute = (path, currentPath) => {
  const { pathname, hashAndQueries } = splitPath(path);
  const routePath = resolveRoutePath(pathname, currentPath);
  const routeFullPath = routePath + hashAndQueries;
  if (!routes.value[routePath]) {
    return {
      ...routes.value["/404.html"],
      path: routeFullPath,
      notFound: true
    };
  }
  return {
    ...routes.value[routePath],
    path: routeFullPath,
    notFound: false
  };
};
var resolveRouteFullPath = (path, currentPath) => {
  const { pathname, hashAndQueries } = splitPath(path);
  return resolveRoutePath(pathname, currentPath) + hashAndQueries;
};
var guardEvent = (event) => {
  if (event.metaKey || event.altKey || event.ctrlKey || event.shiftKey) return;
  if (event.defaultPrevented) return;
  if (event.button !== void 0 && event.button !== 0) return;
  if (event.currentTarget) {
    const target = event.currentTarget.getAttribute("target");
    if (target == null ? void 0 : target.match(/\b_blank\b/i)) return;
  }
  event.preventDefault();
  return true;
};
var RouteLink = defineComponent({
  name: "RouteLink",
  props: {
    /**
     * The route path to link to
     */
    to: {
      type: String,
      required: true
    },
    /**
     * Whether the link is active to have an active class
     *
     * Notice that the active status is not automatically determined according to the current route.
     */
    active: Boolean,
    /**
     * The class to add when the link is active
     */
    activeClass: {
      type: String,
      default: "route-link-active"
    }
  },
  slots: Object,
  setup(props, { slots }) {
    const router = useRouter();
    const route = useRoute();
    const path = computed(
      () => props.to.startsWith("#") || props.to.startsWith("?") ? props.to : `${"/"}${resolveRouteFullPath(props.to, route.path).substring(1)}`
    );
    return () => h$1(
      "a",
      {
        class: ["route-link", { [props.activeClass]: props.active }],
        href: path.value,
        onClick: (event = {}) => {
          if (guardEvent(event)) {
            void router.push(props.to).catch();
          }
        }
      },
      slots.default()
    );
  }
});
var AutoLink = defineComponent({
  name: "AutoLink",
  props: {
    config: {
      type: Object,
      required: true
    }
  },
  slots: Object,
  setup(props, { slots }) {
    const config2 = toRef(props, "config");
    const route = useRoute();
    const siteData2 = useSiteData();
    const withProtocol = computed(() => isLinkWithProtocol(config2.value.link));
    const linkTarget = computed(
      () => config2.value.target || (withProtocol.value ? "_blank" : void 0)
    );
    const isBlankTarget = computed(() => linkTarget.value === "_blank");
    const isInternal = computed(
      () => !withProtocol.value && !isBlankTarget.value
    );
    const linkRel = computed(
      () => config2.value.rel || (isBlankTarget.value ? "noopener noreferrer" : null)
    );
    const linkAriaLabel = computed(
      () => config2.value.ariaLabel ?? config2.value.text
    );
    const shouldBeActiveInSubpath = computed(() => {
      if (config2.value.exact) return false;
      const localePaths = Object.keys(siteData2.value.locales);
      return localePaths.length ? (
        // Check all the locales
        localePaths.every((key) => key !== config2.value.link)
      ) : (
        // Check root
        config2.value.link !== "/"
      );
    });
    const isActive = computed(() => {
      if (!isInternal.value) return false;
      if (config2.value.activeMatch) {
        return (config2.value.activeMatch instanceof RegExp ? config2.value.activeMatch : new RegExp(config2.value.activeMatch, "u")).test(route.path);
      }
      if (shouldBeActiveInSubpath.value) {
        return route.path.startsWith(config2.value.link);
      }
      return route.path === config2.value.link;
    });
    return () => {
      const { before, after, default: defaultSlot } = slots;
      const content = (defaultSlot == null ? void 0 : defaultSlot(config2.value)) ?? [
        before == null ? void 0 : before(config2.value),
        config2.value.text,
        after == null ? void 0 : after(config2.value)
      ];
      return isInternal.value ? h$1(
        RouteLink,
        {
          "class": "auto-link",
          "to": config2.value.link,
          "active": isActive.value,
          "aria-label": linkAriaLabel.value
        },
        () => content
      ) : h$1(
        "a",
        {
          "class": "auto-link external-link",
          "href": config2.value.link,
          "aria-label": linkAriaLabel.value,
          "rel": linkRel.value,
          "target": linkTarget.value
        },
        content
      );
    };
  }
});
var ClientOnly = defineComponent({
  name: "ClientOnly",
  setup(_, ctx) {
    const isMounted = ref(false);
    onMounted(() => {
      isMounted.value = true;
    });
    return () => {
      var _a, _b;
      return isMounted.value ? (_b = (_a = ctx.slots).default) == null ? void 0 : _b.call(_a) : null;
    };
  }
});
var runContentUpdatedCallbacks = (reason) => {
  contentUpdatedCallbacks.value.forEach((fn) => fn(reason));
};
var Content = defineComponent({
  name: "Content",
  props: {
    path: {
      type: String,
      required: false,
      default: ""
    }
  },
  setup(props) {
    const pageComponent = usePageComponent();
    const ContentComponent = computed(() => {
      if (!props.path) return pageComponent.value;
      const route = resolveRoute(props.path);
      return defineAsyncComponent(
        async () => route.loader().then(({ comp }) => comp)
      );
    });
    const frontmatter = usePageFrontmatter();
    watch(
      frontmatter,
      () => {
        runContentUpdatedCallbacks("updated");
      },
      { deep: true, flush: "post" }
    );
    return () => h$1(ContentComponent.value, {
      onVnodeMounted: () => {
        runContentUpdatedCallbacks("mounted");
      },
      onVnodeUpdated: () => {
        runContentUpdatedCallbacks("updated");
      },
      onVnodeBeforeUnmount: () => {
        runContentUpdatedCallbacks("beforeUnmount");
      }
    });
  }
});
var LAYOUT_NAME_DEFAULT = "Layout";
var LANG_DEFAULT = "en-US";
var resolvers = reactive({
  /**
   * Resolve layouts component map
   */
  resolveLayouts: (clientConfigs2) => clientConfigs2.reduce(
    (prev, item) => ({
      ...prev,
      ...item.layouts
    }),
    {}
  ),
  /**
   * Merge the head config in frontmatter and site locale
   *
   * Frontmatter should take priority over site locale
   */
  resolvePageHead: (pageHeadTitle, pageFrontmatter, siteLocaleDate) => {
    const description = isString(pageFrontmatter.description) ? pageFrontmatter.description : siteLocaleDate.description;
    const head = [
      ...Array.isArray(pageFrontmatter.head) ? pageFrontmatter.head : [],
      ...siteLocaleDate.head,
      ["title", {}, pageHeadTitle],
      ["meta", { name: "description", content: description }]
    ];
    return dedupeHead(head);
  },
  /**
   * Resolve the content of page head title
   *
   * It would be used as the content of the `<title>` tag
   */
  resolvePageHeadTitle: (pageData, siteLocaleDate) => [pageData.title, siteLocaleDate.title].filter((item) => !!item).join(" | "),
  /**
   * Resolve page language from page data
   *
   * It would be used as the `lang` attribute of `<html>` tag
   */
  resolvePageLang: (pageData, siteLocaleData) => pageData.lang || siteLocaleData.lang || LANG_DEFAULT,
  /**
   * Resolve layout component of current page
   */
  resolvePageLayout: (pageData, layouts) => {
    const layoutName = isString(pageData.frontmatter.layout) ? pageData.frontmatter.layout : LAYOUT_NAME_DEFAULT;
    if (!layouts[layoutName]) {
      throw new Error(`[vuepress] Cannot resolve layout: ${layoutName}`);
    }
    return layouts[layoutName];
  },
  /**
   * Resolve locale path according to route path and locales config
   */
  resolveRouteLocale: (locales2, routePath) => resolveLocalePath(locales2, decodeURI(routePath)),
  /**
   * Resolve site data for specific locale
   *
   * It would merge the locales fields to the root fields
   */
  resolveSiteLocaleData: ({ base, locales: locales2, ...siteData2 }, routeLocale) => {
    var _a;
    return {
      ...siteData2,
      ...locales2[routeLocale],
      head: [
        // when merging head, the locales head should be placed before root head
        // to get higher priority
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition -- unsafe indexed access
        ...((_a = locales2[routeLocale]) == null ? void 0 : _a.head) ?? [],
        ...siteData2.head
      ]
    };
  }
});
var defineClientConfig = (clientConfig = {}) => clientConfig;
var withBase = (url) => {
  if (isLinkHttp(url)) return url;
  return `${"/"}${removeLeadingSlash(url)}`;
};
var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var constants_exports = {};
__export(constants_exports, {
  COMPONENT_STATE_TYPE: () => COMPONENT_STATE_TYPE,
  INSPECTOR_ID: () => INSPECTOR_ID,
  INSPECTOR_LABEL: () => INSPECTOR_LABEL,
  INSPECTOR_NODES: () => INSPECTOR_NODES,
  INSPECTOR_STATE_SECTION_NAME: () => INSPECTOR_STATE_SECTION_NAME,
  PLUGIN_ID: () => PLUGIN_ID$1,
  PLUGIN_LABEL: () => PLUGIN_LABEL$1
});
var PLUGIN_ID$1 = "org.vuejs.vuepress";
var PLUGIN_LABEL$1 = "VuePress";
var COMPONENT_STATE_TYPE = PLUGIN_LABEL$1;
var INSPECTOR_ID = PLUGIN_ID$1;
var INSPECTOR_LABEL = PLUGIN_LABEL$1;
var INSPECTOR_NODE_INTERNAL = {
  id: "INTERNAL",
  label: "Internal",
  keys: ["layouts", "routes", "redirects"]
};
var INSPECTOR_NODE_SITE = {
  id: "SITE",
  label: "Site",
  keys: ["siteData", "siteLocaleData"]
};
var INSPECTOR_NODE_ROUTE = {
  id: "ROUTE",
  label: "Route",
  keys: ["routePath", "routeLocale"]
};
var INSPECTOR_NODE_PAGE = {
  id: "PAGE",
  label: "Page",
  keys: [
    "pageData",
    "pageFrontmatter",
    "pageLang",
    "pageHead",
    "pageHeadTitle",
    "pageLayout",
    "pageComponent"
  ]
};
var INSPECTOR_NODES = {
  [INSPECTOR_NODE_INTERNAL.id]: INSPECTOR_NODE_INTERNAL,
  [INSPECTOR_NODE_SITE.id]: INSPECTOR_NODE_SITE,
  [INSPECTOR_NODE_ROUTE.id]: INSPECTOR_NODE_ROUTE,
  [INSPECTOR_NODE_PAGE.id]: INSPECTOR_NODE_PAGE
};
var INSPECTOR_STATE_SECTION_NAME = "State";
const config$a = defineClientConfig({
  setup() {
  }
});
const clientConfig0 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$a
}, Symbol.toStringTag, { value: "Module" }));
const FadeInExpandTransition = defineComponent({
  name: "FadeInExpandTransition",
  props: {
    /* Whether to group transitions */
    group: Boolean,
    appear: Boolean,
    /* Whether to switch to the transition of `width` */
    width: Boolean,
    mode: String,
    onLeave: Function,
    onAfterLeave: Function,
    onAfterEnter: Function
  },
  setup(props, { slots }) {
    const handleBeforeLeave = (el) => {
      if (props.width) {
        el.style.maxWidth = `${el.offsetWidth}px`;
      } else {
        el.style.maxHeight = `${el.offsetHeight}px`;
      }
      void el.offsetWidth;
    };
    const handleLeave = (el) => {
      var _a;
      if (props.width) {
        el.style.maxWidth = "0";
      } else {
        el.style.maxHeight = "0";
      }
      void el.offsetWidth;
      (_a = props.onLeave) == null ? void 0 : _a.call(props);
    };
    const handleAfterLeave = (el) => {
      var _a;
      if (props.width) {
        el.style.maxWidth = "";
      } else {
        el.style.maxHeight = "";
      }
      (_a = props.onAfterLeave) == null ? void 0 : _a.call(props);
    };
    const handleEnter = (el) => {
      el.style.transition = "none";
      if (props.width) {
        const memorizedWidth = el.offsetWidth;
        el.style.maxWidth = "0";
        void el.offsetWidth;
        el.style.transition = "";
        el.style.maxWidth = `${memorizedWidth}px`;
      } else {
        const memorizedHeight = el.offsetHeight;
        el.style.maxHeight = "0";
        void el.offsetWidth;
        el.style.transition = "";
        el.style.maxHeight = `${memorizedHeight}px`;
      }
      void el.offsetWidth;
    };
    const handleAfterEnter = (el) => {
      var _a;
      if (props.width) {
        el.style.maxWidth = "";
      } else {
        el.style.maxHeight = "";
      }
      (_a = props.onAfterEnter) == null ? void 0 : _a.call(props);
    };
    return () => h$1(props.group ? TransitionGroup : Transition, {
      name: props.width ? "fade-in-width-expand" : "fade-in-height-expand",
      appear: props.appear,
      onEnter: handleEnter,
      onAfterEnter: handleAfterEnter,
      onBeforeLeave: handleBeforeLeave,
      onLeave: handleLeave,
      onAfterLeave: handleAfterLeave,
      ...props.group ? void 0 : { mode: props.mode }
    }, slots);
  }
});
const DEFAULT_HEADER_SELECTOR = [...new Array(6)].map((_, i2) => `[vp-content] h${i2 + 1}`).join(",");
const resolveHeaders = (headers, levels = 2) => {
  if (levels === false) {
    return [];
  }
  const [high, low] = typeof levels === "number" ? [levels, levels] : levels === "deep" ? [2, 6] : levels;
  const allowedHeaders = headers.filter((header) => header.level >= high && header.level <= low);
  const result = [];
  outer: for (let i2 = 0; i2 < allowedHeaders.length; i2++) {
    const current = allowedHeaders[i2];
    if (i2 === 0) {
      result.push(current);
    } else {
      for (let j = i2 - 1; j >= 0; j--) {
        const prev = allowedHeaders[j];
        if (prev.level < current.level) {
          prev.children.push(current);
          continue outer;
        }
      }
      result.push(current);
    }
  }
  return result;
};
const serializeHeader = (h2, ignore = []) => {
  let text;
  if (ignore.length) {
    const clone = h2.cloneNode(true);
    clone.querySelectorAll(ignore.join(",")).forEach((el) => {
      el.remove();
    });
    text = clone.textContent || "";
  } else {
    text = h2.textContent || "";
  }
  return text.trim();
};
const getHeadersFromDom = (selector = DEFAULT_HEADER_SELECTOR, ignore = []) => Array.from(document.querySelectorAll(selector)).filter((el) => el.id && el.hasChildNodes()).map((el) => ({
  element: el,
  title: serializeHeader(el, ignore),
  link: `#${el.id}`,
  slug: el.id,
  level: Number(el.tagName[1]),
  children: []
}));
const getHeaders = ({ selector = DEFAULT_HEADER_SELECTOR, levels = 2, ignore = [] } = {}) => resolveHeaders(getHeadersFromDom(selector, ignore), levels);
const hasGlobalComponent = (name, app) => {
  var _a;
  const globalComponents = (_a = getCurrentInstance()) == null ? void 0 : _a.appContext.components;
  if (!globalComponents)
    return false;
  return name in globalComponents || camelize(name) in globalComponents || capitalize(camelize(name)) in globalComponents;
};
const isDef = (val) => typeof val !== "undefined";
const startsWith = (str, prefix) => isString(str) && str.startsWith(prefix);
const { keys } = Object;
const isLinkAbsolute = (test) => startsWith(test, "/") && test[1] !== "/";
const isLinkRelative = (link) => !isLinkExternal(link) && !isLinkWithProtocol(link);
const useLocaleConfig = (localesConfig) => {
  const routeLocale = useRouteLocale();
  return computed(() => toValue(localesConfig)[routeLocale.value] ?? {});
};
const useRoutePaths = () => {
  const routes2 = useRoutes();
  return computed(() => Object.keys(routes2.value));
};
var define_BACK_TO_TOP_LOCALES_default = { "/": { backToTop: "返回顶部" }, "/en/": { backToTop: "Back to top" } };
const BackToTop = defineComponent({
  name: "BackToTop",
  setup() {
    const pageFrontmatter = usePageFrontmatter();
    const locale = useLocaleConfig(define_BACK_TO_TOP_LOCALES_default);
    const body = shallowRef();
    const { height: bodyHeight } = useElementSize(body);
    const { height: windowHeight } = useWindowSize();
    const { y } = useWindowScroll();
    const show = computed(() => (pageFrontmatter.value.backToTop ?? true) && y.value > 100);
    const progress = computed(() => y.value / (bodyHeight.value - windowHeight.value) * 100);
    onMounted(() => {
      body.value = document.body;
    });
    return () => h$1(Transition, { name: "fade-in" }, () => show.value ? h$1("button", {
      "type": "button",
      "class": "vp-back-to-top-button",
      "aria-label": locale.value.backToTop,
      "onClick": () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    }, [
      h$1("span", {
        "class": "vp-scroll-progress",
        "role": "progressbar",
        "aria-labelledby": "loadinglabel",
        "aria-valuenow": progress.value
      }, h$1("svg", h$1("circle", {
        "cx": "26",
        "cy": "26",
        "r": "24",
        "fill": "none",
        "stroke": "currentColor",
        "stroke-width": "4",
        "stroke-dasharray": `${Math.PI * progress.value * 0.48} ${Math.PI * (100 - progress.value) * 0.48}`
      }))),
      h$1("div", { class: "back-to-top-icon" })
    ]) : null);
  }
});
const config$9 = defineClientConfig({
  rootComponents: [BackToTop]
});
const clientConfig1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$9
}, Symbol.toStringTag, { value: "Module" }));
const config$8 = defineClientConfig({
  setup: () => {
  }
});
const clientConfig2 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$8
}, Symbol.toStringTag, { value: "Module" }));
const config$7 = defineClientConfig({
  setup() {
    useEventListener("beforeprint", () => {
      document.querySelectorAll("details").forEach((detail) => {
        detail.open = true;
      });
    }, { passive: true });
  }
});
const clientConfig3 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$7
}, Symbol.toStringTag, { value: "Module" }));
var define_GIT_OPTIONS_default = { provider: "github", pattern: { commit: ":repo/commit/:hash", issue: ":repo/issues/:issue", tag: ":repo/releases/tag/:tag" }, repo: "" };
const gitOptions = typeof define_GIT_OPTIONS_default === "undefined" ? {} : define_GIT_OPTIONS_default;
const resolveRepoLink = (link, provider) => {
  if (!link || isLinkHttp(link))
    return link;
  if (provider === "github")
    return `https://github.com/${link}`;
  if (provider === "gitee")
    return `https://gitee.com/${link}`;
  return link;
};
const RE_ISSUE = /#(\d+)/g;
const useChangelog = (enabled = true) => {
  const frontmatter = usePageFrontmatter();
  const lang = usePageLang();
  const page = usePageData();
  const { pattern = {}, provider } = gitOptions;
  const repo = resolveRepoLink(gitOptions.repo, provider);
  return computed(() => {
    var _a;
    if (frontmatter.value.changelog === false || !toValue(enabled))
      return [];
    const formatter = new Intl.DateTimeFormat(lang.value, {
      dateStyle: "short"
    });
    return (((_a = page.value.git) == null ? void 0 : _a.changelog) ?? []).map((item) => {
      const res = {
        date: formatter.format(item.time),
        ...item
      };
      if (pattern.issue && repo) {
        res.message = res.message.replace(RE_ISSUE, (matched, issue) => {
          const url = pattern.issue.replace(":issue", issue).replace(":repo", repo);
          return `<a href="${url}" target="_blank" rel="noopener noreferrer">${matched}</a>`;
        });
      }
      if (pattern.commit && repo) {
        res.commitUrl = pattern.commit.replace(":hash", res.hash).replace(":repo", repo);
      }
      if (pattern.tag && repo && res.tag)
        res.tagUrl = pattern.tag.replace(":tag", res.tag).replace(":repo", repo);
      return res;
    });
  });
};
const useContributors = (enabled = true) => {
  const frontmatter = usePageFrontmatter();
  const page = usePageData();
  return computed(() => {
    var _a;
    if (frontmatter.value.contributors === false || !toValue(enabled))
      return [];
    return ((_a = page.value.git) == null ? void 0 : _a.contributors) ?? [];
  });
};
var define_GIT_LOCALES_default = { "/": { contributors: "贡献者", changelog: "更新日志", timeOn: "于", viewChangelog: "查看所有更新日志", latestUpdateAt: "最近更新：" }, "/en/": { contributors: "Contributors", changelog: "Changelog", timeOn: "on", viewChangelog: "View All Changelog", latestUpdateAt: "Last Updated:" } };
const locales = typeof define_GIT_LOCALES_default === "undefined" ? {} : define_GIT_LOCALES_default;
const useGitLocaleConfig = () => useLocaleConfig(locales);
const useLastUpdated = (enabled = true) => {
  const lang = usePageLang();
  const locale = useGitLocaleConfig();
  const page = usePageData();
  return computed(() => {
    var _a, _b, _c;
    if (!toValue(enabled))
      return null;
    const timeStamp = (
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      ((_a = page.value.git) == null ? void 0 : _a.updatedTime) ?? ((_c = (_b = page.value.git) == null ? void 0 : _b.changelog) == null ? void 0 : _c[0].time)
    );
    if (!timeStamp)
      return null;
    const date = new Date(timeStamp);
    const text = new Intl.DateTimeFormat(lang.value, {
      dateStyle: "short",
      timeStyle: "short"
    }).format(timeStamp);
    return {
      date,
      text,
      iso: date.toISOString(),
      locale: locale.value.latestUpdateAt
    };
  });
};
const VPHeader = ({ level = 2, text, anchor }) => h$1(`h${level || 2}`, { id: anchor, tabindex: "-1" }, h$1("a", { href: `#${anchor}`, class: "header-anchor" }, h$1("span", text)));
const GitContributor = ({ name, url, avatar }) => h$1(url ? "a" : "span", {
  href: url,
  target: "_blank",
  rel: "noreferrer",
  class: "vp-contributor"
}, [
  avatar ? h$1("img", { src: avatar, alt: "", class: "vp-contributor-avatar" }) : null,
  h$1("span", { class: "vp-contributor-name" }, name)
]);
const GitContributors = defineComponent({
  name: "GitContributors",
  props: {
    /** Contributor title */
    title: String,
    /** header level of contributor title */
    headerLevel: {
      type: Number,
      default: 2
    }
  },
  setup(props) {
    const contributors = useContributors();
    const locale = useGitLocaleConfig();
    return () => contributors.value.length ? [
      h$1(VPHeader, {
        level: props.headerLevel,
        anchor: "doc-contributors",
        text: props.title || locale.value.contributors
      }),
      h$1("div", { class: "vp-contributors" }, contributors.value.map((item) => h$1(GitContributor, item)))
    ] : null;
  }
});
const GitChangelog = defineComponent({
  name: "GitChangelog",
  props: {
    /** Title of changelog */
    title: String,
    /** header level of changelog */
    headerLevel: {
      type: Number,
      default: 2
    }
  },
  setup(props) {
    const changelog = useChangelog();
    const locale = useGitLocaleConfig();
    const lastUpdated = useLastUpdated();
    const [active, toggleActive] = useToggle();
    const ChangelogHeader = () => h$1("div", { class: "vp-changelog-header", onClick: () => toggleActive() }, [
      h$1("div", { class: "vp-latest-updated" }, [
        h$1("span", { class: "vp-changelog-icon" }),
        h$1("span", { "data-allow-mismatch": "" }, lastUpdated.value.text)
      ]),
      h$1("div", [
        h$1("span", { class: "vp-changelog-menu-icon" }),
        h$1("span", locale.value.viewChangelog)
      ])
    ]);
    const ReleaseTag = ({ item }) => h$1("li", { class: "vp-changelog-item-tag" }, h$1("div", [
      h$1("a", { class: "vp-changelog-tag" }, h$1("code", item.tag)),
      h$1("span", { "class": "vp-changelog-date", "data-allow-mismatch": "" }, [
        locale.value.timeOn,
        " ",
        h$1("time", { datetime: new Date(item.time).toISOString() }, item.date)
      ])
    ]));
    const Commit = ({ item }) => h$1("li", { class: "vp-changelog-item-commit" }, [
      h$1(item.commitUrl ? "a" : "span", {
        class: "vp-changelog-hash",
        href: item.commitUrl,
        target: "_blank",
        rel: "noreferrer"
      }, [h$1("code", item.hash.slice(0, 5))]),
      h$1("span", { class: "vp-changelog-divider" }, "-"),
      h$1("span", { class: "vp-changelog-message", innerHTML: item.message }),
      h$1("span", { "class": "vp-changelog-date", "data-allow-mismatch": "" }, [
        locale.value.timeOn || "on",
        " ",
        h$1("time", { datetime: new Date(item.time).toISOString() }, item.date)
      ])
    ]);
    return () => changelog.value.length ? [
      h$1(VPHeader, {
        level: props.headerLevel,
        anchor: "doc-changelog",
        text: props.title || locale.value.changelog
      }),
      h$1("div", { class: ["vp-changelog-wrapper", { active: active.value }] }, [
        h$1(ChangelogHeader),
        h$1("ul", { class: "vp-changelog-list" }, [
          changelog.value.map((item) => item.tag ? h$1(ReleaseTag, { item, key: item.tag }) : h$1(Commit, { item, key: item.hash }))
        ])
      ])
    ] : null;
  }
});
const config$6 = {
  enhance: ({ app }) => {
    app.component("GitContributors", GitContributors);
    app.component("GitChangelog", GitChangelog);
  }
};
const clientConfig4 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$6
}, Symbol.toStringTag, { value: "Module" }));
const useMediumZoom = () => {
  return null;
};
const config$5 = defineClientConfig({
  enhance({ app }) {
    return;
  },
  setup() {
    const zoom = useMediumZoom();
    onContentUpdated((reason) => {
      if (reason !== "beforeUnmount")
        zoom.refresh();
    });
  }
});
const clientConfig5 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$5
}, Symbol.toStringTag, { value: "Module" }));
/**
 * NProgress, (c) 2013, 2014 Rico Sta. Cruz - http://ricostacruz.com/nprogress
 * @license MIT
 */
const addClass = (element, name) => {
  element.classList.add(name);
};
const removeClass = (element, name) => {
  element.classList.remove(name);
};
const removeElement = (element) => {
  var _a;
  (_a = element == null ? void 0 : element.parentNode) == null ? void 0 : _a.removeChild(element);
};
const clamp = (n2, min, max) => {
  if (n2 < min)
    return min;
  if (n2 > max)
    return max;
  return n2;
};
const toBarPercent = (n2) => (-1 + n2) * 100;
const queue = /* @__PURE__ */ (() => {
  const pending = [];
  const nextStep = () => {
    const fn = pending.shift();
    if (fn) {
      fn(nextStep);
    }
  };
  return (fn) => {
    pending.push(fn);
    if (pending.length === 1)
      nextStep();
  };
})();
const camelCase = (content) => content.replace(/^-ms-/, "ms-").replace(/-([\da-z])/gi, (_, letter) => letter.toUpperCase());
const addStyle = /* @__PURE__ */ (() => {
  const cssPrefixes = ["Webkit", "O", "Moz", "ms"];
  const cssProps = {};
  const getVendorProp = (name) => {
    const { style } = document.body;
    if (name in style)
      return name;
    const capName = name.charAt(0).toUpperCase() + name.slice(1);
    let index = cssPrefixes.length;
    while (index--) {
      const vendorName = `${cssPrefixes[index]}${capName}`;
      if (vendorName in style)
        return vendorName;
    }
    return name;
  };
  const getStyleProp = (name) => {
    const finalizedName = camelCase(name);
    return cssProps[finalizedName] ?? (cssProps[finalizedName] = getVendorProp(finalizedName));
  };
  const applyCss = (element, prop, value) => {
    element.style[getStyleProp(prop)] = value;
  };
  return (element, properties) => {
    for (const prop in properties) {
      const value = properties[prop];
      if (Object.hasOwn(properties, prop) && isDef(value))
        applyCss(element, prop, value);
    }
  };
})();
const SETTINGS = {
  minimum: 0.08,
  easing: "ease",
  speed: 200,
  trickleRate: 0.02,
  trickleSpeed: 800,
  barSelector: '[role="bar"]',
  parent: "body",
  template: '<div class="bar" role="bar"></div>'
};
const nprogress = {
  percent: null,
  isRendered: () => Boolean(document.getElementById("nprogress")),
  set: (progress) => {
    const { speed, easing } = SETTINGS;
    const inProgress = nprogress.isStarted();
    const newPercent = clamp(progress, SETTINGS.minimum, 1);
    nprogress.percent = newPercent === 1 ? null : newPercent;
    const nprogressElement = nprogress.render(!inProgress);
    const barElement = nprogressElement.querySelector(SETTINGS.barSelector);
    nprogressElement.offsetWidth;
    queue((next) => {
      addStyle(barElement, {
        transform: `translate3d(${toBarPercent(newPercent)}%,0,0)`,
        transition: `all ${speed}ms ${easing}`
      });
      if (newPercent === 1) {
        addStyle(nprogressElement, {
          transition: "none",
          opacity: "1"
        });
        nprogressElement.offsetWidth;
        setTimeout(() => {
          addStyle(nprogressElement, {
            transition: `all ${speed}ms linear`,
            opacity: "0"
          });
          setTimeout(() => {
            nprogress.remove();
            next();
          }, speed);
        }, speed);
      } else {
        setTimeout(() => {
          next();
        }, speed);
      }
    });
    return nprogress;
  },
  isStarted: () => typeof nprogress.percent === "number",
  start: () => {
    if (!nprogress.percent)
      nprogress.set(0);
    const work = () => {
      setTimeout(() => {
        if (!nprogress.percent)
          return;
        nprogress.trickle();
        work();
      }, SETTINGS.trickleSpeed);
    };
    work();
    return nprogress;
  },
  done: (force) => {
    if (!force && !nprogress.percent)
      return nprogress;
    return nprogress.increase(0.3 + 0.5 * Math.random()).set(1);
  },
  increase: (amount) => {
    let { percent } = nprogress;
    if (!percent) {
      return nprogress.start();
    }
    percent = clamp(percent + (typeof amount === "number" ? amount : (1 - percent) * clamp(Math.random() * percent, 0.1, 0.95)), 0, 0.994);
    return nprogress.set(percent);
  },
  trickle: () => nprogress.increase(Math.random() * SETTINGS.trickleRate),
  render: (fromStart) => {
    if (nprogress.isRendered()) {
      return document.getElementById("nprogress");
    }
    addClass(document.documentElement, "nprogress-busy");
    const nprogressElement = document.createElement("div");
    nprogressElement.id = "nprogress";
    nprogressElement.innerHTML = SETTINGS.template;
    const barElement = nprogressElement.querySelector(SETTINGS.barSelector);
    const parentElement = document.querySelector(SETTINGS.parent);
    const percent = fromStart ? "-100" : toBarPercent(nprogress.percent ?? 0);
    addStyle(barElement, {
      transition: "all 0 linear",
      transform: `translate3d(${percent}%,0,0)`
    });
    if (parentElement) {
      if (parentElement !== document.body) {
        addClass(parentElement, "nprogress-custom-parent");
      }
      parentElement.appendChild(nprogressElement);
    }
    return nprogressElement;
  },
  remove: () => {
    removeClass(document.documentElement, "nprogress-busy");
    removeClass(document.querySelector(SETTINGS.parent), "nprogress-custom-parent");
    removeElement(document.getElementById("nprogress"));
  }
};
const useNprogress = () => {
  onMounted(() => {
    const router = useRouter();
    const loadedPages = /* @__PURE__ */ new Set();
    loadedPages.add(router.currentRoute.value.path);
    router.beforeEach((to) => {
      if (!loadedPages.has(to.path)) {
        nprogress.start();
      }
    });
    router.afterEach((to) => {
      loadedPages.add(to.path);
      nprogress.done();
    });
  });
};
const config$4 = defineClientConfig({
  setup() {
    useNprogress();
  }
});
const clientConfig6 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$4
}, Symbol.toStringTag, { value: "Module" }));
const setupCollapsedLines = ({ selector = 'div[class*="language-"].has-collapsed-lines > .collapsed-lines' } = {}) => {
  useEventListener("click", (e) => {
    const target = e.target;
    if (target.matches(selector)) {
      const parent = target.parentElement;
      if (parent == null ? void 0 : parent.classList.toggle("collapsed")) {
        parent.scrollIntoView({ block: "center", behavior: "instant" });
      }
    }
  }, { passive: true });
};
const config$3 = {
  setup() {
    setupCollapsedLines();
  }
};
const clientConfig7 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$3
}, Symbol.toStringTag, { value: "Module" }));
const A = "VUEPRESS_CODE_TAB_STORE", i = useStorage(A, {}), S = defineComponent({ name: "CodeTabs", props: { active: { type: Number, default: 0 }, data: { type: Array, required: true }, id: { type: String, required: true }, tabId: String }, slots: Object, setup(a, { slots: r }) {
  const l = ref(a.active), o = shallowRef([]), n2 = () => {
    a.tabId && (i.value[a.tabId] = a.data[l.value].id);
  }, s = (e = l.value) => {
    l.value = e < o.value.length - 1 ? e + 1 : 0, o.value[l.value].focus();
  }, c = (e = l.value) => {
    l.value = e > 0 ? e - 1 : o.value.length - 1, o.value[l.value].focus();
  }, b = (e, t) => {
    e.key === " " || e.key === "Enter" ? (e.preventDefault(), l.value = t) : e.key === "ArrowRight" ? (e.preventDefault(), s()) : e.key === "ArrowLeft" && (e.preventDefault(), c()), a.tabId && (i.value[a.tabId] = a.data[l.value].id);
  }, p = () => {
    if (a.tabId) {
      const e = a.data.findIndex(({ id: t }) => i.value[a.tabId] === t);
      if (e !== -1) return e;
    }
    return a.active;
  };
  return onMounted(() => {
    l.value = p(), watch(() => a.tabId && i.value[a.tabId], (e, t) => {
      if (a.tabId && e !== t) {
        const d = a.data.findIndex(({ id: v }) => v === e);
        d !== -1 && (l.value = d);
      }
    });
  }), () => a.data.length ? h$1("div", { class: "vp-code-tabs" }, [h$1("div", { class: "vp-code-tabs-nav", role: "tablist" }, a.data.map(({ id: e }, t) => {
    const d = t === l.value;
    return h$1("button", { type: "button", ref: (v) => {
      v && (o.value[t] = v);
    }, class: ["vp-code-tab-nav", { active: d }], role: "tab", "aria-controls": `codetab-${a.id}-${t}`, "aria-selected": d, onClick: () => {
      l.value = t, n2();
    }, onKeydown: (v) => {
      b(v, t);
    } }, r[`title${t}`]({ value: e, isActive: d }));
  })), a.data.map(({ id: e }, t) => {
    const d = t === l.value;
    return h$1("div", { class: ["vp-code-tab", { active: d }], id: `codetab-${a.id}-${t}`, role: "tabpanel", "aria-expanded": d }, [h$1("div", { class: "vp-code-tab-title" }, r[`title${t}`]({ value: e, isActive: d })), r[`tab${t}`]({ value: e, isActive: d })]);
  })]) : null;
} });
const h = "VUEPRESS_TAB_STORE", n = useStorage(h, {}), $ = defineComponent({ name: "Tabs", props: { active: { type: Number, default: 0 }, data: { type: Array, required: true }, id: { type: String, required: true }, tabId: String }, slots: Object, setup(a, { slots: d }) {
  const l = ref(a.active), v = shallowRef([]), s = () => {
    a.tabId && (n.value[a.tabId] = a.data[l.value].id);
  }, o = (t = l.value) => {
    l.value = t < v.value.length - 1 ? t + 1 : 0, v.value[l.value].focus();
  }, c = (t = l.value) => {
    l.value = t > 0 ? t - 1 : v.value.length - 1, v.value[l.value].focus();
  }, b = (t, e) => {
    t.key === " " || t.key === "Enter" ? (t.preventDefault(), l.value = e) : t.key === "ArrowRight" ? (t.preventDefault(), o()) : t.key === "ArrowLeft" && (t.preventDefault(), c()), s();
  }, p = () => {
    if (a.tabId) {
      const t = a.data.findIndex(({ id: e }) => n.value[a.tabId] === e);
      if (t !== -1) return t;
    }
    return a.active;
  };
  return onMounted(() => {
    l.value = p(), watch(() => a.tabId && n.value[a.tabId], (t, e) => {
      if (a.tabId && t !== e) {
        const i2 = a.data.findIndex(({ id: r }) => r === t);
        i2 !== -1 && (l.value = i2);
      }
    });
  }), () => a.data.length ? h$1("div", { class: "vp-tabs" }, [h$1("div", { class: "vp-tabs-nav", role: "tablist" }, a.data.map(({ id: t }, e) => {
    const i2 = e === l.value;
    return h$1("button", { type: "button", ref: (r) => {
      r && (v.value[e] = r);
    }, class: ["vp-tab-nav", { active: i2 }], role: "tab", "aria-controls": `tab-${a.id}-${e}`, "aria-selected": i2, onClick: () => {
      l.value = e, s();
    }, onKeydown: (r) => {
      b(r, e);
    } }, d[`title${e}`]({ value: t, isActive: i2 }));
  })), a.data.map(({ id: t }, e) => {
    const i2 = e === l.value;
    return h$1("div", { class: ["vp-tab", { active: i2 }], id: `tab-${a.id}-${e}`, role: "tabpanel", "aria-expanded": i2 }, [h$1("div", { class: "vp-tab-title" }, d[`title${e}`]({ value: t, isActive: i2 })), d[`tab${e}`]({ value: t, isActive: i2 })]);
  })]) : null;
} });
const config$2 = {
  enhance: ({ app }) => {
    app.component("CodeTabs", S);
    app.component("Tabs", $);
  }
};
const clientConfig8 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$2
}, Symbol.toStringTag, { value: "Module" }));
const PLUGIN_ID = "org.vuejs.vuepress.plugin-theme-data";
const PLUGIN_LABEL = "VuePress Theme Data";
const INSPECTOR_THEME_DATA_ID = "THEME_DATA";
const INSPECTOR_THEME_DATA_LABEL = "Theme Data";
const setupDevTools = (app, themeData2, themeLocaleData) => {
  {
    setupDevtoolsPlugin({
      // fix recursive reference
      app,
      id: PLUGIN_ID,
      label: PLUGIN_LABEL,
      packageName: "@vuepress/plugin-theme-data",
      homepage: "https://vuepress.vuejs.org",
      logo: "https://vuepress.vuejs.org/images/hero.png",
      componentStateTypes: [constants_exports.COMPONENT_STATE_TYPE]
    }, (api) => {
      api.on.inspectComponent((payload) => {
        payload.instanceData.state.push({
          type: "VuePress",
          key: "themeData",
          editable: false,
          value: themeData2.value
        }, {
          type: "VuePress",
          key: "themeLocaleData",
          editable: false,
          value: themeLocaleData.value
        });
      });
      api.addInspector({
        id: constants_exports.INSPECTOR_ID,
        label: constants_exports.INSPECTOR_LABEL,
        icon: "article"
      });
      api.on.getInspectorTree((payload) => {
        if (payload.inspectorId !== constants_exports.INSPECTOR_ID)
          return;
        payload.rootNodes.push({
          id: INSPECTOR_THEME_DATA_ID,
          label: INSPECTOR_THEME_DATA_LABEL,
          children: [
            {
              id: "themeData",
              label: "themeData"
            },
            {
              id: "themeLocaleData",
              label: "themeLocaleData"
            }
          ]
        });
      });
      api.on.getInspectorState((payload) => {
        if (payload.inspectorId !== constants_exports.INSPECTOR_ID)
          return;
        if (payload.nodeId === INSPECTOR_THEME_DATA_ID) {
          payload.state[INSPECTOR_THEME_DATA_LABEL].push({
            key: "themeData",
            value: themeData2.value
          }, {
            key: "themeLocaleData",
            value: themeLocaleData.value
          });
        }
        if (["themeData", "themeLocaleData"].includes(payload.nodeId)) {
          payload.state = {
            [constants_exports.INSPECTOR_STATE_SECTION_NAME]: [
              {
                key: payload.nodeId,
                value: payload.nodeId === "themeData" ? themeData2.value : themeLocaleData.value
              }
            ]
          };
        }
      });
      watch([themeData2, themeLocaleData], () => {
        api.notifyComponentUpdate();
        api.sendInspectorState(constants_exports.INSPECTOR_ID);
      });
    });
  }
};
const themeData$1 = JSON.parse(`{"locales":{"/":{"navbar":[{"text":"首页","link":"/"},{"text":"数据挖掘概述","link":"/overview/"},{"text":"学习路径图","link":"/learning-path/"},{"text":"核心知识","children":[{"text":"数据预处理","children":["/core/preprocessing/data-presentation.md","/core/preprocessing/missing-values.md"]},{"text":"分类算法","children":["/core/classification/svm.md","/core/classification/naive-bayes.md","/core/classification/decision-trees.md"]},{"text":"聚类分析","children":["/core/clustering/kmeans.md","/core/clustering/evaluation.md","/core/clustering/applications.md"]},{"text":"预测与回归","children":["/core/regression/linear-regression.md","/core/regression/nonlinear-regression.md"]}]},{"text":"实践项目","link":"/projects/"},{"text":"关于","link":"/about/"}],"sidebar":{"/overview/":[{"text":"数据挖掘概述","children":["/overview/README.md","/overview/definition.md","/overview/process.md","/overview/applications.md","/overview/tools.md"]}],"/learning-path/":[{"text":"学习路径图","children":["/learning-path/README.md","/learning-path/beginner.md","/learning-path/advanced.md","/learning-path/practical.md"]}],"/core/preprocessing/":[{"text":"数据预处理","children":["/core/preprocessing/data-presentation.md","/core/preprocessing/missing-values.md"]}],"/core/classification/":[{"text":"分类算法","children":["/core/classification/svm.md","/core/classification/naive-bayes.md","/core/classification/decision-trees.md","/core/classification/comparison.md"]}],"/core/clustering/":[{"text":"聚类分析","children":["/core/clustering/kmeans.md","/core/clustering/evaluation.md","/core/clustering/applications.md"]}],"/core/regression/":[{"text":"预测与回归","children":["/core/regression/linear-regression.md","/core/regression/nonlinear-regression.md","/core/regression/evaluation-metrics.md"]}],"/projects/":[{"text":"实践项目","children":["/projects/README.md","/projects/preprocessing.md","/projects/classification.md","/projects/clustering.md","/projects/prediction.md"]}],"/resources/":[{"text":"学习资源","children":["/resources/README.md"]}],"/core/":[{"text":"数据预处理","children":["/core/preprocessing/data-presentation.md","/core/preprocessing/missing-values.md"]}]},"logo":"/images/大顿河.png","logoDark":"/images/大顿河.png","selectLanguageName":"简体中文","selectLanguageText":"选择语言"},"/en/":{"navbar":[{"text":"Home","link":"/en/"},{"text":"Overview","link":"/en/overview/"},{"text":"Learning Path","link":"/en/learning-path/"},{"text":"Core knowledge","children":[{"text":"Data preprocessing","children":["/en/core/preprocessing/data-presentation.md","/en/core/preprocessing/missing-values.md"]},{"text":"Classification algorithm","children":["/en/core/classification/svm.md","/en/core/classification/naive-bayes.md","/en/core/classification/decision-trees.md"]},{"text":"Cluster analysis","children":["/en/core/clustering/kmeans.md","/en/core/clustering/evaluation.md","/en/core/clustering/applications.md"]},{"text":"Prediction and regression","children":["/en/core/regression/linear-regression.md","/en/core/regression/nonlinear-regression.md"]}]},{"text":"Practical project","link":"/en/projects/"},{"text":"About","link":"/en/about/"}],"sidebar":{"/en/overview/":[{"text":"Data Mining Overview","children":["/en/overview/README.md","/en/overview/definition.md","/en/overview/process.md","/en/overview/applications.md","/en/overview/tools.md"]}],"/en/learning-path/":[{"text":"Learning Path","children":["/en/learning-path/README.md","/en/learning-path/beginner.md","/en/learning-path/advanced.md","/en/learning-path/practical.md"]}],"/en/core/preprocessing/":[{"text":"Data Preprocessing","children":["/en/core/preprocessing/data-presentation.md","/en/core/preprocessing/missing-values.md"]}],"/en/core/classification/":[{"text":"Classification Algorithms","children":["/en/core/classification/svm.md","/en/core/classification/naive-bayes.md","/en/core/classification/decision-trees.md","/en/core/classification/comparison.md"]}],"/en/core/clustering/":[{"text":"Clustering Analysis","children":["/en/core/clustering/kmeans.md","/en/core/clustering/evaluation.md","/en/core/clustering/applications.md"]}],"/en/core/regression/":[{"text":"Prediction and Regression","children":["/en/core/regression/linear-regression.md","/en/core/regression/nonlinear-regression.md","/en/core/regression/evaluation-metrics.md"]}],"/en/projects/":[{"text":"Projects","children":["/en/projects/README.md","/en/projects/preprocessing.md","/en/projects/classification.md","/en/projects/clustering.md","/en/projects/prediction.md"]}],"/en/resources/":[{"text":"Learning Resources","children":["/en/resources/README.md"]}],"/en/core/":[{"text":"Data Preprocessing","children":["/en/core/preprocessing/data-presentation.md","/en/core/preprocessing/missing-values.md"]}]},"logo":"/images/大顿河.png","logoDark":"/images/大顿河.png","selectLanguageName":"English","selectLanguageText":"Languages"}},"colorMode":"auto","colorModeSwitch":true,"navbar":[],"logo":null,"repo":null,"selectLanguageText":"Languages","selectLanguageAriaLabel":"Select language","sidebar":"heading","sidebarDepth":2,"editLink":true,"editLinkText":"Edit this page","lastUpdated":true,"contributors":true,"contributorsText":"Contributors","notFound":["There's nothing here.","How did we get here?","That's a Four-Oh-Four.","Looks like we've got some broken links."],"backToHome":"Take me home","openInNewWindow":"open in new window","toggleColorMode":"toggle color mode","toggleSidebar":"toggle sidebar"}`);
const themeData = ref(themeData$1);
const useThemeData = () => themeData;
const themeLocaleDataSymbol = Symbol("");
const useThemeLocaleData = () => {
  const themeLocaleData = inject(themeLocaleDataSymbol);
  if (!themeLocaleData) {
    throw new Error("useThemeLocaleData() is called without provider.");
  }
  return themeLocaleData;
};
const resolveThemeLocaleData = (theme, routeLocale) => {
  const { locales: locales2, ...baseOptions } = theme;
  return {
    ...baseOptions,
    ...locales2 == null ? void 0 : locales2[routeLocale]
  };
};
const config$1 = defineClientConfig({
  enhance({ app }) {
    const themeData2 = useThemeData();
    const clientData = app._context.provides[clientDataSymbol];
    const themeLocaleData = computed(() => resolveThemeLocaleData(themeData2.value, clientData.routeLocale.value));
    app.provide(themeLocaleDataSymbol, themeLocaleData);
    Object.defineProperties(app.config.globalProperties, {
      $theme: {
        get() {
          return themeData2.value;
        }
      },
      $themeLocale: {
        get() {
          return themeLocaleData.value;
        }
      }
    });
    setupDevTools(app, themeData2, themeLocaleData);
  }
});
const clientConfig9 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config$1
}, Symbol.toStringTag, { value: "Module" }));
const useData = () => {
  const { pageData, pageFrontmatter, pageLang, siteData: siteData2, siteLocaleData, ...rest } = useClientData();
  return {
    ...rest,
    page: pageData,
    frontmatter: pageFrontmatter,
    lang: pageLang,
    site: siteData2,
    siteLocale: siteLocaleData,
    theme: useThemeData(),
    themeLocale: useThemeLocaleData()
  };
};
const darkModeSymbol = Symbol("");
const applyDarkModeToHTML = (isDarkMode) => {
  const update = (value = isDarkMode.value) => {
    const el = window.document.documentElement;
    el.dataset.theme = value ? "dark" : "light";
  };
  onMounted(() => {
    watchImmediate(isDarkMode, update);
  });
  onUnmounted(() => {
    update();
  });
};
const useDarkMode = () => {
  const isDarkMode = inject(darkModeSymbol);
  if (!isDarkMode) {
    throw new Error("useDarkMode() is called without provider.");
  }
  return isDarkMode;
};
const setupDarkMode = () => {
  const { themeLocale } = useData();
  const isDarkPreferred = usePreferredDark();
  const darkStorage = useStorage("vuepress-color-scheme", themeLocale.value.colorMode);
  const isDarkMode = computed({
    get() {
      if (!themeLocale.value.colorModeSwitch) {
        return themeLocale.value.colorMode === "dark";
      }
      if (darkStorage.value === "auto") {
        return isDarkPreferred.value;
      }
      return darkStorage.value === "dark";
    },
    set(val) {
      if (val === isDarkPreferred.value) {
        darkStorage.value = "auto";
      } else {
        darkStorage.value = val ? "dark" : "light";
      }
    }
  });
  provide(darkModeSymbol, isDarkMode);
  applyDarkModeToHTML(isDarkMode);
};
const headersSymbol = Symbol("headers");
const useHeaders = () => {
  const headers = injectLocal(headersSymbol);
  if (!headers) {
    throw new Error("useHeaders() is called without provider.");
  }
  return headers;
};
const setupHeaders = () => {
  const { frontmatter, themeLocale } = useData();
  const headersRef = ref([]);
  const levels = computed(() => frontmatter.value.sidebarDepth ?? themeLocale.value.sidebarDepth ?? 2);
  const updateHeaders = () => {
    if (levels.value <= 0) {
      headersRef.value = [];
      return;
    }
    headersRef.value = getHeaders({
      levels: [2, levels.value + 1],
      ignore: [".vp-badge"]
    });
  };
  provideLocal(headersSymbol, headersRef);
  onContentUpdated((reason) => {
    if (reason === "beforeUnmount")
      headersRef.value = [];
    else
      updateHeaders();
  });
};
let promise = null;
let promiseResolve = null;
const scrollPromise = {
  wait: () => promise,
  pending: () => {
    promise = new Promise((resolve) => {
      promiseResolve = resolve;
    });
  },
  resolve: () => {
    promiseResolve == null ? void 0 : promiseResolve();
    promise = null;
    promiseResolve = null;
  }
};
const useScrollPromise = () => scrollPromise;
const resolveAutoLink = (config2, currentPath) => {
  const { notFound, meta, path } = resolveRoute(config2, currentPath);
  return notFound ? { text: path, link: path } : {
    text: meta.title || path,
    link: path
  };
};
const resolvePrefix = (prefix = "", path = "") => isLinkAbsolute(path) || isLinkWithProtocol(path) ? path : `${ensureEndingSlash(prefix)}${path}`;
const resolveSidebarPageHeader = (header) => ({
  text: header.title,
  link: header.link,
  // eslint-disable-next-line @typescript-eslint/no-use-before-define
  children: resolveSidebarPageHeaders(header.children)
});
const resolveSidebarPageHeaders = (headers) => headers ? headers.map((header) => resolveSidebarPageHeader(header)) : [];
const resolveSidebarHeadingItem = (page, headers) => [
  {
    text: page.title,
    children: resolveSidebarPageHeaders(headers)
  }
];
const resolveArraySidebarItems = (sidebarConfig, headers, path, prefix = "") => {
  const handleChildItem = (item, pathPrefix) => {
    var _a;
    const childItem = isString(item) ? resolveAutoLink(resolvePrefix(pathPrefix, item)) : isString(item.link) ? {
      ...item,
      link: isLinkRelative(item.link) ? resolveAutoLink(resolvePrefix(pathPrefix, item.link)).link : item.link
    } : item;
    if ("children" in childItem) {
      return {
        ...childItem,
        children: childItem.children.map((child) => handleChildItem(child, resolvePrefix(pathPrefix, childItem.prefix)))
      };
    }
    if (childItem.link === path) {
      const currentHeaders = ((_a = headers[0]) == null ? void 0 : _a.level) === 1 ? headers[0].children : headers;
      return {
        ...childItem,
        children: resolveSidebarPageHeaders(currentHeaders)
      };
    }
    return childItem;
  };
  return sidebarConfig.map((item) => handleChildItem(item, prefix));
};
const resolveMultiSidebarItems = (sidebarConfig, page, headers, path) => {
  const sidebarRoutes = keys(sidebarConfig).sort((x, y) => y.length - x.length);
  for (const base of sidebarRoutes)
    if (startsWith(decodeURI(path), base)) {
      const matched = sidebarConfig[base];
      return matched ? matched === "heading" ? resolveSidebarHeadingItem(page, headers) : resolveArraySidebarItems(matched, headers, path, base) : [];
    }
  console.warn(`${decodeURI(path)} is missing sidebar config.`);
  return [];
};
const sidebarItemsSymbol = Symbol("sidebarItems");
const useSidebarItems = () => {
  const sidebarItems = inject(sidebarItemsSymbol);
  if (!sidebarItems) {
    throw new Error("useSidebarItems() is called without provider.");
  }
  return sidebarItems;
};
const resolveSidebarItems = (sidebarConfig, page, path, routeLocale, headers) => {
  if (sidebarConfig === false) {
    return [];
  }
  if (sidebarConfig === "heading") {
    return resolveSidebarHeadingItem(page, headers);
  }
  if (Array.isArray(sidebarConfig)) {
    return resolveArraySidebarItems(sidebarConfig, headers, path, routeLocale);
  }
  if (isPlainObject(sidebarConfig)) {
    return resolveMultiSidebarItems(sidebarConfig, page, headers, path);
  }
  return [];
};
const setupSidebarItems = () => {
  const { frontmatter, page, routeLocale, themeLocale } = useData();
  const headers = useHeaders();
  const routePath = useRoutePath();
  const sidebarConfig = computed(() => frontmatter.value.home ? false : frontmatter.value.sidebar ?? themeLocale.value.sidebar ?? "heading");
  const sidebarItems = computed(() => resolveSidebarItems(sidebarConfig.value, page.value, routePath.value, routeLocale.value, headers.value));
  provide(sidebarItemsSymbol, sidebarItems);
};
const _sfc_main$p = /* @__PURE__ */ defineComponent({
  __name: "Badge",
  props: {
    type: { default: "tip" },
    text: { default: "" },
    vertical: { default: void 0 }
  },
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = {};
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
const _export_sfc = (sfc, props) => {
  const target = sfc.__vccOpts || sfc;
  for (const [key, val] of props) {
    target[key] = val;
  }
  return target;
};
function _sfc_ssrRender$p(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<span${ssrRenderAttrs(mergeProps({
    class: ["vp-badge", $props.type],
    style: {
      verticalAlign: $props.vertical
    }
  }, _attrs))}>`);
  ssrRenderSlot(_ctx.$slots, "default", {}, () => {
    _push(`${ssrInterpolate($props.text)}`);
  }, _push, _parent);
  _push(`</span>`);
}
const _sfc_setup$p = _sfc_main$p.setup;
_sfc_main$p.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/global/Badge.vue");
  return _sfc_setup$p ? _sfc_setup$p(props, ctx) : void 0;
};
const Badge = /* @__PURE__ */ _export_sfc(_sfc_main$p, [["ssrRender", _sfc_ssrRender$p], ["__file", "Badge.vue"]]);
const _sfc_main$o = /* @__PURE__ */ defineComponent({
  __name: "VPFadeSlideYTransition",
  emits: ["beforeEnter", "beforeLeave"],
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = {};
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$o(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  ssrRenderSlotInner(_ctx.$slots, "default", {}, null, _push, _parent, null, true);
}
const _sfc_setup$o = _sfc_main$o.setup;
_sfc_main$o.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPFadeSlideYTransition.vue");
  return _sfc_setup$o ? _sfc_setup$o(props, ctx) : void 0;
};
const VPFadeSlideYTransition = /* @__PURE__ */ _export_sfc(_sfc_main$o, [["ssrRender", _sfc_ssrRender$o], ["__file", "VPFadeSlideYTransition.vue"]]);
const _sfc_main$n = /* @__PURE__ */ defineComponent({
  __name: "VPHomeFeatures",
  setup(__props, { expose: __expose }) {
    __expose();
    const { frontmatter } = useData();
    const features = computed(() => frontmatter.value.features ?? []);
    const __returned__ = { frontmatter, features };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$n(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  if ($setup.features.length) {
    _push(`<div${ssrRenderAttrs(mergeProps({ class: "vp-features" }, _attrs))}><!--[-->`);
    ssrRenderList($setup.features, (feature) => {
      _push(`<div class="vp-feature"><h2>${ssrInterpolate(feature.title)}</h2><p>${ssrInterpolate(feature.details)}</p></div>`);
    });
    _push(`<!--]--></div>`);
  } else {
    _push(`<!---->`);
  }
}
const _sfc_setup$n = _sfc_main$n.setup;
_sfc_main$n.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPHomeFeatures.vue");
  return _sfc_setup$n ? _sfc_setup$n(props, ctx) : void 0;
};
const VPHomeFeatures = /* @__PURE__ */ _export_sfc(_sfc_main$n, [["ssrRender", _sfc_ssrRender$n], ["__file", "VPHomeFeatures.vue"]]);
const _sfc_main$m = /* @__PURE__ */ defineComponent({
  __name: "VPHomeFooter",
  setup(__props, { expose: __expose }) {
    __expose();
    const frontmatter = usePageFrontmatter();
    const footer = computed(() => frontmatter.value.footer);
    const footerHtml = computed(() => frontmatter.value.footerHtml);
    const __returned__ = { frontmatter, footer, footerHtml };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$m(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  if ($setup.footer) {
    _push(`<!--[-->`);
    if ($setup.footerHtml) {
      _push(`<div class="vp-footer" vp-footer>${$setup.footer ?? ""}</div>`);
    } else {
      _push(`<div class="vp-footer" vp-footer>${ssrInterpolate($setup.footer)}</div>`);
    }
    _push(`<!--]-->`);
  } else {
    _push(`<!---->`);
  }
}
const _sfc_setup$m = _sfc_main$m.setup;
_sfc_main$m.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPHomeFooter.vue");
  return _sfc_setup$m ? _sfc_setup$m(props, ctx) : void 0;
};
const VPHomeFooter = /* @__PURE__ */ _export_sfc(_sfc_main$m, [["ssrRender", _sfc_ssrRender$m], ["__file", "VPHomeFooter.vue"]]);
const _sfc_main$l = /* @__PURE__ */ defineComponent({
  __name: "VPAutoLink",
  props: {
    config: {}
  },
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = { get AutoLink() {
      return AutoLink;
    } };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$l(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(ssrRenderComponent($setup["AutoLink"], mergeProps({ config: $props.config }, _attrs), createSlots({
    before: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        ssrRenderSlot(_ctx.$slots, "before", $props.config, null, _push2, _parent2, _scopeId);
      } else {
        return [
          renderSlot(_ctx.$slots, "before", $props.config)
        ];
      }
    }),
    after: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        ssrRenderSlot(_ctx.$slots, "after", $props.config, null, _push2, _parent2, _scopeId);
      } else {
        return [
          renderSlot(_ctx.$slots, "after", $props.config)
        ];
      }
    }),
    _: 2
  }, [
    _ctx.$slots.default ? {
      name: "default",
      fn: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          ssrRenderSlot(_ctx.$slots, "default", $props.config, null, _push2, _parent2, _scopeId);
        } else {
          return [
            renderSlot(_ctx.$slots, "default", $props.config)
          ];
        }
      }),
      key: "0"
    } : void 0
  ]), _parent));
}
const _sfc_setup$l = _sfc_main$l.setup;
_sfc_main$l.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPAutoLink.vue");
  return _sfc_setup$l ? _sfc_setup$l(props, ctx) : void 0;
};
const VPAutoLink = /* @__PURE__ */ _export_sfc(_sfc_main$l, [["ssrRender", _sfc_ssrRender$l], ["__file", "VPAutoLink.vue"]]);
const _sfc_main$k = /* @__PURE__ */ defineComponent({
  __name: "VPHomeHero",
  setup(__props, { expose: __expose }) {
    __expose();
    const { frontmatter, siteLocale } = useData();
    const isDarkMode = useDarkMode();
    const heroText = computed(() => {
      if (frontmatter.value.heroText === null) {
        return null;
      }
      return frontmatter.value.heroText || siteLocale.value.title || "Hello";
    });
    const tagline = computed(() => {
      if (frontmatter.value.tagline === null) {
        return null;
      }
      return frontmatter.value.tagline || siteLocale.value.description || "Welcome to your VuePress site";
    });
    const heroImage = computed(() => {
      if (isDarkMode.value && frontmatter.value.heroImageDark !== void 0) {
        return frontmatter.value.heroImageDark;
      }
      return frontmatter.value.heroImage;
    });
    const heroAlt = computed(
      () => frontmatter.value.heroAlt || heroText.value || "hero"
    );
    const heroHeight = computed(() => frontmatter.value.heroHeight ?? 280);
    const actions = computed(() => {
      if (!Array.isArray(frontmatter.value.actions)) {
        return [];
      }
      return frontmatter.value.actions.map(({ type = "primary", ...rest }) => ({
        type,
        ...rest
      }));
    });
    const HomeHeroImage = () => {
      if (!heroImage.value) return null;
      const img = h$1("img", {
        class: "vp-hero-image",
        src: withBase(heroImage.value),
        alt: heroAlt.value,
        height: heroHeight.value
      });
      if (frontmatter.value.heroImageDark === void 0) {
        return img;
      }
      return h$1(ClientOnly, () => img);
    };
    const __returned__ = { frontmatter, siteLocale, isDarkMode, heroText, tagline, heroImage, heroAlt, heroHeight, actions, HomeHeroImage, VPAutoLink };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$k(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<header${ssrRenderAttrs(mergeProps({ class: "vp-hero" }, _attrs))}>`);
  _push(ssrRenderComponent($setup["HomeHeroImage"], null, null, _parent));
  if ($setup.heroText) {
    _push(`<h1 id="main-title">${ssrInterpolate($setup.heroText)}</h1>`);
  } else {
    _push(`<!---->`);
  }
  if ($setup.tagline) {
    _push(`<p class="vp-hero-description">${ssrInterpolate($setup.tagline)}</p>`);
  } else {
    _push(`<!---->`);
  }
  if ($setup.actions.length) {
    _push(`<p class="vp-hero-actions"><!--[-->`);
    ssrRenderList($setup.actions, (action) => {
      _push(ssrRenderComponent($setup["VPAutoLink"], {
        key: action.text,
        class: ["vp-hero-action-button", [action.type]],
        config: action
      }, null, _parent));
    });
    _push(`<!--]--></p>`);
  } else {
    _push(`<!---->`);
  }
  _push(`</header>`);
}
const _sfc_setup$k = _sfc_main$k.setup;
_sfc_main$k.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPHomeHero.vue");
  return _sfc_setup$k ? _sfc_setup$k(props, ctx) : void 0;
};
const VPHomeHero = /* @__PURE__ */ _export_sfc(_sfc_main$k, [["ssrRender", _sfc_ssrRender$k], ["__file", "VPHomeHero.vue"]]);
const _sfc_main$j = /* @__PURE__ */ defineComponent({
  __name: "VPHome",
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = { VPHomeFeatures, VPHomeFooter, VPHomeHero, get Content() {
      return Content;
    } };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$j(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<main${ssrRenderAttrs(mergeProps({ class: "vp-home" }, _attrs))}>`);
  _push(ssrRenderComponent($setup["VPHomeHero"], null, null, _parent));
  _push(ssrRenderComponent($setup["VPHomeFeatures"], null, null, _parent));
  _push(`<div vp-content>`);
  _push(ssrRenderComponent($setup["Content"], null, null, _parent));
  _push(`</div>`);
  _push(ssrRenderComponent($setup["VPHomeFooter"], null, null, _parent));
  _push(`</main>`);
}
const _sfc_setup$j = _sfc_main$j.setup;
_sfc_main$j.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPHome.vue");
  return _sfc_setup$j ? _sfc_setup$j(props, ctx) : void 0;
};
const VPHome = /* @__PURE__ */ _export_sfc(_sfc_main$j, [["ssrRender", _sfc_ssrRender$j], ["__file", "VPHome.vue"]]);
const _sfc_main$i = /* @__PURE__ */ defineComponent({
  __name: "VPNavbarBrand",
  setup(__props, { expose: __expose }) {
    __expose();
    const { routeLocale, siteLocale, themeLocale } = useData();
    const isDarkMode = useDarkMode();
    const navbarBrandLink = computed(
      () => themeLocale.value.home || routeLocale.value
    );
    const navbarBrandTitle = computed(() => siteLocale.value.title);
    const navbarBrandLogo = computed(() => {
      if (isDarkMode.value && themeLocale.value.logoDark !== void 0) {
        return themeLocale.value.logoDark;
      }
      return themeLocale.value.logo;
    });
    const navbarBrandLogoAlt = computed(
      () => themeLocale.value.logoAlt ?? navbarBrandTitle.value
    );
    const navBarLogoAltMatchesTitle = computed(
      () => navbarBrandTitle.value.toLocaleUpperCase().trim() === navbarBrandLogoAlt.value.toLocaleUpperCase().trim()
    );
    const NavbarBrandLogo = () => {
      if (!navbarBrandLogo.value) return null;
      const img = h$1("img", {
        class: "vp-site-logo",
        src: withBase(navbarBrandLogo.value),
        alt: navbarBrandLogoAlt.value
      });
      if (themeLocale.value.logoDark === void 0) {
        return img;
      }
      return h$1(ClientOnly, () => img);
    };
    const __returned__ = { routeLocale, siteLocale, themeLocale, isDarkMode, navbarBrandLink, navbarBrandTitle, navbarBrandLogo, navbarBrandLogoAlt, navBarLogoAltMatchesTitle, NavbarBrandLogo, get RouteLink() {
      return RouteLink;
    } };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$i(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(ssrRenderComponent($setup["RouteLink"], mergeProps({ to: $setup.navbarBrandLink }, _attrs), {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(ssrRenderComponent($setup["NavbarBrandLogo"], null, null, _parent2, _scopeId));
        if ($setup.navbarBrandTitle) {
          _push2(`<span class="${ssrRenderClass([{ "vp-hide-mobile": $setup.navbarBrandLogo }, "vp-site-name"])}"${ssrRenderAttr("aria-hidden", $setup.navBarLogoAltMatchesTitle)}${_scopeId}>${ssrInterpolate($setup.navbarBrandTitle)}</span>`);
        } else {
          _push2(`<!---->`);
        }
      } else {
        return [
          createVNode($setup["NavbarBrandLogo"]),
          $setup.navbarBrandTitle ? (openBlock(), createBlock("span", {
            key: 0,
            class: ["vp-site-name", { "vp-hide-mobile": $setup.navbarBrandLogo }],
            "aria-hidden": $setup.navBarLogoAltMatchesTitle
          }, toDisplayString($setup.navbarBrandTitle), 11, ["aria-hidden"])) : createCommentVNode("", true)
        ];
      }
    }),
    _: 1
  }, _parent));
}
const _sfc_setup$i = _sfc_main$i.setup;
_sfc_main$i.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPNavbarBrand.vue");
  return _sfc_setup$i ? _sfc_setup$i(props, ctx) : void 0;
};
const VPNavbarBrand = /* @__PURE__ */ _export_sfc(_sfc_main$i, [["ssrRender", _sfc_ssrRender$i], ["__file", "VPNavbarBrand.vue"]]);
const _sfc_main$h = /* @__PURE__ */ defineComponent({
  __name: "VPNavbarDropdown",
  props: {
    config: {}
  },
  setup(__props, { expose: __expose }) {
    __expose();
    const props = __props;
    const { config: config2 } = toRefs(props);
    const [open, toggleOpen] = useToggle();
    const dropdownAriaLabel = computed(
      () => config2.value.ariaLabel || config2.value.text
    );
    const isLastItemOfArray = (arrayItem, array) => array[array.length - 1] === arrayItem;
    const handleDropdown = (e) => {
      if (e.detail === 0) toggleOpen();
      else toggleOpen(false);
    };
    onContentUpdated(() => {
      toggleOpen(false);
    });
    const __returned__ = { props, config: config2, open, toggleOpen, dropdownAriaLabel, isLastItemOfArray, handleDropdown, VPAutoLink, get FadeInExpandTransition() {
      return FadeInExpandTransition;
    } };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$h(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({
    class: ["vp-navbar-dropdown-wrapper", { open: $setup.open }]
  }, _attrs))}><button class="vp-navbar-dropdown-title" type="button"${ssrRenderAttr("aria-label", $setup.dropdownAriaLabel)}><span class="title">${ssrInterpolate($setup.config.text)}</span><span class="arrow down"></span></button><button class="vp-navbar-dropdown-title-mobile" type="button"${ssrRenderAttr("aria-label", $setup.dropdownAriaLabel)}><span class="title">${ssrInterpolate($setup.config.text)}</span><span class="${ssrRenderClass([$setup.open ? "down" : "right", "arrow"])}"></span></button>`);
  _push(ssrRenderComponent($setup["FadeInExpandTransition"], null, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`<ul style="${ssrRenderStyle($setup.open ? null : { display: "none" })}" class="vp-navbar-dropdown"${_scopeId}><!--[-->`);
        ssrRenderList($setup.config.children, (child) => {
          _push2(`<li class="vp-navbar-dropdown-item"${_scopeId}>`);
          if ("children" in child) {
            _push2(`<!--[--><h4 class="vp-navbar-dropdown-subtitle"${_scopeId}>`);
            if (child.link) {
              _push2(ssrRenderComponent($setup["VPAutoLink"], {
                config: child,
                onFocusout: () => {
                  if ($setup.isLastItemOfArray(child, $setup.config.children) && child.children.length === 0) {
                    $setup.open = false;
                  }
                }
              }, null, _parent2, _scopeId));
            } else {
              _push2(`<span${_scopeId}>${ssrInterpolate(child.text)}</span>`);
            }
            _push2(`</h4><ul class="vp-navbar-dropdown-subitem-wrapper"${_scopeId}><!--[-->`);
            ssrRenderList(child.children, (grandchild) => {
              _push2(`<li class="vp-navbar-dropdown-subitem"${_scopeId}>`);
              _push2(ssrRenderComponent($setup["VPAutoLink"], {
                config: grandchild,
                onFocusout: () => {
                  if ($setup.isLastItemOfArray(grandchild, child.children) && $setup.isLastItemOfArray(child, $setup.config.children)) {
                    $setup.toggleOpen(false);
                  }
                }
              }, null, _parent2, _scopeId));
              _push2(`</li>`);
            });
            _push2(`<!--]--></ul><!--]-->`);
          } else {
            _push2(ssrRenderComponent($setup["VPAutoLink"], {
              config: child,
              onFocusout: () => {
                if ($setup.isLastItemOfArray(child, $setup.config.children)) {
                  $setup.toggleOpen(false);
                }
              }
            }, null, _parent2, _scopeId));
          }
          _push2(`</li>`);
        });
        _push2(`<!--]--></ul>`);
      } else {
        return [
          withDirectives(createVNode("ul", { class: "vp-navbar-dropdown" }, [
            (openBlock(true), createBlock(Fragment, null, renderList($setup.config.children, (child) => {
              return openBlock(), createBlock("li", {
                key: child.text,
                class: "vp-navbar-dropdown-item"
              }, [
                "children" in child ? (openBlock(), createBlock(Fragment, { key: 0 }, [
                  createVNode("h4", { class: "vp-navbar-dropdown-subtitle" }, [
                    child.link ? (openBlock(), createBlock($setup["VPAutoLink"], {
                      key: 0,
                      config: child,
                      onFocusout: () => {
                        if ($setup.isLastItemOfArray(child, $setup.config.children) && child.children.length === 0) {
                          $setup.open = false;
                        }
                      }
                    }, null, 8, ["config", "onFocusout"])) : (openBlock(), createBlock("span", { key: 1 }, toDisplayString(child.text), 1))
                  ]),
                  createVNode("ul", { class: "vp-navbar-dropdown-subitem-wrapper" }, [
                    (openBlock(true), createBlock(Fragment, null, renderList(child.children, (grandchild) => {
                      return openBlock(), createBlock("li", {
                        key: grandchild.link,
                        class: "vp-navbar-dropdown-subitem"
                      }, [
                        createVNode($setup["VPAutoLink"], {
                          config: grandchild,
                          onFocusout: () => {
                            if ($setup.isLastItemOfArray(grandchild, child.children) && $setup.isLastItemOfArray(child, $setup.config.children)) {
                              $setup.toggleOpen(false);
                            }
                          }
                        }, null, 8, ["config", "onFocusout"])
                      ]);
                    }), 128))
                  ])
                ], 64)) : (openBlock(), createBlock($setup["VPAutoLink"], {
                  key: 1,
                  config: child,
                  onFocusout: () => {
                    if ($setup.isLastItemOfArray(child, $setup.config.children)) {
                      $setup.toggleOpen(false);
                    }
                  }
                }, null, 8, ["config", "onFocusout"]))
              ]);
            }), 128))
          ], 512), [
            [vShow, $setup.open]
          ])
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</div>`);
}
const _sfc_setup$h = _sfc_main$h.setup;
_sfc_main$h.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPNavbarDropdown.vue");
  return _sfc_setup$h ? _sfc_setup$h(props, ctx) : void 0;
};
const VPNavbarDropdown = /* @__PURE__ */ _export_sfc(_sfc_main$h, [["ssrRender", _sfc_ssrRender$h], ["__file", "VPNavbarDropdown.vue"]]);
const resolveNavbarItem = (item, prefix = "") => {
  if (isString(item)) {
    return resolveAutoLink(resolvePrefix(prefix, item));
  }
  if ("children" in item) {
    return {
      ...item,
      children: item.children.map((child) => resolveNavbarItem(child, resolvePrefix(prefix, item.prefix)))
    };
  }
  return {
    ...item,
    link: isLinkRelative(item.link) ? resolveAutoLink(resolvePrefix(prefix, item.link)).link : item.link
  };
};
const useNavbarConfig = () => {
  const { themeLocale } = useData();
  return computed(() => (
    // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
    (themeLocale.value.navbar || []).map((item) => resolveNavbarItem(item))
  ));
};
const resolveRepoType = (repo) => {
  if (!isLinkHttp(repo) || repo.includes("github.com"))
    return "GitHub";
  if (repo.includes("bitbucket.org"))
    return "Bitbucket";
  if (repo.includes("gitlab.com"))
    return "GitLab";
  if (repo.includes("gitee.com"))
    return "Gitee";
  return null;
};
const useNavbarRepo = () => {
  const { themeLocale } = useData();
  const repo = computed(() => themeLocale.value.repo);
  const repoType = computed(() => repo.value ? resolveRepoType(repo.value) : null);
  const repoLink = computed(() => {
    if (repo.value && !isLinkHttp(repo.value)) {
      return `https://github.com/${repo.value}`;
    }
    return repo.value;
  });
  const repoLabel = computed(() => {
    if (!repoLink.value)
      return null;
    if (themeLocale.value.repoLabel)
      return themeLocale.value.repoLabel;
    if (repoType.value === null)
      return "Source";
    return repoType.value;
  });
  return computed(() => {
    if (!repoLink.value || !repoLabel.value) {
      return [];
    }
    return [
      {
        text: repoLabel.value,
        link: repoLink.value
      }
    ];
  });
};
const useNavbarSelectLanguage = () => {
  const route = useRoute();
  const routePaths = useRoutePaths();
  const { routeLocale, site, siteLocale, theme, themeLocale } = useData();
  return computed(() => {
    const localePaths = Object.keys(site.value.locales);
    if (localePaths.length < 2) {
      return [];
    }
    const currentPath = route.path;
    const currentFullPath = route.fullPath;
    const languageDropdown = {
      text: `${themeLocale.value.selectLanguageText}`,
      ariaLabel: `${themeLocale.value.selectLanguageAriaLabel ?? themeLocale.value.selectLanguageText}`,
      children: localePaths.map((targetLocalePath) => {
        var _a, _b;
        const targetSiteLocale = ((_a = site.value.locales) == null ? void 0 : _a[targetLocalePath]) ?? {};
        const targetThemeLocale = ((_b = theme.value.locales) == null ? void 0 : _b[targetLocalePath]) ?? {};
        const targetLang = `${targetSiteLocale.lang}`;
        const text = targetThemeLocale.selectLanguageName ?? targetLang;
        if (targetLang === siteLocale.value.lang) {
          return {
            text,
            activeMatch: ".",
            link: route.fullPath
          };
        }
        const targetLocalePage = currentPath.replace(routeLocale.value, targetLocalePath);
        return {
          text,
          // try to keep current hash and params across languages
          link: routePaths.value.some((item) => item === targetLocalePage) ? currentFullPath.replace(currentPath, targetLocalePage) : targetThemeLocale.home ?? targetLocalePath
        };
      })
    };
    return [languageDropdown];
  });
};
const mobile = "719px";
const cssVariables = {
  mobile
};
var DeviceType;
(function(DeviceType2) {
  DeviceType2["Mobile"] = "mobile";
})(DeviceType || (DeviceType = {}));
const DeviceTypeMap = {
  [DeviceType.Mobile]: Number.parseInt(cssVariables.mobile.replace("px", ""), 10)
};
const useUpdateDeviceStatus = (deviceType, callback) => {
  const width = DeviceTypeMap[deviceType];
  if (!Number.isInteger(width)) {
    return;
  }
  useEventListener("orientationchange", () => {
    callback(width);
  });
  useEventListener("resize", () => {
    callback(width);
  });
  onMounted(() => {
    callback(width);
  });
};
const _sfc_main$g = /* @__PURE__ */ defineComponent({
  __name: "VPNavbarItems",
  setup(__props, { expose: __expose }) {
    __expose();
    const { themeLocale } = useData();
    const navbarConfig = useNavbarConfig();
    const navbarSelectLanguage = useNavbarSelectLanguage();
    const navbarRepo = useNavbarRepo();
    const isMobile = ref(false);
    const navbarLabel = computed(() => {
      return themeLocale.value.navbarLabel ?? "site navigation";
    });
    const navbarLinks = computed(() => [
      ...navbarConfig.value,
      ...navbarSelectLanguage.value,
      ...navbarRepo.value
    ]);
    useUpdateDeviceStatus(
      DeviceType.Mobile,
      (mobileDesktopBreakpoint) => {
        isMobile.value = window.innerWidth < mobileDesktopBreakpoint;
      }
    );
    const __returned__ = { themeLocale, navbarConfig, navbarSelectLanguage, navbarRepo, isMobile, navbarLabel, navbarLinks, VPAutoLink, VPNavbarDropdown };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$g(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  if ($setup.navbarLinks.length) {
    _push(`<nav${ssrRenderAttrs(mergeProps({
      class: "vp-navbar-items",
      "aria-label": $setup.navbarLabel
    }, _attrs))}><!--[-->`);
    ssrRenderList($setup.navbarLinks, (item) => {
      _push(`<div class="vp-navbar-item">`);
      if ("children" in item) {
        _push(ssrRenderComponent($setup["VPNavbarDropdown"], {
          class: { mobile: $setup.isMobile },
          config: item
        }, null, _parent));
      } else {
        _push(ssrRenderComponent($setup["VPAutoLink"], { config: item }, null, _parent));
      }
      _push(`</div>`);
    });
    _push(`<!--]--></nav>`);
  } else {
    _push(`<!---->`);
  }
}
const _sfc_setup$g = _sfc_main$g.setup;
_sfc_main$g.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPNavbarItems.vue");
  return _sfc_setup$g ? _sfc_setup$g(props, ctx) : void 0;
};
const VPNavbarItems = /* @__PURE__ */ _export_sfc(_sfc_main$g, [["ssrRender", _sfc_ssrRender$g], ["__file", "VPNavbarItems.vue"]]);
const _sfc_main$f = {};
function _sfc_ssrRender$f(_ctx, _push, _parent, _attrs) {
  _push(`<svg${ssrRenderAttrs(mergeProps({
    class: "dark-icon",
    viewBox: "0 0 32 32"
  }, _attrs))}><path d="M13.502 5.414a15.075 15.075 0 0 0 11.594 18.194a11.113 11.113 0 0 1-7.975 3.39c-.138 0-.278.005-.418 0a11.094 11.094 0 0 1-3.2-21.584M14.98 3a1.002 1.002 0 0 0-.175.016a13.096 13.096 0 0 0 1.825 25.981c.164.006.328 0 .49 0a13.072 13.072 0 0 0 10.703-5.555a1.01 1.01 0 0 0-.783-1.565A13.08 13.08 0 0 1 15.89 4.38A1.015 1.015 0 0 0 14.98 3z" fill="currentColor"></path></svg>`);
}
const _sfc_setup$f = _sfc_main$f.setup;
_sfc_main$f.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPDarkIcon.vue");
  return _sfc_setup$f ? _sfc_setup$f(props, ctx) : void 0;
};
const VPDarkIcon = /* @__PURE__ */ _export_sfc(_sfc_main$f, [["ssrRender", _sfc_ssrRender$f], ["__file", "VPDarkIcon.vue"]]);
const _sfc_main$e = {};
function _sfc_ssrRender$e(_ctx, _push, _parent, _attrs) {
  _push(`<svg${ssrRenderAttrs(mergeProps({
    class: "light-icon",
    viewBox: "0 0 32 32"
  }, _attrs))}><path d="M16 12.005a4 4 0 1 1-4 4a4.005 4.005 0 0 1 4-4m0-2a6 6 0 1 0 6 6a6 6 0 0 0-6-6z" fill="currentColor"></path><path d="M5.394 6.813l1.414-1.415l3.506 3.506L8.9 10.318z" fill="currentColor"></path><path d="M2 15.005h5v2H2z" fill="currentColor"></path><path d="M5.394 25.197L8.9 21.691l1.414 1.415l-3.506 3.505z" fill="currentColor"></path><path d="M15 25.005h2v5h-2z" fill="currentColor"></path><path d="M21.687 23.106l1.414-1.415l3.506 3.506l-1.414 1.414z" fill="currentColor"></path><path d="M25 15.005h5v2h-5z" fill="currentColor"></path><path d="M21.687 8.904l3.506-3.506l1.414 1.415l-3.506 3.505z" fill="currentColor"></path><path d="M15 2.005h2v5h-2z" fill="currentColor"></path></svg>`);
}
const _sfc_setup$e = _sfc_main$e.setup;
_sfc_main$e.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPLightIcon.vue");
  return _sfc_setup$e ? _sfc_setup$e(props, ctx) : void 0;
};
const VPLightIcon = /* @__PURE__ */ _export_sfc(_sfc_main$e, [["ssrRender", _sfc_ssrRender$e], ["__file", "VPLightIcon.vue"]]);
const _sfc_main$d = /* @__PURE__ */ defineComponent({
  __name: "VPToggleColorModeButton",
  setup(__props, { expose: __expose }) {
    __expose();
    const { themeLocale } = useData();
    const isDarkMode = useDarkMode();
    const toggleColorMode = () => {
      isDarkMode.value = !isDarkMode.value;
    };
    const __returned__ = { themeLocale, isDarkMode, toggleColorMode, VPDarkIcon, VPLightIcon };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$d(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<button${ssrRenderAttrs(mergeProps({
    type: "button",
    class: "vp-toggle-color-mode-button",
    title: $setup.themeLocale.toggleColorMode
  }, _attrs))}>`);
  _push(ssrRenderComponent($setup["VPLightIcon"], {
    style: !$setup.isDarkMode ? null : { display: "none" }
  }, null, _parent));
  _push(ssrRenderComponent($setup["VPDarkIcon"], {
    style: $setup.isDarkMode ? null : { display: "none" }
  }, null, _parent));
  _push(`</button>`);
}
const _sfc_setup$d = _sfc_main$d.setup;
_sfc_main$d.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPToggleColorModeButton.vue");
  return _sfc_setup$d ? _sfc_setup$d(props, ctx) : void 0;
};
const VPToggleColorModeButton = /* @__PURE__ */ _export_sfc(_sfc_main$d, [["ssrRender", _sfc_ssrRender$d], ["__file", "VPToggleColorModeButton.vue"]]);
const _sfc_main$c = /* @__PURE__ */ defineComponent({
  __name: "VPToggleSidebarButton",
  emits: ["toggle"],
  setup(__props, { expose: __expose }) {
    __expose();
    const { themeLocale } = useData();
    const __returned__ = { themeLocale };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$c(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({
    class: "vp-toggle-sidebar-button",
    title: $setup.themeLocale.toggleSidebar,
    "aria-expanded": "false",
    role: "button",
    tabindex: "0"
  }, _attrs))}><div class="icon" aria-hidden="true"><span></span><span></span><span></span></div></div>`);
}
const _sfc_setup$c = _sfc_main$c.setup;
_sfc_main$c.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPToggleSidebarButton.vue");
  return _sfc_setup$c ? _sfc_setup$c(props, ctx) : void 0;
};
const VPToggleSidebarButton = /* @__PURE__ */ _export_sfc(_sfc_main$c, [["ssrRender", _sfc_ssrRender$c], ["__file", "VPToggleSidebarButton.vue"]]);
const _sfc_main$b = /* @__PURE__ */ defineComponent({
  __name: "VPNavbar",
  emits: ["toggleSidebar"],
  setup(__props, { expose: __expose }) {
    __expose();
    const SearchBox = hasGlobalComponent("SearchBox") ? resolveComponent("SearchBox") : () => null;
    const { themeLocale } = useData();
    const navbar = useTemplateRef("navbar");
    const navbarBrand = useTemplateRef("navbar-brand");
    const linksWrapperMaxWidth = ref(0);
    const linksWrapperStyle = computed(() => {
      if (!linksWrapperMaxWidth.value) {
        return {};
      }
      return {
        maxWidth: `${linksWrapperMaxWidth.value}px`
      };
    });
    const getCssValue = (el, property) => {
      var _a;
      const val = (_a = el == null ? void 0 : el.ownerDocument.defaultView) == null ? void 0 : _a.getComputedStyle(el, null)[property];
      const num = Number.parseInt(val, 10);
      return Number.isNaN(num) ? 0 : num;
    };
    useUpdateDeviceStatus(
      DeviceType.Mobile,
      (mobileDesktopBreakpoint) => {
        var _a;
        const navbarHorizontalPadding = getCssValue(navbar.value, "paddingLeft") + getCssValue(navbar.value, "paddingRight");
        if (window.innerWidth < mobileDesktopBreakpoint) {
          linksWrapperMaxWidth.value = 0;
        } else {
          linksWrapperMaxWidth.value = navbar.value.offsetWidth - navbarHorizontalPadding - (((_a = navbarBrand.value) == null ? void 0 : _a.offsetWidth) ?? 0);
        }
      }
    );
    const __returned__ = { SearchBox, themeLocale, navbar, navbarBrand, linksWrapperMaxWidth, linksWrapperStyle, getCssValue, VPNavbarBrand, VPNavbarItems, VPToggleColorModeButton, VPToggleSidebarButton };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$b(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<header${ssrRenderAttrs(mergeProps({
    ref: "navbar",
    class: "vp-navbar",
    "vp-navbar": ""
  }, _attrs))}>`);
  _push(ssrRenderComponent($setup["VPToggleSidebarButton"], {
    onToggle: ($event) => _ctx.$emit("toggleSidebar")
  }, null, _parent));
  _push(`<span>`);
  _push(ssrRenderComponent($setup["VPNavbarBrand"], null, null, _parent));
  _push(`</span><div class="vp-navbar-items-wrapper" style="${ssrRenderStyle($setup.linksWrapperStyle)}">`);
  ssrRenderSlot(_ctx.$slots, "before", {}, null, _push, _parent);
  _push(ssrRenderComponent($setup["VPNavbarItems"], { class: "vp-hide-mobile" }, null, _parent));
  ssrRenderSlot(_ctx.$slots, "after", {}, null, _push, _parent);
  if ($setup.themeLocale.colorModeSwitch) {
    _push(ssrRenderComponent($setup["VPToggleColorModeButton"], null, null, _parent));
  } else {
    _push(`<!---->`);
  }
  _push(ssrRenderComponent($setup["SearchBox"], null, null, _parent));
  _push(`</div></header>`);
}
const _sfc_setup$b = _sfc_main$b.setup;
_sfc_main$b.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPNavbar.vue");
  return _sfc_setup$b ? _sfc_setup$b(props, ctx) : void 0;
};
const VPNavbar = /* @__PURE__ */ _export_sfc(_sfc_main$b, [["ssrRender", _sfc_ssrRender$b], ["__file", "VPNavbar.vue"]]);
const _sfc_main$a = {};
function _sfc_ssrRender$a(_ctx, _push, _parent, _attrs) {
  _push(`<svg${ssrRenderAttrs(mergeProps({
    class: "edit-icon",
    viewBox: "0 0 1024 1024"
  }, _attrs))}><g fill="currentColor"><path d="M430.818 653.65a60.46 60.46 0 0 1-50.96-93.281l71.69-114.012 7.773-10.365L816.038 80.138A60.46 60.46 0 0 1 859.225 62a60.46 60.46 0 0 1 43.186 18.138l43.186 43.186a60.46 60.46 0 0 1 0 86.373L588.879 565.55l-8.637 8.637-117.466 68.234a60.46 60.46 0 0 1-31.958 11.229z"></path><path d="M728.802 962H252.891A190.883 190.883 0 0 1 62.008 771.98V296.934a190.883 190.883 0 0 1 190.883-192.61h267.754a60.46 60.46 0 0 1 0 120.92H252.891a69.962 69.962 0 0 0-69.098 69.099V771.98a69.962 69.962 0 0 0 69.098 69.098h475.911A69.962 69.962 0 0 0 797.9 771.98V503.363a60.46 60.46 0 1 1 120.922 0V771.98A190.883 190.883 0 0 1 728.802 962z"></path></g></svg>`);
}
const _sfc_setup$a = _sfc_main$a.setup;
_sfc_main$a.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPEditIcon.vue");
  return _sfc_setup$a ? _sfc_setup$a(props, ctx) : void 0;
};
const VPEditIcon = /* @__PURE__ */ _export_sfc(_sfc_main$a, [["ssrRender", _sfc_ssrRender$a], ["__file", "VPEditIcon.vue"]]);
const EDIT_LINK_PATTERNS = {
  GitHub: ":repo/edit/:branch/:path",
  GitLab: ":repo/-/edit/:branch/:path",
  Gitee: ":repo/edit/:branch/:path",
  Bitbucket: ":repo/src/:branch/:path?mode=edit&spa=0&at=:branch&fileviewer=file-view-default"
};
const resolveEditLinkPatterns = ({ docsRepo, editLinkPattern }) => {
  if (editLinkPattern) {
    return editLinkPattern;
  }
  const repoType = resolveRepoType(docsRepo);
  if (repoType !== null) {
    return EDIT_LINK_PATTERNS[repoType];
  }
  return null;
};
const resolveEditLink = ({ docsRepo, docsBranch, docsDir, filePathRelative, editLinkPattern }) => {
  if (!filePathRelative)
    return null;
  const pattern = resolveEditLinkPatterns({ docsRepo, editLinkPattern });
  if (!pattern)
    return null;
  return pattern.replace(/:repo/, isLinkHttp(docsRepo) ? docsRepo : `https://github.com/${docsRepo}`).replace(/:branch/, docsBranch).replace(/:path/, removeLeadingSlash(`${removeEndingSlash(docsDir)}/${filePathRelative}`));
};
const useEditLink = () => {
  const { frontmatter, page, themeLocale } = useData();
  return computed(() => {
    const showEditLink = frontmatter.value.editLink ?? themeLocale.value.editLink ?? true;
    if (!showEditLink) {
      return null;
    }
    const { repo, docsRepo = repo, docsBranch = "main", docsDir = "", editLinkText } = themeLocale.value;
    if (!docsRepo)
      return null;
    const editLink = resolveEditLink({
      docsRepo,
      docsBranch,
      docsDir,
      filePathRelative: page.value.filePathRelative,
      editLinkPattern: frontmatter.value.editLinkPattern ?? themeLocale.value.editLinkPattern
    });
    if (!editLink)
      return null;
    return {
      text: editLinkText ?? "Edit this page",
      link: editLink
    };
  });
};
const _sfc_main$9 = /* @__PURE__ */ defineComponent({
  __name: "VPPageMeta",
  setup(__props, { expose: __expose }) {
    __expose();
    const { frontmatter, themeLocale } = useData();
    const contributors = useContributors(
      () => frontmatter.value.contributors ?? themeLocale.value.contributors ?? true
    );
    const editLink = useEditLink();
    const lastUpdated = useLastUpdated(
      () => frontmatter.value.lastUpdated ?? themeLocale.value.lastUpdated ?? true
    );
    const __returned__ = { frontmatter, themeLocale, contributors, editLink, lastUpdated, VPAutoLink, VPEditIcon };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$9(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<footer${ssrRenderAttrs(mergeProps({ class: "vp-page-meta" }, _attrs))}>`);
  if ($setup.editLink) {
    _push(`<div class="vp-meta-item edit-link">`);
    _push(ssrRenderComponent($setup["VPAutoLink"], {
      class: "label",
      config: $setup.editLink
    }, {
      before: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          _push2(ssrRenderComponent($setup["VPEditIcon"], null, null, _parent2, _scopeId));
        } else {
          return [
            createVNode($setup["VPEditIcon"])
          ];
        }
      }),
      _: 1
    }, _parent));
    _push(`</div>`);
  } else {
    _push(`<!---->`);
  }
  _push(`<div class="vp-meta-item git-info">`);
  if ($setup.lastUpdated) {
    _push(`<div class="vp-meta-item last-updated"><span class="meta-item-label">${ssrInterpolate($setup.themeLocale.lastUpdatedText ?? $setup.lastUpdated.locale)}: </span><time class="meta-item-info"${ssrRenderAttr("datetime", $setup.lastUpdated.iso)} data-allow-mismatch>${ssrInterpolate($setup.lastUpdated.text)}</time></div>`);
  } else {
    _push(`<!---->`);
  }
  if ($setup.contributors.length) {
    _push(`<div class="vp-meta-item contributors"><span class="meta-item-label">${ssrInterpolate($setup.themeLocale.contributorsText)}: </span><span class="meta-item-info"><!--[-->`);
    ssrRenderList($setup.contributors, (contributor, index) => {
      _push(`<!--[--><span class="contributor"${ssrRenderAttr("title", `email: ${contributor.email}`)}>${ssrInterpolate(contributor.name)}</span>`);
      if (index !== $setup.contributors.length - 1) {
        _push(`<!--[-->, <!--]-->`);
      } else {
        _push(`<!---->`);
      }
      _push(`<!--]-->`);
    });
    _push(`<!--]--></span></div>`);
  } else {
    _push(`<!---->`);
  }
  _push(`</div></footer>`);
}
const _sfc_setup$9 = _sfc_main$9.setup;
_sfc_main$9.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPPageMeta.vue");
  return _sfc_setup$9 ? _sfc_setup$9(props, ctx) : void 0;
};
const VPPageMeta = /* @__PURE__ */ _export_sfc(_sfc_main$9, [["ssrRender", _sfc_ssrRender$9], ["__file", "VPPageMeta.vue"]]);
const useNavigate = () => {
  const router = useRouter();
  const route = useRoute();
  return (url) => {
    if (url)
      if (isLinkAbsolute(url)) {
        if (route.fullPath !== url)
          router.push(url);
      } else if (isLinkWithProtocol(url)) {
        window.open(url);
      } else {
        router.push(encodeURI(url));
      }
  };
};
const resolveFromFrontmatterConfig = (config2, currentPath) => {
  if (config2 === false) {
    return false;
  }
  if (isString(config2)) {
    return resolveAutoLink(config2, currentPath);
  }
  if (isPlainObject(config2)) {
    return {
      ...config2,
      link: resolveAutoLink(config2.link, currentPath).link
    };
  }
  return null;
};
const resolveFromSidebarItems = (sidebarItems, currentPath, offset) => {
  const linkIndex = sidebarItems.findIndex((item) => item.link === currentPath);
  if (linkIndex !== -1) {
    const targetItem = sidebarItems[linkIndex + offset];
    if (!targetItem)
      return null;
    if (targetItem.link)
      return targetItem;
    if ("prefix" in targetItem && !resolveRoute(targetItem.prefix).notFound)
      return {
        ...targetItem,
        link: targetItem.prefix
      };
    return null;
  }
  for (const item of sidebarItems) {
    if ("children" in item) {
      const childResult = resolveFromSidebarItems(item.children, currentPath, offset);
      if (childResult) {
        return childResult;
      }
    }
  }
  const prefixIndex = sidebarItems.findIndex((item) => "prefix" in item && item.prefix === currentPath);
  if (prefixIndex !== -1) {
    const targetItem = sidebarItems[prefixIndex + offset];
    if (!targetItem)
      return null;
    if (targetItem.link)
      return targetItem;
    if ("prefix" in targetItem && !resolveRoute(targetItem.prefix).notFound)
      return {
        ...targetItem,
        link: targetItem.prefix
      };
    return null;
  }
  return null;
};
const useRelatedLinks = () => {
  const { frontmatter, themeLocale } = useData();
  const sidebarItems = useSidebarItems();
  const routePath = useRoutePath();
  const prevLink = computed(() => {
    const prevConfig = resolveFromFrontmatterConfig(frontmatter.value.prev, routePath.value);
    return prevConfig === false ? null : prevConfig ?? (themeLocale.value.prev === false ? null : resolveFromSidebarItems(sidebarItems.value, routePath.value, -1));
  });
  const nextLink = computed(() => {
    const nextConfig = resolveFromFrontmatterConfig(frontmatter.value.next, routePath.value);
    return nextConfig === false ? null : nextConfig ?? (themeLocale.value.next === false ? null : resolveFromSidebarItems(sidebarItems.value, routePath.value, 1));
  });
  return {
    prevLink,
    nextLink
  };
};
const _sfc_main$8 = /* @__PURE__ */ defineComponent({
  __name: "VPPageNav",
  setup(__props, { expose: __expose }) {
    __expose();
    const { themeLocale } = useData();
    const navigate = useNavigate();
    const { prevLink, nextLink } = useRelatedLinks();
    const navbarLabel = computed(
      () => themeLocale.value.pageNavbarLabel ?? "page navigation"
    );
    useEventListener("keydown", (event) => {
      if (event.altKey) {
        if (event.key === "ArrowRight") {
          if (nextLink.value) {
            navigate(nextLink.value.link);
            event.preventDefault();
          }
        } else if (event.key === "ArrowLeft") {
          if (prevLink.value) {
            navigate(prevLink.value.link);
            event.preventDefault();
          }
        }
      }
    });
    const __returned__ = { themeLocale, navigate, prevLink, nextLink, navbarLabel, VPAutoLink };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$8(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  if ($setup.prevLink || $setup.nextLink) {
    _push(`<nav${ssrRenderAttrs(mergeProps({
      class: "vp-page-nav",
      "aria-label": $setup.navbarLabel
    }, _attrs))}>`);
    if ($setup.prevLink) {
      _push(ssrRenderComponent($setup["VPAutoLink"], {
        class: "prev",
        config: $setup.prevLink
      }, {
        default: withCtx((_, _push2, _parent2, _scopeId) => {
          if (_push2) {
            _push2(`<div class="hint"${_scopeId}><span class="arrow left"${_scopeId}></span> ${ssrInterpolate($setup.themeLocale.prev ?? "Prev")}</div><div class="link"${_scopeId}><span class="external-link"${_scopeId}>${ssrInterpolate($setup.prevLink.text)}</span></div>`);
          } else {
            return [
              createVNode("div", { class: "hint" }, [
                createVNode("span", { class: "arrow left" }),
                createTextVNode(" " + toDisplayString($setup.themeLocale.prev ?? "Prev"), 1)
              ]),
              createVNode("div", { class: "link" }, [
                createVNode("span", { class: "external-link" }, toDisplayString($setup.prevLink.text), 1)
              ])
            ];
          }
        }),
        _: 1
      }, _parent));
    } else {
      _push(`<!---->`);
    }
    if ($setup.nextLink) {
      _push(ssrRenderComponent($setup["VPAutoLink"], {
        class: "next",
        config: $setup.nextLink
      }, {
        default: withCtx((_, _push2, _parent2, _scopeId) => {
          if (_push2) {
            _push2(`<div class="hint"${_scopeId}>${ssrInterpolate($setup.themeLocale.next ?? "Next")} <span class="arrow right"${_scopeId}></span></div><div class="link"${_scopeId}><span class="external-link"${_scopeId}>${ssrInterpolate($setup.nextLink.text)}</span></div>`);
          } else {
            return [
              createVNode("div", { class: "hint" }, [
                createTextVNode(toDisplayString($setup.themeLocale.next ?? "Next") + " ", 1),
                createVNode("span", { class: "arrow right" })
              ]),
              createVNode("div", { class: "link" }, [
                createVNode("span", { class: "external-link" }, toDisplayString($setup.nextLink.text), 1)
              ])
            ];
          }
        }),
        _: 1
      }, _parent));
    } else {
      _push(`<!---->`);
    }
    _push(`</nav>`);
  } else {
    _push(`<!---->`);
  }
}
const _sfc_setup$8 = _sfc_main$8.setup;
_sfc_main$8.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPPageNav.vue");
  return _sfc_setup$8 ? _sfc_setup$8(props, ctx) : void 0;
};
const VPPageNav = /* @__PURE__ */ _export_sfc(_sfc_main$8, [["ssrRender", _sfc_ssrRender$8], ["__file", "VPPageNav.vue"]]);
const _sfc_main$7 = /* @__PURE__ */ defineComponent({
  __name: "VPPage",
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = { VPPageMeta, VPPageNav, get Content() {
      return Content;
    } };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$7(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<main${ssrRenderAttrs(mergeProps({ class: "vp-page" }, _attrs))}>`);
  ssrRenderSlot(_ctx.$slots, "top", {}, null, _push, _parent);
  _push(`<div vp-content>`);
  ssrRenderSlot(_ctx.$slots, "content-top", {}, null, _push, _parent);
  _push(ssrRenderComponent($setup["Content"], null, null, _parent));
  ssrRenderSlot(_ctx.$slots, "content-bottom", {}, null, _push, _parent);
  _push(`</div>`);
  _push(ssrRenderComponent($setup["VPPageMeta"], null, null, _parent));
  _push(ssrRenderComponent($setup["VPPageNav"], null, null, _parent));
  ssrRenderSlot(_ctx.$slots, "bottom", {}, null, _push, _parent);
  _push(`</main>`);
}
const _sfc_setup$7 = _sfc_main$7.setup;
_sfc_main$7.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPPage.vue");
  return _sfc_setup$7 ? _sfc_setup$7(props, ctx) : void 0;
};
const VPPage = /* @__PURE__ */ _export_sfc(_sfc_main$7, [["ssrRender", _sfc_ssrRender$7], ["__file", "VPPage.vue"]]);
const _sfc_main$6 = /* @__PURE__ */ defineComponent({
  __name: "VPDropdownTransition",
  setup(__props, { expose: __expose }) {
    __expose();
    const setHeight = (item) => {
      item.style.height = `${item.scrollHeight}px`;
    };
    const unsetHeight = (item) => {
      item.style.height = "";
    };
    const __returned__ = { setHeight, unsetHeight };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$6(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  ssrRenderSlotInner(_ctx.$slots, "default", {}, null, _push, _parent, null, true);
}
const _sfc_setup$6 = _sfc_main$6.setup;
_sfc_main$6.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPDropdownTransition.vue");
  return _sfc_setup$6 ? _sfc_setup$6(props, ctx) : void 0;
};
const VPDropdownTransition = /* @__PURE__ */ _export_sfc(_sfc_main$6, [["ssrRender", _sfc_ssrRender$6], ["__file", "VPDropdownTransition.vue"]]);
const normalizePath = (path) => decodeURI(path).replace(/#.*$/, "").replace(/(index)?\.(md|html)$/, "");
const isActiveLink = (link, route) => {
  if (route.hash === link) {
    return true;
  }
  const currentPath = normalizePath(route.path);
  const targetPath = normalizePath(link);
  return currentPath === targetPath;
};
const isActiveSidebarItem = (item, route) => {
  if (item.link && isActiveLink(item.link, route)) {
    return true;
  }
  if ("children" in item) {
    return item.children.some((child) => isActiveSidebarItem(child, route));
  }
  return false;
};
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "VPSidebarItem",
  props: {
    item: {},
    depth: { default: 0 }
  },
  setup(__props, { expose: __expose }) {
    __expose();
    const props = __props;
    const { item, depth } = toRefs(props);
    const route = useRoute();
    const router = useRouter();
    const collapsible = computed(() => item.value.collapsible);
    const isActive = computed(() => isActiveSidebarItem(item.value, route));
    const itemClass = computed(() => ({
      "vp-sidebar-item": true,
      "vp-sidebar-heading": depth.value === 0,
      "active": isActive.value,
      "collapsible": collapsible.value
    }));
    const isOpenDefault = computed(
      () => collapsible.value ? isActive.value : true
    );
    const [isOpen, toggleIsOpen] = useToggle(isOpenDefault.value);
    const onClick = (e) => {
      if (collapsible.value) {
        e.preventDefault();
        toggleIsOpen();
      }
    };
    const unregisterRouterHook = router.afterEach(() => {
      void nextTick(() => {
        isOpen.value = isOpenDefault.value;
      });
    });
    onBeforeUnmount(() => {
      unregisterRouterHook();
    });
    const __returned__ = { props, item, depth, route, router, collapsible, isActive, itemClass, isOpenDefault, isOpen, toggleIsOpen, onClick, unregisterRouterHook, VPAutoLink, VPDropdownTransition };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$5(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  const _component_VPSidebarItem = resolveComponent("VPSidebarItem", true);
  _push(`<li${ssrRenderAttrs(_attrs)}>`);
  if ($setup.item.link) {
    _push(ssrRenderComponent($setup["VPAutoLink"], {
      class: $setup.itemClass,
      config: $setup.item
    }, {
      after: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          if ($setup.collapsible) {
            _push2(`<span class="${ssrRenderClass([$setup.isOpen ? "down" : "right", "arrow"])}"${_scopeId}></span>`);
          } else {
            _push2(`<!---->`);
          }
        } else {
          return [
            $setup.collapsible ? (openBlock(), createBlock("span", {
              key: 0,
              class: ["arrow", $setup.isOpen ? "down" : "right"]
            }, null, 2)) : createCommentVNode("", true)
          ];
        }
      }),
      _: 1
    }, _parent));
  } else {
    _push(`<p tabindex="0" class="${ssrRenderClass($setup.itemClass)}">${ssrInterpolate($setup.item.text)} `);
    if ($setup.collapsible) {
      _push(`<span class="${ssrRenderClass([$setup.isOpen ? "down" : "right", "arrow"])}"></span>`);
    } else {
      _push(`<!---->`);
    }
    _push(`</p>`);
  }
  if ("children" in $setup.item && $setup.item.children.length) {
    _push(ssrRenderComponent($setup["VPDropdownTransition"], null, {
      default: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          _push2(`<ul style="${ssrRenderStyle($setup.isOpen ? null : { display: "none" })}" class="vp-sidebar-children"${_scopeId}><!--[-->`);
          ssrRenderList($setup.item.children, (child) => {
            _push2(ssrRenderComponent(_component_VPSidebarItem, {
              key: `${$setup.depth}${child.text}${child.link}`,
              item: child,
              depth: $setup.depth + 1
            }, null, _parent2, _scopeId));
          });
          _push2(`<!--]--></ul>`);
        } else {
          return [
            withDirectives(createVNode("ul", { class: "vp-sidebar-children" }, [
              (openBlock(true), createBlock(Fragment, null, renderList($setup.item.children, (child) => {
                return openBlock(), createBlock(_component_VPSidebarItem, {
                  key: `${$setup.depth}${child.text}${child.link}`,
                  item: child,
                  depth: $setup.depth + 1
                }, null, 8, ["item", "depth"]);
              }), 128))
            ], 512), [
              [vShow, $setup.isOpen]
            ])
          ];
        }
      }),
      _: 1
    }, _parent));
  } else {
    _push(`<!---->`);
  }
  _push(`</li>`);
}
const _sfc_setup$5 = _sfc_main$5.setup;
_sfc_main$5.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPSidebarItem.vue");
  return _sfc_setup$5 ? _sfc_setup$5(props, ctx) : void 0;
};
const VPSidebarItem = /* @__PURE__ */ _export_sfc(_sfc_main$5, [["ssrRender", _sfc_ssrRender$5], ["__file", "VPSidebarItem.vue"]]);
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "VPSidebarItems",
  setup(__props, { expose: __expose }) {
    __expose();
    const route = useRoute();
    const sidebarItems = useSidebarItems();
    onMounted(() => {
      watch(
        () => route.hash,
        (hash) => {
          const sidebar = document.querySelector(".vp-sidebar");
          if (!sidebar) return;
          const activeSidebarItem = document.querySelector(
            `.vp-sidebar .vp-sidebar-item.auto-link[href="${route.path}${hash}"]`
          );
          if (!activeSidebarItem) return;
          const { top: sidebarTop, height: sidebarHeight } = sidebar.getBoundingClientRect();
          const { top: activeSidebarItemTop, height: activeSidebarItemHeight } = activeSidebarItem.getBoundingClientRect();
          if (activeSidebarItemTop < sidebarTop) {
            activeSidebarItem.scrollIntoView(true);
          } else if (activeSidebarItemTop + activeSidebarItemHeight > sidebarTop + sidebarHeight) {
            activeSidebarItem.scrollIntoView(false);
          }
        }
      );
    });
    const __returned__ = { route, sidebarItems, VPSidebarItem };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$4(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  if ($setup.sidebarItems.length) {
    _push(`<ul${ssrRenderAttrs(mergeProps({ class: "vp-sidebar-items" }, _attrs))}><!--[-->`);
    ssrRenderList($setup.sidebarItems, (item) => {
      _push(ssrRenderComponent($setup["VPSidebarItem"], {
        key: `${item.text}${item.link}`,
        item
      }, null, _parent));
    });
    _push(`<!--]--></ul>`);
  } else {
    _push(`<!---->`);
  }
}
const _sfc_setup$4 = _sfc_main$4.setup;
_sfc_main$4.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPSidebarItems.vue");
  return _sfc_setup$4 ? _sfc_setup$4(props, ctx) : void 0;
};
const VPSidebarItems = /* @__PURE__ */ _export_sfc(_sfc_main$4, [["ssrRender", _sfc_ssrRender$4], ["__file", "VPSidebarItems.vue"]]);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "VPSidebar",
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = { VPNavbarItems, VPSidebarItems };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$3(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<aside${ssrRenderAttrs(mergeProps({
    class: "vp-sidebar",
    "vp-sidebar": ""
  }, _attrs))}>`);
  _push(ssrRenderComponent($setup["VPNavbarItems"], null, null, _parent));
  ssrRenderSlot(_ctx.$slots, "top", {}, null, _push, _parent);
  _push(ssrRenderComponent($setup["VPSidebarItems"], null, null, _parent));
  ssrRenderSlot(_ctx.$slots, "bottom", {}, null, _push, _parent);
  _push(`</aside>`);
}
const _sfc_setup$3 = _sfc_main$3.setup;
_sfc_main$3.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/components/VPSidebar.vue");
  return _sfc_setup$3 ? _sfc_setup$3(props, ctx) : void 0;
};
const VPSidebar = /* @__PURE__ */ _export_sfc(_sfc_main$3, [["ssrRender", _sfc_ssrRender$3], ["__file", "VPSidebar.vue"]]);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "Layout",
  setup(__props, { expose: __expose }) {
    __expose();
    const { frontmatter, page, themeLocale } = useData();
    const shouldShowNavbar = computed(
      () => frontmatter.value.navbar ?? themeLocale.value.navbar ?? true
    );
    const sidebarItems = useSidebarItems();
    const isSidebarOpen = ref(false);
    const toggleSidebar = (to) => {
      isSidebarOpen.value = typeof to === "boolean" ? to : !isSidebarOpen.value;
    };
    const touchStart = { x: 0, y: 0 };
    const onTouchStart = (e) => {
      touchStart.x = e.changedTouches[0].clientX;
      touchStart.y = e.changedTouches[0].clientY;
    };
    const onTouchEnd = (e) => {
      const dx = e.changedTouches[0].clientX - touchStart.x;
      const dy = e.changedTouches[0].clientY - touchStart.y;
      if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 40) {
        if (dx > 0 && touchStart.x <= 80) {
          toggleSidebar(true);
        } else {
          toggleSidebar(false);
        }
      }
    };
    const enableExternalLinkIcon = computed(
      () => frontmatter.value.externalLinkIcon ?? themeLocale.value.externalLinkIcon ?? true
    );
    const containerClass = computed(() => [
      {
        "no-navbar": !shouldShowNavbar.value,
        "no-sidebar": !sidebarItems.value.length,
        "sidebar-open": isSidebarOpen.value,
        "external-link-icon": enableExternalLinkIcon.value
      },
      frontmatter.value.pageClass
    ]);
    onContentUpdated(() => {
      toggleSidebar(false);
    });
    const scrollPromise2 = useScrollPromise();
    const onBeforeEnter = scrollPromise2.resolve;
    const onBeforeLeave = scrollPromise2.pending;
    const __returned__ = { frontmatter, page, themeLocale, shouldShowNavbar, sidebarItems, isSidebarOpen, toggleSidebar, touchStart, onTouchStart, onTouchEnd, enableExternalLinkIcon, containerClass, scrollPromise: scrollPromise2, onBeforeEnter, onBeforeLeave, VPFadeSlideYTransition, VPHome, VPNavbar, VPPage, VPSidebar };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$2(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({
    class: ["vp-theme-container", $setup.containerClass],
    "vp-container": ""
  }, _attrs))}>`);
  ssrRenderSlot(_ctx.$slots, "navbar", {}, () => {
    if ($setup.shouldShowNavbar) {
      _push(ssrRenderComponent($setup["VPNavbar"], { onToggleSidebar: $setup.toggleSidebar }, {
        before: withCtx((_, _push2, _parent2, _scopeId) => {
          if (_push2) {
            ssrRenderSlot(_ctx.$slots, "navbar-before", {}, null, _push2, _parent2, _scopeId);
          } else {
            return [
              renderSlot(_ctx.$slots, "navbar-before")
            ];
          }
        }),
        after: withCtx((_, _push2, _parent2, _scopeId) => {
          if (_push2) {
            ssrRenderSlot(_ctx.$slots, "navbar-after", {}, null, _push2, _parent2, _scopeId);
          } else {
            return [
              renderSlot(_ctx.$slots, "navbar-after")
            ];
          }
        }),
        _: 3
      }, _parent));
    } else {
      _push(`<!---->`);
    }
  }, _push, _parent);
  _push(`<div class="vp-sidebar-mask"></div>`);
  ssrRenderSlot(_ctx.$slots, "sidebar", {}, () => {
    _push(ssrRenderComponent($setup["VPSidebar"], null, {
      top: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          ssrRenderSlot(_ctx.$slots, "sidebar-top", {}, null, _push2, _parent2, _scopeId);
        } else {
          return [
            renderSlot(_ctx.$slots, "sidebar-top")
          ];
        }
      }),
      bottom: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          ssrRenderSlot(_ctx.$slots, "sidebar-bottom", {}, null, _push2, _parent2, _scopeId);
        } else {
          return [
            renderSlot(_ctx.$slots, "sidebar-bottom")
          ];
        }
      }),
      _: 3
    }, _parent));
  }, _push, _parent);
  ssrRenderSlot(_ctx.$slots, "page", {}, () => {
    _push(ssrRenderComponent($setup["VPFadeSlideYTransition"], {
      onBeforeEnter: $setup.onBeforeEnter,
      onBeforeLeave: $setup.onBeforeLeave
    }, {
      default: withCtx((_, _push2, _parent2, _scopeId) => {
        if (_push2) {
          if ($setup.frontmatter.home) {
            _push2(ssrRenderComponent($setup["VPHome"], null, null, _parent2, _scopeId));
          } else {
            _push2(ssrRenderComponent($setup["VPPage"], {
              key: $setup.page.path
            }, {
              top: withCtx((_2, _push3, _parent3, _scopeId2) => {
                if (_push3) {
                  ssrRenderSlot(_ctx.$slots, "page-top", {}, null, _push3, _parent3, _scopeId2);
                } else {
                  return [
                    renderSlot(_ctx.$slots, "page-top")
                  ];
                }
              }),
              "content-top": withCtx((_2, _push3, _parent3, _scopeId2) => {
                if (_push3) {
                  ssrRenderSlot(_ctx.$slots, "page-content-top", {}, null, _push3, _parent3, _scopeId2);
                } else {
                  return [
                    renderSlot(_ctx.$slots, "page-content-top")
                  ];
                }
              }),
              "content-bottom": withCtx((_2, _push3, _parent3, _scopeId2) => {
                if (_push3) {
                  ssrRenderSlot(_ctx.$slots, "page-content-bottom", {}, null, _push3, _parent3, _scopeId2);
                } else {
                  return [
                    renderSlot(_ctx.$slots, "page-content-bottom")
                  ];
                }
              }),
              bottom: withCtx((_2, _push3, _parent3, _scopeId2) => {
                if (_push3) {
                  ssrRenderSlot(_ctx.$slots, "page-bottom", {}, null, _push3, _parent3, _scopeId2);
                } else {
                  return [
                    renderSlot(_ctx.$slots, "page-bottom")
                  ];
                }
              }),
              _: 3
            }, _parent2, _scopeId));
          }
        } else {
          return [
            $setup.frontmatter.home ? (openBlock(), createBlock($setup["VPHome"], { key: 0 })) : (openBlock(), createBlock($setup["VPPage"], {
              key: $setup.page.path
            }, {
              top: withCtx(() => [
                renderSlot(_ctx.$slots, "page-top")
              ]),
              "content-top": withCtx(() => [
                renderSlot(_ctx.$slots, "page-content-top")
              ]),
              "content-bottom": withCtx(() => [
                renderSlot(_ctx.$slots, "page-content-bottom")
              ]),
              bottom: withCtx(() => [
                renderSlot(_ctx.$slots, "page-bottom")
              ]),
              _: 3
            }))
          ];
        }
      }),
      _: 3
    }, _parent));
  }, _push, _parent);
  _push(`</div>`);
}
const _sfc_setup$2 = _sfc_main$2.setup;
_sfc_main$2.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/layouts/Layout.vue");
  return _sfc_setup$2 ? _sfc_setup$2(props, ctx) : void 0;
};
const Layout = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["ssrRender", _sfc_ssrRender$2], ["__file", "Layout.vue"]]);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "NotFound",
  setup(__props, { expose: __expose }) {
    __expose();
    const { routeLocale, themeLocale } = useData();
    const messages = computed(() => themeLocale.value.notFound ?? ["Not Found"]);
    const getMsg = () => messages.value[Math.floor(Math.random() * messages.value.length)];
    const homeLink = computed(() => themeLocale.value.home ?? routeLocale.value);
    const homeText = computed(() => themeLocale.value.backToHome ?? "Back to home");
    const __returned__ = { routeLocale, themeLocale, messages, getMsg, homeLink, homeText, get RouteLink() {
      return RouteLink;
    } };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
});
function _sfc_ssrRender$1(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({
    class: "vp-theme-container",
    "vp-container": ""
  }, _attrs))} data-v-ec9917b8><main class="page" data-v-ec9917b8><div vp-content data-v-ec9917b8><h1 data-v-ec9917b8>404</h1><blockquote data-v-ec9917b8>${ssrInterpolate($setup.getMsg())}</blockquote>`);
  _push(ssrRenderComponent($setup["RouteLink"], { to: $setup.homeLink }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`${ssrInterpolate($setup.homeText)}`);
      } else {
        return [
          createTextVNode(toDisplayString($setup.homeText), 1)
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</div></main></div>`);
}
const _sfc_setup$1 = _sfc_main$1.setup;
_sfc_main$1.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("../node_modules/@vuepress/theme-default/lib/client/layouts/NotFound.vue");
  return _sfc_setup$1 ? _sfc_setup$1(props, ctx) : void 0;
};
const NotFound = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["ssrRender", _sfc_ssrRender$1], ["__scopeId", "data-v-ec9917b8"], ["__file", "NotFound.vue"]]);
const config = defineClientConfig({
  enhance({ app, router }) {
    if (!hasGlobalComponent("Badge"))
      app.component("Badge", Badge);
    const scrollBehavior = router.options.scrollBehavior;
    router.options.scrollBehavior = async (...args) => {
      await useScrollPromise().wait();
      return scrollBehavior(...args);
    };
  },
  setup() {
    setupDarkMode();
    setupHeaders();
    setupSidebarItems();
  },
  layouts: {
    Layout,
    NotFound
  }
});
const clientConfig10 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: config
}, Symbol.toStringTag, { value: "Module" }));
const clientConfig_26f3a498 = {
  enhance: ({ app }) => {
    app.component("BackToPath", defineAsyncComponent(() => import("./assets/BackToPath-DY8UutRF.js")));
    app.component("CaseStudies", defineAsyncComponent(() => import("./assets/CaseStudies-DaOuMVMr.js")));
    app.component("chart-example", defineAsyncComponent(() => import("./assets/chart-example-BLJdowBJ.js")));
    app.component("CrispDmModel", defineAsyncComponent(() => import("./assets/CrispDmModel-CtEahnox.js")));
    app.component("data-representation", defineAsyncComponent(() => import("./assets/data-representation-Da8DXfau.js")));
    app.component("did-you-know", defineAsyncComponent(() => import("./assets/did-you-know-9e9zV3sf.js")));
    app.component("DisciplineMap", defineAsyncComponent(() => import("./assets/DisciplineMap-BR3Zj6pm.js")));
    app.component("doc-resources", defineAsyncComponent(() => import("./assets/doc-resources-Q5TUu-ca.js")));
    app.component("home-features", defineAsyncComponent(() => import("./assets/home-features-B2n-u8c0.js")));
    app.component("input-form", defineAsyncComponent(() => import("./assets/input-form-BQkR9WhU.js")));
    app.component("LanguageSwitcher", defineAsyncComponent(() => import("./assets/LanguageSwitcher-Bp_4bSn7.js")));
    app.component("learning-path-visualization", defineAsyncComponent(() => import("./assets/learning-path-visualization-C-0BsQdR.js")));
    app.component("MathJaxLoader", defineAsyncComponent(() => Promise.resolve().then(() => MathJaxLoader$1)));
    app.component("ppt-resources", defineAsyncComponent(() => import("./assets/ppt-resources-D95ydLY2.js")));
    app.component("practice-notebook", defineAsyncComponent(() => import("./assets/practice-notebook-Do--o09-.js")));
    app.component("resource-cards", defineAsyncComponent(() => import("./assets/resource-cards-spnYuj4L.js")));
    app.component("TeamMembers", defineAsyncComponent(() => import("./assets/TeamMembers-DkMivPse.js")));
  }
};
const clientConfig11 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: clientConfig_26f3a498
}, Symbol.toStringTag, { value: "Module" }));
const clientConfig12 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null
}, Symbol.toStringTag, { value: "Module" }));
const _sfc_main = {
  name: "MathJaxLoader",
  mounted() {
    if (typeof window !== "undefined") {
      window.MathJax = {
        tex: {
          inlineMath: [["$", "$"]],
          displayMath: [["$$", "$$"]],
          processEscapes: true,
          processEnvironments: true
        },
        svg: {
          fontCache: "global"
        },
        options: {
          enableMenu: false
        }
      };
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
      script.async = true;
      document.head.appendChild(script);
      script.onload = () => {
        setTimeout(() => {
          if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise();
          }
        }, 100);
      };
    }
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ style: { "display": "none" } }, _attrs))}>MathJax Loader</div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/MathJaxLoader.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const MathJaxLoader = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "MathJaxLoader.vue"]]);
const MathJaxLoader$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: MathJaxLoader
}, Symbol.toStringTag, { value: "Module" }));
const client = defineClientConfig({
  enhance({ app }) {
    app.component("MathJaxLoader", MathJaxLoader);
  },
  setup() {
    if (typeof window !== "undefined") {
      window.MathJax = {
        tex: {
          inlineMath: [["$", "$"]],
          displayMath: [["$$", "$$"]],
          processEscapes: true,
          processEnvironments: true
        },
        svg: {
          fontCache: "global"
        },
        options: {
          enableMenu: false
        }
      };
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
      script.async = true;
      document.head.appendChild(script);
      script.onload = () => {
        if (window.MathJax && window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise();
        }
        const renderMathJax = () => {
          if (window.MathJax && window.MathJax.typesetPromise) {
            setTimeout(() => {
              window.MathJax.typesetPromise();
            }, 100);
          }
        };
        const observer = new MutationObserver((mutations) => {
          mutations.forEach((mutation) => {
            if (mutation.type === "childList" && mutation.addedNodes.length > 0) {
              renderMathJax();
            }
          });
        });
        observer.observe(document.body, {
          childList: true,
          subtree: true
        });
      };
    }
  }
});
const clientConfig13 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: client
}, Symbol.toStringTag, { value: "Module" }));
const clientConfigs = [
  clientConfig0,
  clientConfig1,
  clientConfig2,
  clientConfig3,
  clientConfig4,
  clientConfig5,
  clientConfig6,
  clientConfig7,
  clientConfig8,
  clientConfig9,
  clientConfig10,
  clientConfig11,
  clientConfig12,
  clientConfig13
].map((m) => m.default).filter(Boolean);
const siteData$1 = JSON.parse('{"base":"/","lang":"zh-CN","title":"数据挖掘学习指南","description":"一个帮助学生更好地理解数据挖掘课程的指南网站","head":[["link",{"rel":"icon","href":"/favicon.ico"}]],"locales":{"/":{"lang":"zh-CN","title":"数据挖掘学习指南","description":"一个帮助学生更好地理解数据挖掘课程的指南网站"},"/en/":{"lang":"en-US","title":"Data Mining Learning Guide","description":"A guide to help students better understand data mining courses"}}}');
var siteData = shallowRef(siteData$1);
var historyCreator = createMemoryHistory;
var createVueRouter = () => {
  const router = createRouter({
    // it might be an issue of vue-router that have to remove the ending slash
    history: historyCreator(removeEndingSlash("/")),
    routes: [
      {
        name: "vuepress-route",
        path: "/:catchAll(.*)",
        components: {}
      }
    ],
    scrollBehavior: (to, _from, savedPosition) => {
      if (savedPosition) return savedPosition;
      if (to.hash) return { el: to.hash };
      return { top: 0 };
    }
  });
  router.beforeResolve(async (to, from) => {
    if (to.path !== from.path || from === START_LOCATION) {
      const route = resolveRoute(to.fullPath);
      if (route.path !== to.fullPath) {
        return route.path;
      }
      const pageChunk = await route.loader();
      to.meta = {
        // attach route meta
        ...route.meta,
        // attach page chunk route meta
        _pageChunk: pageChunk
      };
    } else if (to.path === from.path) {
      to.meta = from.meta;
    }
  });
  return router;
};
var setupGlobalComponents = (app) => {
  app.component("ClientOnly", ClientOnly);
  app.component("Content", Content);
  app.component("RouteLink", RouteLink);
};
var setupGlobalComputed = (app, router, clientConfigs2) => {
  const routePath = computed(() => router.currentRoute.value.path);
  const pageChunk = customRef((track, trigger) => ({
    get() {
      track();
      return router.currentRoute.value.meta._pageChunk;
    },
    set(value) {
      router.currentRoute.value.meta._pageChunk = value;
      trigger();
    }
  }));
  const layouts = computed(() => resolvers.resolveLayouts(clientConfigs2));
  const routeLocale = computed(
    () => resolvers.resolveRouteLocale(siteData.value.locales, routePath.value)
  );
  const siteLocaleData = computed(
    () => resolvers.resolveSiteLocaleData(siteData.value, routeLocale.value)
  );
  const pageComponent = computed(() => pageChunk.value.comp);
  const pageData = computed(() => pageChunk.value.data);
  const pageFrontmatter = computed(() => pageData.value.frontmatter);
  const pageHeadTitle = computed(
    () => resolvers.resolvePageHeadTitle(pageData.value, siteLocaleData.value)
  );
  const pageHead = computed(
    () => resolvers.resolvePageHead(
      pageHeadTitle.value,
      pageFrontmatter.value,
      siteLocaleData.value
    )
  );
  const pageLang = computed(
    () => resolvers.resolvePageLang(pageData.value, siteLocaleData.value)
  );
  const pageLayout = computed(
    () => resolvers.resolvePageLayout(pageData.value, layouts.value)
  );
  const clientData = {
    layouts,
    pageData,
    pageComponent,
    pageFrontmatter,
    pageHead,
    pageHeadTitle,
    pageLang,
    pageLayout,
    redirects,
    routeLocale,
    routePath,
    routes,
    siteData,
    siteLocaleData
  };
  app.provide(clientDataSymbol, clientData);
  Object.defineProperties(app.config.globalProperties, {
    $frontmatter: { get: () => pageFrontmatter.value },
    $head: { get: () => pageHead.value },
    $headTitle: { get: () => pageHeadTitle.value },
    $lang: { get: () => pageLang.value },
    $page: { get: () => pageData.value },
    $routeLocale: { get: () => routeLocale.value },
    $site: { get: () => siteData.value },
    $siteLocale: { get: () => siteLocaleData.value },
    $withBase: { get: () => withBase }
  });
  return clientData;
};
var setupUpdateHead = () => {
  const head = usePageHead();
  const lang = usePageLang();
  {
    const ssrContext = useSSRContext();
    if (ssrContext) {
      ssrContext.head = head.value;
      ssrContext.lang = lang.value;
    }
    return;
  }
};
var appCreator = createSSRApp;
var createVueApp = async () => {
  var _a;
  const app = appCreator({
    name: "Vuepress",
    setup() {
      var _a2;
      setupUpdateHead();
      for (const clientConfig of clientConfigs) {
        (_a2 = clientConfig.setup) == null ? void 0 : _a2.call(clientConfig);
      }
      const clientRootComponents = clientConfigs.flatMap(
        ({ rootComponents = [] }) => rootComponents.map((component) => h$1(component))
      );
      const pageLayout = usePageLayout();
      return () => [h$1(pageLayout.value), clientRootComponents];
    }
  });
  const router = createVueRouter();
  setupGlobalComponents(app);
  const globalComputed = setupGlobalComputed(app, router, clientConfigs);
  {
    const { setupDevtools } = await import("./assets/setupDevtools-QXEOFQJV-DzyQ0WM_.js");
    setupDevtools(app, globalComputed);
  }
  for (const clientConfig of clientConfigs) {
    await ((_a = clientConfig.enhance) == null ? void 0 : _a.call(clientConfig, { app, router, siteData }));
  }
  app.use(router);
  return {
    app,
    router
  };
};
export {
  COMPONENT_STATE_TYPE as C,
  INSPECTOR_LABEL as I,
  PLUGIN_LABEL$1 as P,
  _export_sfc as _,
  PLUGIN_ID$1 as a,
  INSPECTOR_ID as b,
  INSPECTOR_NODES as c,
  createVueApp,
  INSPECTOR_STATE_SECTION_NAME as d
};
