import{_ as d,e as l,o as r,f as e,k as o,F as c,l as p,u,t as s,x as _}from"./app-CPyUwv8V.js";const k={name:"LearningPathVisualization",data(){return{activeStep:0,isEnglish:this.$lang==="en-US"}},computed:{steps(){return this.isEnglish?[{title:"Data Mining Basics",description:"Understand the fundamental concepts and processes of data mining",topics:[{title:"What is Data Mining",link:"/en/overview/definition.html"},{title:"Data Mining Process",link:"/en/overview/process.html"},{title:"Applications",link:"/en/overview/applications.html"}],skills:["Understanding data mining terminology","Identifying suitable problems for data mining","Recognizing the data mining process steps"],action:{text:"Start Learning Basics",link:"/en/overview/definition.html"}},{title:"Data Preprocessing",description:"Learn how to clean, transform and prepare data for analysis",topics:[{title:"Data Representation",link:"/en/core/preprocessing/data-presentation.html"},{title:"Missing Values",link:"/en/core/preprocessing/missing-values.html"},{title:"Feature Engineering",link:"/en/core/preprocessing/feature-engineering.html"}],skills:["Data cleaning techniques","Handling missing values","Feature selection and transformation","Data normalization and standardization"],action:{text:"Learn Data Preprocessing",link:"/en/core/preprocessing/data-presentation.html"}},{title:"Classification Algorithms",description:"Master various methods for predicting categorical outcomes",topics:[{title:"Decision Trees",link:"/en/core/classification/decision-trees.html"},{title:"Support Vector Machines",link:"/en/core/classification/svm.html"},{title:"Naive Bayes",link:"/en/core/classification/naive-bayes.html"}],skills:["Building classification models","Model evaluation and validation","Hyperparameter tuning","Ensemble methods"],action:{text:"Learn Classification Algorithms",link:"/en/core/classification/svm.html"}},{title:"Clustering Analysis",description:"Explore unsupervised learning methods to discover natural groupings",topics:[{title:"K-Means Clustering",link:"/en/core/clustering/kmeans.html"},{title:"Hierarchical Clustering",link:"/en/core/clustering/hierarchical.html"},{title:"Evaluation Methods",link:"/en/core/clustering/evaluation.html"}],skills:["Identifying appropriate clustering algorithms","Determining optimal number of clusters","Interpreting clustering results","Visualizing high-dimensional clusters"],action:{text:"Learn Clustering Analysis",link:"/en/core/clustering/kmeans.html"}},{title:"Prediction and Regression",description:"Learn techniques for predicting continuous values and time series",topics:[{title:"Linear Regression",link:"/en/core/regression/linear-regression.html"},{title:"Non-linear Models",link:"/en/core/regression/nonlinear-regression.html"},{title:"Model Evaluation",link:"/en/core/regression/evaluation-metrics.html"}],skills:["Building regression models","Feature selection for regression","Time series forecasting","Regression model evaluation"],action:{text:"Learn Prediction and Regression",link:"/en/core/regression/linear-regression.html"}},{title:"Practice Project",description:"Apply learned knowledge to real-world projects",topics:[{title:"Titanic Survival Prediction",link:"/en/projects/classification/titanic.html"},{title:"Customer Segmentation Analysis",link:"/en/projects/clustering/customer-segmentation.html"},{title:"House Price Prediction",link:"/en/projects/regression/house-price.html"}],skills:["Project Practice","Comprehensive Application","Result Interpretation"],action:{text:"Start Practice Project",link:"/en/projects/"}}]:[{title:"数据挖掘基础",description:"理解数据挖掘的基本概念和流程",topics:[{title:"什么是数据挖掘",link:"/overview/definition.html"},{title:"数据挖掘流程",link:"/overview/process.html"},{title:"应用场景",link:"/overview/applications.html"}],skills:["理解数据挖掘术语","识别适合数据挖掘的问题","认识数据挖掘流程步骤"],action:{text:"开始学习基础知识",link:"/overview/definition.html"}},{title:"数据预处理",description:"学习如何清洗、转换和准备数据以进行分析",topics:[{title:"数据表示",link:"/core/preprocessing/data-presentation.html"},{title:"缺失值处理",link:"/core/preprocessing/missing-values.html"},{title:"特征工程",link:"/core/preprocessing/feature-engineering.html"}],skills:["数据清洗技术","处理缺失值","特征选择与转换","数据归一化和标准化"],action:{text:"学习数据预处理",link:"/core/preprocessing/data-presentation.html"}},{title:"分类算法",description:"掌握预测分类结果的各种方法",topics:[{title:"决策树",link:"/core/classification/decision-trees.html"},{title:"支持向量机",link:"/core/classification/svm.html"},{title:"朴素贝叶斯",link:"/core/classification/naive-bayes.html"}],skills:["构建分类模型","模型评估和验证","超参数调优","集成方法"],action:{text:"学习分类算法",link:"/core/classification/svm.html"}},{title:"聚类分析",description:"探索无监督学习方法以发现自然分组",topics:[{title:"K-均值聚类",link:"/core/clustering/kmeans.html"},{title:"层次聚类",link:"/core/clustering/hierarchical.html"},{title:"评估方法",link:"/core/clustering/evaluation.html"}],skills:["识别适当的聚类算法","确定最佳聚类数量","解释聚类结果","可视化高维聚类"],action:{text:"学习聚类分析",link:"/core/clustering/kmeans.html"}},{title:"预测与回归",description:"学习预测连续值和时间序列的技术",topics:[{title:"线性回归",link:"/core/regression/linear-regression.html"},{title:"非线性模型",link:"/core/regression/nonlinear-regression.html"},{title:"模型评估",link:"/core/regression/evaluation-metrics.html"}],skills:["构建回归模型","回归的特征选择","时间序列预测","回归模型评估"],action:{text:"学习预测与回归",link:"/core/regression/linear-regression.html"}},{title:"实践项目",description:"通过实际项目巩固所学知识，提升实战能力",topics:[{title:"泰坦尼克号生存预测",link:"/projects/classification/titanic.html"},{title:"客户分群分析",link:"/projects/clustering/customer-segmentation.html"},{title:"房价预测",link:"/projects/regression/house-price.html"}],skills:["项目实战","综合应用","结果解释"],action:{text:"开始实践项目",link:"/projects/"}}]}},methods:{setActiveStep(g){this.activeStep=g},setFromLearningPath(){localStorage.setItem("fromLearningPath","true")}}},v={class:"learning-path"},f={class:"learning-path__timeline"},S=["onClick"],y={key:0,class:"learning-path__step-connector"},P={class:"learning-path__step-icon"},C={class:"learning-path__step-number"},x={class:"learning-path__step-content"},L={class:"learning-path__step-title"},j={class:"learning-path__step-description"},w={key:0,class:"learning-path__details"},D={class:"learning-path__details-header"},E={class:"learning-path__details-body"},M={class:"learning-path__topics"},b=["href"],A={class:"learning-path__skills"},z={class:"learning-path__skill-tags"},B={key:0,class:"learning-path__action"},F=["href"],R={class:"learning-path__progress"},V={class:"learning-path__progress-bar"},I={class:"learning-path__progress-text"};function H(g,h,N,T,i,t){return r(),l("div",v,[e("div",f,[(r(!0),l(c,null,p(t.steps,(a,n)=>(r(),l("div",{key:n,class:u(["learning-path__step",{active:i.activeStep===n,completed:n<i.activeStep}]),onClick:m=>t.setActiveStep(n)},[n>0?(r(),l("div",y)):o("",!0),e("div",P,[e("span",C,s(n+1),1)]),e("div",x,[e("h3",L,s(a.title),1),e("p",j,s(a.description),1)])],10,S))),128))]),t.steps[i.activeStep]?(r(),l("div",w,[e("div",D,[e("h3",null,s(t.steps[i.activeStep].title)+" - "+s(i.isEnglish?"Details":"详细内容"),1)]),e("div",E,[e("div",M,[e("h4",null,s(i.isEnglish?"Core Topics":"核心主题"),1),e("ul",null,[(r(!0),l(c,null,p(t.steps[i.activeStep].topics,(a,n)=>(r(),l("li",{key:n},[e("a",{href:a.link,onClick:h[0]||(h[0]=(...m)=>t.setFromLearningPath&&t.setFromLearningPath(...m))},s(a.title),9,b)]))),128))])]),e("div",A,[e("h4",null,s(i.isEnglish?"Key Skills":"关键技能"),1),e("div",z,[(r(!0),l(c,null,p(t.steps[i.activeStep].skills,(a,n)=>(r(),l("span",{key:n,class:"learning-path__skill-tag"},s(a),1))),128))])])]),t.steps[i.activeStep].action?(r(),l("div",B,[e("a",{href:t.steps[i.activeStep].action.link,class:"learning-path__action-button"},s(t.steps[i.activeStep].action.text),9,F)])):o("",!0)])):o("",!0),e("div",R,[e("div",V,[e("div",{class:"learning-path__progress-fill",style:_({width:`${i.activeStep/(t.steps.length-1)*100}%`})},null,4)]),e("div",I,s(i.isEnglish?"Completion":"完成度")+": "+s(Math.round(i.activeStep/(t.steps.length-1)*100))+"% ",1)])])}const U=d(k,[["render",H],["__scopeId","data-v-0df1639d"],["__file","learning-path-visualization.vue"]]);export{U as default};
