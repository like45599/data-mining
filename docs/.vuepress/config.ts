import { defineUserConfig } from 'vuepress'
import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'
import { markdownMathPlugin } from '@vuepress/plugin-markdown-math'
import { getDirname, path } from '@vuepress/utils'

const __dirname = getDirname(import.meta.url)

export default defineUserConfig({
  bundler: viteBundler(),
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]  // 假设你的图片名为 favicon.ico
  ],
  lang: 'zh-CN',
  title: '数据挖掘学习指南',
  description: '一个帮助学生更好地理解数据挖掘课程的指南网站',
  
  locales: {
    '/': {
      lang: 'zh-CN',
      title: '数据挖掘学习指南',
      description: '一个帮助学生更好地理解数据挖掘课程的指南网站',
    },
    '/en/': {
      lang: 'en-US',
      title: 'Data Mining Learning Guide',
      description: 'A guide to help students better understand data mining courses',
    },
  },
  
  plugins: [
    registerComponentsPlugin({
      componentsDir: path.resolve(__dirname, './components'),
    }),
    markdownMathPlugin({
      engine: 'mathjax',
      output: 'svg',
      tex: {
        packages: ['base', 'ams', 'noundefined', 'newcommand', 'boldsymbol'],
        inlineMath: [['$', '$']],
        displayMath: [['$$', '$$']],
        processEscapes: true,
        processEnvironments: true,
      },
      throwOnError: false,
      errorColor: '#cc0000',
      copy: true,
    }),
  ],
  
  theme: defaultTheme({
    locales: {
      '/': {
        navbar: [
          {
            text: '首页',
            link: '/',
          },
          {
            text: '数据挖掘概述',
            link: '/overview/',
          },
          {
            text: '学习路径图',
            link: '/learning-path/',
          },
          {
            text: '核心知识',
            children: [
              {
                text: '数据预处理',
                children: [
                  '/core/preprocessing/data-presentation.md',
                  '/core/preprocessing/missing-values.md',
                ],
              },
              {
                text: '分类算法',
                children: [
                  '/core/classification/svm.md',
                  '/core/classification/naive-bayes.md',
                  '/core/classification/decision-trees.md',
                ],
              },
              {
                text: '聚类分析',
                children: [
                  '/core/clustering/kmeans.md',
                  '/core/clustering/evaluation.md',
                  '/core/clustering/applications.md',
                ],
              },
              {
                text: '预测与回归',
                children: [
                  '/core/regression/linear-regression.md',
                  '/core/regression/nonlinear-regression.md',
                ],
              },
            ],
          },
          {
            text: '实践项目',
            link: '/projects/',
          },
          // {
          //   text: '学习资源',
          //   link: '/resources/',
          // },
          {
            text: '关于',
            link: '/about/',
          },
        ],
        
        sidebar: {
          '/overview/': [
            {
              text: '数据挖掘概述',
              children: [
                '/overview/README.md',
                '/overview/definition.md',
                '/overview/process.md',
                '/overview/applications.md',
                '/overview/tools.md',
              ],
            },
          ],
          '/learning-path/': [
            {
              text: '学习路径图',
              children: [
                '/learning-path/README.md',
                '/learning-path/beginner.md',
                '/learning-path/advanced.md',
                '/learning-path/practical.md',
              ],
            },
          ],
          '/core/preprocessing/': [
            {
              text: '数据预处理',
              children: [
                '/core/preprocessing/data-presentation.md',
                '/core/preprocessing/missing-values.md',
              ],
            },
          ],
          '/core/classification/': [
            {
              text: '分类算法',
              children: [
                '/core/classification/svm.md',
                '/core/classification/naive-bayes.md',
                '/core/classification/decision-trees.md',
                '/core/classification/comparison.md',
              ],
            },
          ],
          '/core/clustering/': [
            {
              text: '聚类分析',
              children: [
                '/core/clustering/kmeans.md',
                '/core/clustering/evaluation.md',
                '/core/clustering/applications.md',
              ],
            },
          ],
          '/core/regression/': [
            {
              text: '预测与回归',
              children: [
                '/core/regression/linear-regression.md',
                '/core/regression/nonlinear-regression.md',
                '/core/regression/evaluation-metrics.md',
              ],
            },
          ],
          '/projects/': [
            {
              text: '实践项目',
              children: [
                '/projects/README.md',
                '/projects/preprocessing.md',
                '/projects/classification.md',
                '/projects/clustering.md',
                '/projects/prediction.md',
              ],
            },
          ],
          '/resources/': [
            {
              text: '学习资源',
              children: [
                '/resources/README.md',
                // '/resources/tools.md',
                // '/resources/datasets.md',
                // '/resources/reading.md',
                // '/resources/faq.md',
              ],
            },
          ],
          '/core/': [
            {
              text: '数据预处理',
              children: [
                '/core/preprocessing/data-presentation.md',
                '/core/preprocessing/missing-values.md',
              ],
            },
          ],
        },
        
        logo: '/images/大顿河.png',
        logoDark: '/images/大顿河.png',
        selectLanguageName: '简体中文',
        selectLanguageText: '选择语言',
      },
      '/en/': {
        navbar: [
          {
            text: 'Home',
            link: '/en/',
          },
          {
            text: 'Overview',
            link: '/en/overview/',
          },
          {
            text: 'Learning Path',
            link: '/en/learning-path/',
          },
          {
            text: 'Core knowledge',
            children: [
              {
                text: 'Data preprocessing',
                children: [
                  '/en/core/preprocessing/data-presentation.md',
                  '/en/core/preprocessing/missing-values.md',
                ],
              },
              {
                text: 'Classification algorithm',
                children: [
                  '/en/core/classification/svm.md',
                  '/en/core/classification/naive-bayes.md',
                  '/en/core/classification/decision-trees.md',
                ],
              },
              {
                text: 'Cluster analysis',
                children: [
                  '/en/core/clustering/kmeans.md',
                  '/en/core/clustering/evaluation.md',
                  '/en/core/clustering/applications.md',
                ],
              },
              {
                text: 'Prediction and regression',
                children: [
                  '/en/core/regression/linear-regression.md',
                  '/en/core/regression/nonlinear-regression.md',
                ],
              },
            ],
          },
          {
            text: 'Practical project',
            link: '/en/projects/',
          },
          // {
          //   text: '学习资源',
          //   link: '/en/resources/',
          // },
          {
            text: 'About',
            link: '/en/about/',
          },
        ],
        
        sidebar: {
          '/en/overview/': [
            {
              text: 'Data Mining Overview',
              children: [
                '/en/overview/README.md',
                '/en/overview/definition.md',
                '/en/overview/process.md',
                '/en/overview/applications.md',
                '/en/overview/tools.md',
              ],
            },
          ],
          '/en/learning-path/': [
            {
              text: 'Learning Path',
              children: [
                '/en/learning-path/README.md',
                '/en/learning-path/beginner.md',
                '/en/learning-path/advanced.md',
                '/en/learning-path/practical.md',
              ],
            },
          ],
          '/en/core/preprocessing/': [
            {
              text: 'Data Preprocessing',
              children: [
                '/en/core/preprocessing/data-presentation.md',
                '/en/core/preprocessing/missing-values.md',
              ],
            },
          ],
          '/en/core/classification/': [
            {
              text: 'Classification Algorithms',
              children: [
                '/en/core/classification/svm.md',
                '/en/core/classification/naive-bayes.md',
                '/en/core/classification/decision-trees.md',
                '/en/core/classification/comparison.md',
              ],
            },
          ],
          '/en/core/clustering/': [
            {
              text: 'Clustering Analysis',
              children: [
                '/en/core/clustering/kmeans.md',
                '/en/core/clustering/evaluation.md',
                '/en/core/clustering/applications.md',
              ],
            },
          ],
          '/en/core/regression/': [
            {
              text: 'Prediction and Regression',
              children: [
                '/en/core/regression/linear-regression.md',
                '/en/core/regression/nonlinear-regression.md',
                '/en/core/regression/evaluation-metrics.md',
              ],
            },
          ],
          '/en/projects/': [
            {
              text: 'Projects',
              children: [
                '/en/projects/README.md',
                '/en/projects/preprocessing.md',
                '/en/projects/classification.md',
                '/en/projects/clustering.md',
                '/en/projects/prediction.md',
              ],
            },
          ],
          '/en/resources/': [
            {
              text: 'Learning Resources',
              children: [
                '/en/resources/README.md',
                // '/en/resources/tools.md',
                // '/en/resources/datasets.md',
                // '/en/resources/reading.md',
                // '/en/resources/faq.md',
              ],
            },
          ],
          '/en/core/': [
            {
              text: 'Data Preprocessing',
              children: [
                '/en/core/preprocessing/data-presentation.md',
                '/en/core/preprocessing/missing-values.md',
              ],
            },
          ],
        },
        
        logo: '/images/大顿河.png',
        logoDark: '/images/大顿河.png',
        selectLanguageName: 'English',
        selectLanguageText: 'Languages',
      },
    },
  }),
}) 