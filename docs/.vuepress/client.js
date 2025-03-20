import { defineClientConfig } from '@vuepress/client'
import MathJaxLoader from './components/MathJaxLoader.vue'

export default defineClientConfig({
  enhance({ app }) {
    app.component('MathJaxLoader', MathJaxLoader)
  },
  setup() {
    // 全局应用MathJax
    if (typeof window !== 'undefined') {
      // 添加MathJax配置
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$']],
          displayMath: [['$$', '$$']],
          processEscapes: true,
          processEnvironments: true,
        },
        svg: {
          fontCache: 'global'
        },
        options: {
          enableMenu: false
        }
      };
      
      // 加载MathJax脚本
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
      script.async = true;
      document.head.appendChild(script);
      
      // 添加重新渲染方法
      script.onload = () => {
        // 初始渲染
        if (window.MathJax && window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise();
        }
        
        // 监听路由变化，在页面切换后重新渲染公式
        const renderMathJax = () => {
          if (window.MathJax && window.MathJax.typesetPromise) {
            setTimeout(() => {
              window.MathJax.typesetPromise();
            }, 100);
          }
        };
        
        // 监听路由变化
        const observer = new MutationObserver((mutations) => {
          mutations.forEach((mutation) => {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
              renderMathJax();
            }
          });
        });
        
        // 观察DOM变化
        observer.observe(document.body, {
          childList: true,
          subtree: true
        });
      };
    }
  }
})