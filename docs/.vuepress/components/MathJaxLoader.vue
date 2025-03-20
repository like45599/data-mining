<template>
  <div style="display:none">MathJax Loader</div>
</template>

<script>
export default {
  name: 'MathJaxLoader',
  mounted() {
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
        // 等待DOM更新后渲染公式
        setTimeout(() => {
          if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise();
          }
        }, 100);
      };
    }
  }
}
</script> 