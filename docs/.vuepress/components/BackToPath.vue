<template>
  <div class="back-to-path" v-if="showBackButton">
    <a :href="pathPrefix + '/learning-path/'" class="back-to-path__button">
      <span class="back-to-path__icon">←</span>
      {{ buttonText }}
    </a>
  </div>
</template>

<script>
export default {
  name: 'BackToPath',
  data() {
    return {
      showBackButton: false
    }
  },
  computed: {
    buttonText() {
      return this.$lang === 'en-US' ? 'Back to Learning Path' : '返回学习路径'
    },
    pathPrefix() {
      return this.$lang === 'en-US' ? '/en' : ''
    }
  },
  mounted() {
    // 检查是否从学习路径页面跳转而来
    const referrer = document.referrer;
    const currentHost = window.location.host;
    
    // 检测逻辑:同时考虑中英文路径
    if (referrer.includes(currentHost) && 
        (referrer.includes('/learning-path/') || referrer.includes('/en/learning-path/'))) {
      this.showBackButton = true;
    }

    // 检查 localStorage 中是否存在标记
    if (localStorage.getItem('fromLearningPath') === 'true') {
      this.showBackButton = true;
      localStorage.removeItem('fromLearningPath');
    }
  },
  created() {
    // 在组件创建时，为所有学习路径页面的链接添加点击事件
    this.$nextTick(() => {
      const learningPathLinks = document.querySelectorAll('a[href*="/learning-path/"]');
      learningPathLinks.forEach(link => {
        link.addEventListener('click', () => {
          sessionStorage.setItem('fromLearningPath', 'true');
        });
      });
    });
  }
}
</script>

<style lang="scss" scoped>
.back-to-path {
  margin: 2rem 0;
  text-align: center;

  &__button {
    display: inline-flex;
    align-items: center;
    padding: 0.8rem 1.5rem;
    background-color: #FFA500;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    transition: all 0.3s ease;
    font-size: 1rem;

    &:hover {
      background-color: #FFA500;
      transform: translateY(-2px);
    }
  }

  &__icon {
    margin-right: 0.5rem;
    font-size: 1.2rem;
  }
}
</style> 