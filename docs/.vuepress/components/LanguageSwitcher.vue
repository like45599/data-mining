<template>
  <div class="language-switcher">
    <a v-if="currentLang === 'zh-CN'" href="/en/">English</a>
    <a v-else href="/">中文</a>
  </div>
</template>

<script>
export default {
  computed: {
    currentLang() {
      return this.$lang;
    }
  }
}
</script>

<style scoped>
.language-switcher {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 100;
}

.language-switcher a {
  display: inline-block;
  padding: 0.5rem 1rem;
  background-color: var(--theme-color);
  color: white;
  border-radius: 4px;
  text-decoration: none;
  font-weight: bold;
}
</style> 