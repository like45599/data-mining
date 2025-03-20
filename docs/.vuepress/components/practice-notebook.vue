<template>
  <div class="notebook-container">
    <div class="notebook-card">
      <h3>{{ title }}</h3>
      <div class="notebook-buttons">
        <a :href="notebook" target="_blank" class="button notebook-button">
          <svg class="icon" viewBox="0 0 1024 1024">
            <!-- Jupyter icon path -->
            <path d="M512 0c282.784 0 512 229.216 512 512s-229.216 512-512 512S0 794.784 0 512 229.216 0 512 0z" fill="#F37626"/>
            <path d="M..." fill="#fff"/>
          </svg>
          在线运行
        </a>
        <a :href="gitee" target="_blank" class="button gitee-button">
          <svg class="icon" viewBox="0 0 1024 1024">
            <!-- Gitee icon path -->
            <path d="M512 0c282.784 0 512 229.216 512 512s-229.216 512-512 512S0 794.784 0 512 229.216 0 512 0z" fill="#C71D23"/>
            <path d="M..." fill="#fff"/>
          </svg>
          查看代码
        </a>
        <a :href="download" class="button download-button" download>
          <svg class="icon" viewBox="0 0 1024 1024">
            <!-- Download icon path -->
            <path d="M..." fill="currentColor"/>
          </svg>
          下载文件
        </a>
      </div>
      <div class="notebook-preview">
        <div class="preview-header">
          <span class="preview-title">代码预览</span>
          <span class="preview-info">建议下载后在本地运行完整版本</span>
        </div>
        <div class="preview-content">
          <slot></slot>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: String,
  notebook: String,
  gitee: String,
  download: String
})
</script>

<style scoped>
.notebook-container {
  margin: 2rem 0;
}

.notebook-card {
  border: 1px solid #eaecef;
  border-radius: 8px;
  padding: 1.5rem;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.notebook-buttons {
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
  flex-wrap: wrap;
}

.button {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  text-decoration: none;
  transition: all 0.3s;
  font-size: 0.9rem;
}

.icon {
  width: 1.2em;
  height: 1.2em;
  margin-right: 0.5rem;
}

.notebook-button {
  background: #F37626;
  color: white;
}

.notebook-button:hover {
  background: #d65d0e;
}

.gitee-button {
  background: #c71d23;
  color: white;
}

.gitee-button:hover {
  background: #a01116;
}

.download-button {
  background: #3eaf7c;
  color: white;
}

.download-button:hover {
  background: #2c8f6c;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.preview-title {
  font-weight: bold;
  color: #2c3e50;
}

.preview-info {
  font-size: 0.9rem;
  color: #666;
}

.notebook-preview {
  margin-top: 1rem;
  padding: 1rem;
  background: #f6f8fa;
  border-radius: 4px;
  border: 1px solid #eaecef;
}

.preview-content {
  background: #f6f8fa;
  padding: 1rem;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  overflow-x: auto;
}

.preview-content :deep(pre) {
  margin: 0;
  padding: 0;
  background: transparent;
}

.preview-content :deep(code) {
  display: block;
  color: #476582;
}
</style> 