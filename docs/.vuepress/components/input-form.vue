<template>
  <div class="input-form">
    <h4>静态数据分析</h4>
    <div class="output-section">
      <p>数据：10, 20, 30, 40, 50</p>
      <p>平均值：{{ mean }}</p>
      <p>标准差：{{ std }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const data = [10, 20, 30, 40, 50]
const mean = (data.reduce((a, b) => a + b, 0) / data.length).toFixed(2)
const std = Math.sqrt(
  data.map(x => (x - mean) ** 2).reduce((a, b) => a + b) / data.length
).toFixed(2)
</script>

<style scoped>
.input-form {
  margin: 20px 0;
  padding: 20px;
  border: 1px solid #eee;
  border-radius: 8px;
}
.output-section {
  padding: 15px;
  background-color: #f8f8f8;
  border-radius: 4px;
}
</style>
