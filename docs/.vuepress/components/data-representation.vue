<template>
  <div class="data-representation">
    <h3>数据表示方法演示</h3>
    <select v-model="selectedMethod" class="method-select">
      <option value="table">表格形式</option>
      <option value="graph">图形可视化</option>
      <option value="math">数学表达式</option>
    </select>

    <div v-if="selectedMethod === 'table'" class="table-view">
      <h4>表格形式展示</h4>
      <table>
        <thead>
          <tr>
            <th>数据点</th>
            <th>值</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(value, index) in tableData" :key="index">
            <td>{{ index + 1 }}</td>
            <td>{{ value }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="selectedMethod === 'graph'" class="graph-view">
      <chart-example></chart-example>
    </div>

    <div v-if="selectedMethod === 'math'" class="math-view">
      <h4>数学表达式展示</h4>
      <p>平均值：{{ mean }}</p>
      <p>标准差：{{ std }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const selectedMethod = ref('table')
const tableData = ref([10, 20, 30, 40, 50])
const mean = (tableData.value.reduce((a, b) => a + b, 0) / tableData.value.length).toFixed(2)
const std = Math.sqrt(
  tableData.value.map(x => (x - mean) ** 2).reduce((a, b) => a + b) / tableData.value.length
).toFixed(2)
</script>

<style scoped>
.data-representation {
  margin: 20px 0;
  padding: 20px;
  border: 1px solid #eee;
  border-radius: 8px;
}
.method-select {
  margin: 20px 0;
  padding: 8px;
  border-radius: 4px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}
th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}
th {
  background-color: #f5f5f5;
}
</style>
