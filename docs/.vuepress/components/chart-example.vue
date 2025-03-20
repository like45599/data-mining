<template>
  <div>
    <h4>图形展示</h4>
    <ClientOnly>
      <div ref="chartRef" style="width: 600px; height: 400px;"></div>
    </ClientOnly>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const chartRef = ref(null)

onMounted(async () => {
  if (typeof window !== 'undefined') {
    const echarts = await import('echarts')
    const myChart = echarts.init(chartRef.value)
    const option = {
      title: { text: '静态数据折线图' },
      tooltip: { trigger: 'axis' }, // 鼠标悬停显示数据
      toolbox: { feature: { saveAsImage: {} } }, // 保存图片
      xAxis: { type: 'category', data: ['A', 'B', 'C', 'D', 'E'] },
      yAxis: { type: 'value' },
      series: [{ data: [10, 20, 30, 40, 50], type: 'line' }]
    }
    myChart.setOption(option)
  }
})
</script>