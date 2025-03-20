<template>
  <div class="crisp-dm-container">
    <div class="crisp-dm-center">
      <div class="crisp-dm-icon">🔄</div>
      <div class="crisp-dm-title">CRISP-DM</div>
      <div class="crisp-dm-subtitle">{{ isEnglish ? 'Cross-Industry Standard Process for Data Mining' : '跨行业标准数据挖掘流程' }}</div>
    </div>
    
    <div class="crisp-dm-phases">
      <div class="crisp-dm-phase" v-for="(phase, index) in phases" :key="index">
        <div class="phase-number">{{ index + 1 }}</div>
        <div class="phase-content">
          <div class="phase-title">{{ phase.title }}</div>
          <div class="phase-desc">{{ phase.description }}</div>
          <div class="phase-tasks">
            <span class="task-item" v-for="(task, taskIndex) in phase.tasks" :key="taskIndex">{{ task }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CrispDmModel',
  computed: {
    isEnglish() {
      return this.$lang === 'en-US';
    },
    phases() {
      if (this.isEnglish) {
        return [
          {
            title: 'Business Understanding',
            description: 'Determine business objectives, convert them into data mining problems, and develop initial plan',
            tasks: [
              'Define business objectives',
              'Assess situation',
              'Determine data mining goals',
              'Produce project plan'
            ]
          },
          {
            title: 'Data Understanding',
            description: 'Collect initial data, get familiar with data, identify data quality issues, discover insights',
            tasks: [
              'Collect initial data',
              'Describe data',
              'Explore data',
              'Verify data quality'
            ]
          },
          {
            title: 'Data Preparation',
            description: 'Select, clean, construct, integrate and format data for modeling',
            tasks: [
              'Select data',
              'Clean data',
              'Construct data',
              'Integrate data',
              'Format data'
            ]
          },
          {
            title: 'Modeling',
            description: 'Select modeling techniques, generate test design, build and assess models',
            tasks: [
              'Select modeling techniques',
              'Generate test design',
              'Build models',
              'Assess models'
            ]
          },
          {
            title: 'Evaluation',
            description: 'Evaluate results, review process, determine next steps',
            tasks: [
              'Evaluate results',
              'Review process',
              'Determine next steps'
            ]
          },
          {
            title: 'Deployment',
            description: 'Plan deployment, monitoring and maintenance, produce final report, review project',
            tasks: [
              'Plan deployment',
              'Plan monitoring',
              'Produce final report',
              'Review project'
            ]
          }
        ];
      } else {
        return [
          {
            title: '业务理解',
            description: '确定业务目标，将其转化为数据挖掘问题，并制定初步计划',
            tasks: [
              '确定业务目标',
              '评估现状',
              '确定数据挖掘目标',
              '制定项目计划'
            ]
          },
          {
            title: '数据理解',
            description: '收集初始数据，熟悉数据，识别数据质量问题，发现数据洞察',
            tasks: [
              '收集初始数据',
              '描述数据',
              '探索数据',
              '验证数据质量'
            ]
          },
          {
            title: '数据准备',
            description: '选择、清洗、构建、集成和格式化数据以供建模使用',
            tasks: [
              '选择数据',
              '清洗数据',
              '构建数据',
              '集成数据',
              '格式化数据'
            ]
          },
          {
            title: '建模',
            description: '选择建模技术，生成测试设计，构建和评估模型',
            tasks: [
              '选择建模技术',
              '生成测试设计',
              '构建模型',
              '评估模型'
            ]
          },
          {
            title: '评估',
            description: '评估结果，审查过程，确定下一步',
            tasks: [
              '评估结果',
              '审查过程',
              '确定下一步'
            ]
          },
          {
            title: '部署',
            description: '规划部署、监控和维护，生成最终报告，回顾项目',
            tasks: [
              '规划部署',
              '规划监控',
              '生成最终报告',
              '回顾项目'
            ]
          }
        ];
      }
    }
  }
}
</script>

<style scoped>
.crisp-dm-container {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 30px;
  margin: 20px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.crisp-dm-center {
  background-color: #3498db;
  color: white;
  width: 200px;
  height: 200px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin: 0 auto 40px auto;
  box-shadow: 0 4px 10px rgba(52,152,219,0.3);
}

.crisp-dm-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.crisp-dm-title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 5px;
}

.crisp-dm-subtitle {
  font-size: 14px;
  opacity: 0.9;
}

.crisp-dm-phases {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: center;
}

.crisp-dm-phase {
  background-color: white;
  border-radius: 8px;
  width: calc(50% - 20px);
  min-width: 300px;
  display: flex;
  overflow: hidden;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: transform 0.3s, box-shadow 0.3s;
}

.crisp-dm-phase:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.phase-number {
  background-color: #3498db;
  color: white;
  font-size: 24px;
  font-weight: bold;
  min-width: 50px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.phase-content {
  padding: 15px;
  flex-grow: 1;
}

.phase-title {
  font-weight: bold;
  font-size: 18px;
  margin-bottom: 8px;
  color: #2c3e50;
}

.phase-desc {
  font-size: 14px;
  color: #7f8c8d;
  margin-bottom: 12px;
  line-height: 1.4;
}

.phase-tasks {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.task-item {
  background-color: #f0f7ff;
  color: #3498db;
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 4px;
}

@media (max-width: 768px) {
  .crisp-dm-phase {
    width: 100%;
  }
}
</style> 