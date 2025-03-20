# 学习路径

<div class="path-intro">
  <div class="path-intro__content">
    <h2>你的数据挖掘学习之旅</h2>
    <p>无论你是初学者还是希望提升技能的专业人士，我们都为你准备了清晰的学习路径。按照以下路径学习，你将逐步掌握数据挖掘的核心概念和实践技能。</p>
    <div class="path-intro__features">
      <div class="path-intro__feature">
        <div class="path-intro__feature-icon">📚</div>
        <div class="path-intro__feature-text">系统化学习内容</div>
      </div>
      <div class="path-intro__feature">
        <div class="path-intro__feature-icon">🔍</div>
        <div class="path-intro__feature-text">循序渐进的难度</div>
      </div>
      <div class="path-intro__feature">
        <div class="path-intro__feature-icon">💻</div>
        <div class="path-intro__feature-text">实践项目巩固</div>
      </div>
    </div>
  </div>
</div>

## 可视化学习路径

跟随下面的学习路径，逐步掌握数据挖掘的核心知识和技能：

<learning-path-visualization></learning-path-visualization>

## 学习建议

1. **循序渐进**：按照推荐的学习顺序进行学习，每个主题都建立在前一个主题的基础上
2. **理论结合实践**：学习每个概念后，尝试通过相关的实践项目巩固所学知识
3. **定期复习**：数据挖掘涉及多个概念和技术，定期回顾已学内容有助于加深理解
4. **参与讨论**：遇到问题时，可以在社区中提问或与其他学习者交流

## 学习资源推荐

<div class="resource-grid">
  <div class="resource-card">
    <div class="resource-card__header">
      <span class="resource-card__icon">📖</span>
      <h3>初学者指南</h3>
    </div>
    <p>适合零基础学习者的入门资料和学习路径</p>
    <a href="/learning-path/beginner.html" class="resource-card__link">查看详情</a>
  </div>
  
  <div class="resource-card">
    <div class="resource-card__header">
      <span class="resource-card__icon">🚀</span>
      <h3>进阶学习</h3>
    </div>
    <p>深入学习高级算法和技术，提升数据挖掘能力</p>
    <a href="/learning-path/advanced.html" class="resource-card__link">查看详情</a>
  </div>
  
  <div class="resource-card">
    <div class="resource-card__header">
      <span class="resource-card__icon">💼</span>
      <h3>实践应用</h3>
    </div>
    <p>将数据挖掘技术应用到实际业务问题中</p>
    <a href="/learning-path/practical.html" class="resource-card__link">查看详情</a>
  </div>
</div>

<div class="did-you-know-container">
  <did-you-know category="general"></did-you-know>
</div>

<style>
.path-intro {
  margin: 2rem 0;
}

.path-intro__content {
  max-width: 800px;
  margin: 0 auto;
}

.path-intro__features {
  display: flex;
  gap: 1.5rem;
  margin-top: 1.5rem;
  justify-content: center;
}

.path-intro__feature {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background-color:rgb(224, 226, 226);
  padding: 0.8rem 1.2rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.path-intro__feature-icon {
  font-size: 1.5rem;
}

.resource-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}

.resource-card {
  background-color: var(--bg-color);
  border-radius: 8px;
  padding: 1.2rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
  border: 1px solid rgba(0, 0, 0, 0.05);
  position: relative;
  overflow: hidden;
  height: 180px;
  display: flex;
  flex-direction: column;
}

.resource-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
  border-color: rgba(24, 144, 255, 0.2);
}

.resource-card:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, var(--theme-color) 0%, var(--theme-color-light) 100%);
}

.resource-card__header {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.8rem;
}

.resource-card__icon {
  font-size: 1.5rem;
  color: var(--theme-color);
  background-color: rgba(24, 144, 255, 0.1);
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  flex-shrink: 0;
}

.resource-card__header h3 {
  /* margin: 0 0 0.5rem 0; */
  color: var(--text-color);
  font-size: 1.1rem;
  font-weight: 600;
  line-height: 1.3;
}

.resource-card p {
  color: #666;
  line-height: 1.5;
  margin: 0;
  font-size: 0.9rem;
  text-align: left;
  flex-grow: 1;
  display: flex;
  align-items: center;
}

.resource-card__link {
  display: inline-flex;
  align-items: center;
  color: var(--theme-color);
  text-decoration: none;
  font-weight: 500;
  font-size: 0.9rem;
  margin-top: auto;
  align-self: flex-start;
}

.resource-card__link:hover {
  color: var(--theme-color-light);
}

.resource-card__link:after {
  content: '→';
  margin-left: 4px;
  transition: transform 0.2s ease;
}

.resource-card__link:hover:after {
  transform: translateX(3px);
}

.did-you-know-container {
  margin: 3rem 0;
}

@media (max-width: 768px) {
  .path-intro__features {
    flex-direction: column;
    align-items: center;
  }
  
  .resource-grid {
    grid-template-columns: 1fr;
  }
  
  .resource-card {
    height: auto;
    min-height: 150px;
  }
}

@media (min-width: 769px) and (max-width: 1024px) {
  .resource-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style> 