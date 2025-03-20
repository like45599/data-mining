# Learning Path

<div class="path-intro">
  <div class="path-intro__content">
    <h2>Your Data Mining Learning Journey</h2>
    <p>Whether you are a beginner or a professional looking to enhance your skills, we have prepared a clear learning path for you. By following this path, you will gradually master the core concepts and practical skills of data mining.</p>
    <div class="path-intro__features">
      <div class="path-intro__feature">
        <div class="path-intro__feature-icon">📚</div>
        <div class="path-intro__feature-text">Systematic Learning Content</div>
      </div>
      <div class="path-intro__feature">
        <div class="path-intro__feature-icon">🔍</div>
        <div class="path-intro__feature-text">Gradual Difficulty Progression</div>
      </div>
      <div class="path-intro__feature">
        <div class="path-intro__feature-icon">💻</div>
        <div class="path-intro__feature-text">Hands-on Projects for Reinforcement</div>
      </div>
    </div>
  </div>
</div>

## Visual Learning Path

Follow the learning path below to gradually master the core knowledge and skills of data mining:

<learning-path-visualization></learning-path-visualization>

## Learning Recommendations

1. **Gradual Progression**: Follow the recommended learning order, where each topic builds on the previous one.
2. **Theory Combined with Practice**: After learning each concept, try to reinforce your knowledge by working on related practical projects.
3. **Regular Review**: Data mining involves multiple concepts and techniques. Regularly reviewing what you have learned will help deepen your understanding.
4. **Participate in Discussions**: If you encounter issues, you can ask questions in the community or interact with other learners.

## Recommended Learning Resources

<div class="resource-grid">
  <div class="resource-card">
    <div class="resource-card__header">
      <span class="resource-card__icon">📖</span>
      <h3>Beginner's Guide</h3>
    </div>
    <p>Introductory materials and learning paths for absolute beginners</p>
    <a href="/learning-path/beginner.html" class="resource-card__link">Learn More</a>
  </div>
  
  <div class="resource-card">
    <div class="resource-card__header">
      <span class="resource-card__icon">🚀</span>
      <h3>Advanced Learning</h3>
    </div>
    <p>Deep dive into advanced algorithms and techniques to enhance your data mining capabilities</p>
    <a href="/learning-path/advanced.html" class="resource-card__link">Learn More</a>
  </div>
  
  <div class="resource-card">
    <div class="resource-card__header">
      <span class="resource-card__icon">💼</span>
      <h3>Practical Applications</h3>
    </div>
    <p>Apply data mining techniques to real-world business problems</p>
    <a href="/learning-path/practical.html" class="resource-card__link">Learn More</a>
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