---
title: 实践指南
---

# 数据挖掘实践指南

## 在线实践环境

::: tip 推荐使用
我们推荐以下在线环境进行代码实践：
1. [阿里云 DSW](https://www.aliyun.com/product/bigdata/dsw) - 数据科学工作环境
2. [百度 AI Studio](https://aistudio.baidu.com/) - 免费的深度学习平台
3. [和鲸社区](https://www.heywhale.com/) - 国内数据科学竞赛平台
:::

## 本地环境配置

::: warning 环境配置
推荐使用 Anaconda 配置本地环境：
1. 下载 [Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
2. 使用清华源加速：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
3. 创建项目环境：
```bash
conda create -n datamining python=3.8
conda activate datamining
pip install -r requirements.txt
```
:::

## 实践项目

### 1. 数据预处理实战

本实践项目将带你完成一个完整的数据预处理流程，包括：

1. 数据加载与探索
   - 使用 Pandas 读取数据
   - 查看数据基本信息
   - 数据可视化分析

2. 缺失值处理
   - 检测缺失值
   - 不同类型特征的填充策略
   - 缺失值处理效果评估

3. 异常值检测
   - 箱线图分析
   - IQR方法检测异常值
   - 异常值处理策略

4. 特征工程
   - 特征创建与转换
   - 类别特征编码
   - 数值特征标准化

5. 特征选择
   - 相关性分析
   - 特征重要性评估
   - 降维技术应用

### 项目要求

1. 环境配置
   - Python 3.8+
   - pandas, numpy, matplotlib, seaborn
   - scikit-learn

2. 数据集
   - 房价预测数据集
   - 包含数值和类别特征
   - 存在缺失值和异常值

3. 完成任务
   - 按照notebook中的步骤完成所有代码练习
   - 理解每个步骤的原理和作用
   - 尝试使用不同的参数和方法

### 实践环境

<ClientOnly>
  <practice-notebook 
    title="数据预处理实战"
    notebook="https://www.heywhale.com/mw/project/67d20eab372b07bacb11d4ea?shareby=67d20e0833a93c9ff914335e#"
    gitee="https://gitee.com/ffeng1271383559/datamining-practice/blob/master/notebooks/数据预处理实战.ipynb"
    download="/notebooks/数据预处理实战.ipynb"
  >
    
    # 数据加载与预处理
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 加载数据
    df = pd.read_csv('house_prices.csv')

    # 查看缺失值情况
    missing = df.isnull().sum()
    missing_percent = missing / len(df) * 100

    # 处理缺失值
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
  </practice-notebook>
</ClientOnly>

### 2. 分类算法实践
<ClientOnly>
  <practice-notebook 
    title="分类算法实践"
    notebook="https://aistudio.baidu.com/notebook链接"
    gitee="https://gitee.com/你的仓库/分类算法.ipynb"
    download="/notebooks/分类算法实践.ipynb"
  />
</ClientOnly> 