
# Data Mining Guide

欢迎使用 **Data Mining Guide**！这是一个基于 **Vue.js** 和 **VuePress** 的项目，旨在为用户提供数据挖掘的学习和实践资源。

## 项目结构

此项目包含以下主要部分：

- **Vue (端口：3000)**：用于构建前端应用。
- **VuePress (端口：3001)**：用于构建文档网站和学习指南。

## 项目目录结构

```
data-mining-guide/
├── docs/                    # VuePress 文档目录
│   ├── .vuepress/           # VuePress 配置和公共文件
│   │   ├── public/          # 公共资源
│   │   ├── styles/          # 样式
│   ├── overview/            # 概览文档
│   ├── learning-path/       # 学习路径文档
│   ├── core/                # 核心概念文档
│   ├── projects/            # 项目展示
│   └── about/               # 关于本项目
└── en/                      # 英文版本
```

## 安装与运行

### 1. 创建项目目录并初始化

```bash
mkdir data-mining-guide
cd data-mining-guide
npm init -y
git init
```

### 2. 安装依赖

安装 VuePress 及其相关依赖：

```bash
npm install -D vuepress@next
npm install -D @vuepress/bundler-vite@next @vuepress/theme-default@next
npm install @vuepress/utils@next @vuepress/plugin-register-components@next 
npm install -D sass-embedded
npm i -D @vuepress/plugin-markdown-math@next
npm i -D mathjax-full
```


### 4. 开发与构建

- **启动开发服务器：**
```bash
npm run docs:dev
```

- **打包项目：**
```bash
npm run docs:build
```


## 贡献

欢迎提出问题或贡献代码！请提交 Pull Request 或报告问题。



## 联系信息

如果您有任何问题或建议，请通过以下方式与我们联系：

- Email: 1271383559@qq.com
- GitHub：[github.com/like45599](https://github.com/like45599) 

感谢您使用 **Data Mining Guide**！
