# 📊 Financial-Report-RAG | 金融年报智能问答系统

基于 LangChain + RAG (检索增强生成) 架构，深度解析上市公司年报，提供精准的智能问答服务。

## 🌟 项目亮点

* **海量数据处理**：自动化清洗与切分 20+ 份 PDF 年报，构建万级向量索引。
* **高精度中文检索**：集成 `BAAI/bge-small-zh-v1.5` Embedding 模型，显著提升中文语义匹配度。
* **全栈交互体验**：基于 Streamlit 开发 Pro 版 Web 界面，支持流式对话与来源溯源。

## 🛠️ 技术栈

* **LLM**: 通义千问 (Qwen-Turbo)
* **Embedding**: BAAI/bge-small-zh-v1.5
* **Vector DB**: ChromaDB
* **Framework**: LangChain 0.2 (LCEL Architecture)
* **UI**: Streamlit

## 🚀 快速开始

### 1. 环境安装
pip install -r requirements.txt

### 2. 构建知识库
将年报 PDF 放入 `data/` 目录，运行：
python 3_build_knowledge_base.py

### 3. 启动 Web 系统
streamlit run app.py