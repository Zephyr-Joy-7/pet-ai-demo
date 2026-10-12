# 宠物店 AI Demo：NL2SQL + RAG

基于 LangChain + DeepSeek 的两个 LLM 应用 Demo。

## 功能

### 1. NL2SQL
用自然语言查询 MySQL `pet_care.pet` 表。

示例：
- "一共有多少只狗？" → `SELECT COUNT(*) FROM pet WHERE type='狗';`
- "身体不健康的宠物有哪些？" → `SELECT name, health_status FROM pet WHERE health_status != '健康';`

### 2. RAG
基于宠物寄养规则文档的问答，避免大模型幻觉。

## 技术栈
Python 3.10+ / LangChain / DeepSeek API / ChromaDB / BGE-small-zh / MySQL

## 快速开始

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
copy .env.example .env
# 编辑 .env 填 API Key 和 MySQL 密码
python nl2sql/main.py
python rag/build_index.py
python rag/main.py