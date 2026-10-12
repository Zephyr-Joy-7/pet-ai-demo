"""
Streamlit 网页版：NL2SQL + RAG 双 Tab（暖色调主题）。
运行：streamlit run app.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from common.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    MYSQL_DB, EMBEDDING_MODEL, get_mysql_uri,
)

# ============ 页面配置 ============
st.set_page_config(
    page_title="宠物店 AI 助手",
    page_icon="🐾",
    layout="wide",
)

# ============ 自定义 CSS（暖色调美化）============
st.markdown("""
<style>
    /* 主标题渐变 */
    h1 {
        background: linear-gradient(90deg, #FF8C42, #FFB870, #FFD9A0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }

    /* 副标题 */
    .stCaption {
        color: #A0693D !important;
    }

    /* 按钮圆角 + 阴影 */
    .stButton > button {
        border-radius: 12px;
        border: 2px solid #FF8C42;
        background: white;
        color: #FF8C42;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #FF8C42;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 140, 66, 0.3);
    }

    /* 主按钮（查询） */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF8C42, #FFB870);
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 16px rgba(255, 140, 66, 0.5);
    }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFE4CC, #FFD9A0);
    }

    /* 聊天消息卡片 */
    .stChatMessage {
        border-radius: 12px;
        padding: 8px 4px;
    }

    /* 侧边栏渐变 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF9F2 0%, #FFE4CC 100%);
    }

    /* 输入框 */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #FFD9A0;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FF8C42;
        box-shadow: 0 0 0 2px rgba(255, 140, 66, 0.2);
    }

    /* 代码块 */
    .stCodeBlock {
        border-radius: 10px;
    }

    /* 展开面板 */
    .streamlit-expanderHeader {
        border-radius: 10px;
        background: #FFF1E0;
    }
</style>
""", unsafe_allow_html=True)


# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <div style="font-size: 60px;">🐾</div>
        <h2 style="color: #FF8C42; margin: 0;">宠物店 AI 助手</h2>
        <p style="color: #A0693D; font-size: 13px;">LangChain · DeepSeek · RAG</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ 配置信息")
    st.markdown(f"**数据库**：`{MYSQL_DB}`")
    st.markdown(f"**大模型**：`{LLM_MODEL}`")
    st.markdown(f"**Embedding**：`{EMBEDDING_MODEL}`")
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("- **智能问数**：自然语言转 SQL语言进行查询数据库")
    st.markdown("- **知识库问答**：咨询宠物寄养政策")
    st.markdown("---")
    st.caption("Powered by LangChain + DeepSeek ✨")


# ============ 懒加载资源 ============
@st.cache_resource(show_spinner="🔌 连接数据库...")
def get_db():
    return SQLDatabase.from_uri(get_mysql_uri())


@st.cache_resource(show_spinner="🤖 初始化大模型...")
def get_llm(temperature=0.0):
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


@st.cache_resource(show_spinner="📚 加载向量库...")
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ============ 顶部标题 ============
st.title("🐾 宠物店 AI 智能助手")
st.caption("自然语言查数据库 · 私有知识库问答 · 由大模型驱动")

tab1, tab2 = st.tabs(["📊 智能问数（NL2SQL）", "💬 知识库问答（RAG）"])


# ============================================================
# Tab 1：NL2SQL
# ============================================================
with tab1:
    st.subheader("💡 自然语言转 SQL语言进行查询数据库")
    st.markdown("**试试这些问题：**")

    examples = [
        "一共有多少只狗？",
        "有哪些猫？它们的名字和品种？",
        "年龄最大的宠物是谁？",
        "身体不健康的宠物有哪些？",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["nl2sql_input"] = ex

    question = st.text_input(
        "🔍 请输入你的问题：",
        value=st.session_state.get("nl2sql_input", ""),
        key="nl2sql_q",
    )

    if st.button("🚀 开始查询", type="primary", key="nl2sql_btn", use_container_width=True) and question:
        with st.spinner("大模型正在生成 SQL..."):
            try:
                db = get_db()
                llm = get_llm(0.0)
                schema_info = db.get_table_info()

                PROMPT = PromptTemplate.from_template("""
你是一个资深 MySQL 专家。请根据以下表结构，把用户问题转换为一条安全的 SQL。

【表结构】：
{schema}

【业务规则】：
1. type 字段取值：'狗' 或 '猫'
2. health_status 取值：'健康', '需驱虫', '治疗中', '老年护理', '需接种疫苗'
3. gender 取值：'公' 或 '母'
4. 只允许 SELECT 查询，禁止 DROP / DELETE / UPDATE / INSERT / ALTER
5. 只输出 SQL 本身，不要解释、不要 markdown 代码块

【用户问题】：
{question}

【SQL】：
""")
                chain = PROMPT | llm | StrOutputParser()
                raw = chain.invoke({"schema": schema_info, "question": question}).strip()
                sql = raw.replace("```sql", "").replace("```", "").strip()

                st.markdown("#### 🤖 生成的 SQL")
                st.code(sql, language="sql")

                forbidden = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE"]
                is_safe = True
                for w in forbidden:
                    if re.search(rf"\b{w}\b", sql.upper()):
                        st.error(f"⚠️ 检测到危险关键字：{w}，已拒绝执行")
                        is_safe = False
                        break

                if is_safe:
                    st.markdown("#### 📊 查询结果")
                    result = db.run(sql)
                    st.text(result)
            except Exception as e:
                st.error(f"❌ 出错：{e}")


# ============================================================
# Tab 2：RAG
# ============================================================
with tab2:
    st.subheader("💬 宠物寄养知识库问答")
    st.markdown("**试试这些问题：**")

    rag_examples = [
        "寄养期间宠物生病了怎么办？",
        "中型犬寄养一天多少钱？",
        "寄养10天有优惠吗？",
    ]
    cols2 = st.columns(3)
    for i, ex in enumerate(rag_examples):
        if cols2[i].button(ex, key=f"rag_ex_{i}", use_container_width=True):
            st.session_state["rag_input"] = ex

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🐾"):
            st.markdown(msg["content"])

    # 输入
    user_input = st.chat_input("请输入你的问题...")

    # 快捷按钮触发
    if st.session_state.get("rag_input"):
        user_input = st.session_state["rag_input"]
        st.session_state["rag_input"] = ""

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🐾"):
            with st.spinner("🔍 检索知识库中..."):
                retriever = get_retriever()
                llm = get_llm(0.3)

                docs = retriever.invoke(user_input)
                context = "\n\n".join(d.page_content for d in docs)

                RAG_PROMPT = PromptTemplate.from_template("""
你是一个专业的宠物寄养客服。请严格根据以下【参考资料】回答用户问题。
如果参考资料中没有相关信息，请直接说"抱歉，我暂时没有相关信息"，不要编造。

【参考资料】：
{context}

【用户问题】：
{question}

【回答】：
""")
                chain = RAG_PROMPT | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": user_input})

                st.markdown(answer)
                with st.expander("🔍 查看检索到的原始片段"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**片段 {i}：** {d.page_content}")

        st.session_state.messages.append({"role": "assistant", "content": answer})