"""RAG：宠物寄养知识库问答。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMBEDDING_MODEL

PERSIST_DIR = "./chroma_db"

if not os.path.exists(PERSIST_DIR):
    print("❌ 未找到向量库，请先运行：python rag/build_index.py")
    sys.exit(1)

print("📚 加载向量库...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0.3
)

RAG_PROMPT = PromptTemplate.from_template("""
你是一个专业的宠物寄养客服。请严格根据以下【参考资料】回答用户问题。
如果参考资料中没有相关信息，请直接说"抱歉，我暂时没有相关信息"，不要编造。

【参考资料】：
{context}

【用户问题】：
{question}

【回答】：
""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)


def ask(question: str):
    docs = retriever.invoke(question)
    print(f"\n🔍 检索到 {len(docs)} 个相关片段：")
    for i, d in enumerate(docs, 1):
        preview = d.page_content.replace(chr(10), " ")[:60]
        print(f"   [{i}] {preview}...")
    answer = rag_chain.invoke(question)
    print(f"\n🤖 AI 回答：\n{answer}")


if __name__ == "__main__":
    print("=" * 60)
    print("🐾 宠物寄养知识库问答")
    print("=" * 60)
    print("示例问题：")
    print("  - 寄养期间宠物生病了怎么办？")
    print("  - 中型犬寄养一天多少钱？")
    print("  - 寄养10天有优惠吗？")
    print("=" * 60)

    while True:
        try:
            q = input("\n👤 请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not q:
            continue
        if q.lower() in ("q", "quit", "exit", "退出"):
            print("👋 再见！")
            break

        ask(q)