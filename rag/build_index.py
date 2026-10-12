"""构建 RAG 向量库。首次运行会下载 BGE 模型（约100MB）。"""
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from common.config import EMBEDDING_MODEL

PERSIST_DIR = "./chroma_db"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "pet_rules.txt")


def build():
    print("📄 加载文档...")
    docs = TextLoader(DATA_FILE, encoding="utf-8").load()

    print("✂️ 切分文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = splitter.split_documents(docs)
    print(f"   共 {len(splits)} 个片段")

    print("🔢 加载 Embedding 模型（首次会下载）...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    print("💾 写入 ChromaDB...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"✅ 向量库完成：{PERSIST_DIR}")


if __name__ == "__main__":
    build()