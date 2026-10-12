"""
NL2SQL：基于 pet_care.pet 表的自然语言查询。
"""
import re
import sys
import os

# 让脚本能导入根目录的 common 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, get_mysql_uri

# ========== 1. 连接数据库 ==========
print("🔌 连接 MySQL pet_care...")
db = SQLDatabase.from_uri(get_mysql_uri())
schema_info = db.get_table_info()
print(f"📋 已读取表结构：\n{schema_info}\n")

# ========== 2. 大模型 ==========
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0
)

# ========== 3. Prompt（针对 pet 表定制）==========
PROMPT = PromptTemplate.from_template("""
你是一个资深 MySQL 专家。请根据以下表结构，把用户问题转换为一条安全的 SQL。

【表结构】：
{schema}

【业务规则】：
1. 表名是 pet，字段：id, name, type, breed, age, gender, weight, owner_name, owner_phone, health_status, description, create_time, update_time
2. type 字段取值：'狗' 或 '猫'
3. breed 是具体品种，如 '柯基', '布偶猫', '哈士奇' 等
4. health_status 取值：'健康', '需驱虫', '治疗中', '老年护理', '需接种疫苗'
5. gender 取值：'公' 或 '母'
6. create_time 是 datetime 类型
7. 只允许 SELECT 查询，禁止 DROP / DELETE / UPDATE / INSERT / ALTER
8. 只输出 SQL 本身，不要解释、不要 markdown 代码块

【用户问题】：
{question}

【SQL】：
""")
chain = PROMPT | llm | StrOutputParser()

# ========== 4. 安全校验 ==========
FORBIDDEN = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE"]


def is_safe(sql: str) -> bool:
    up = sql.upper()
    for w in FORBIDDEN:
        if re.search(rf"\b{w}\b", up):
            print(f"⚠️ 检测到危险关键字：{w}")
            return False
    return True


# ========== 5. 主流程 ==========
def nl2sql(question: str) -> str:
    raw = chain.invoke({"schema": schema_info, "question": question}).strip()
    sql = raw.replace("```sql", "").replace("```", "").strip()
    print(f"\n🤖 生成的 SQL：\n{sql}\n")

    if not is_safe(sql):
        return "⚠️ 安全拦截：SQL 含危险操作，已拒绝执行。"

    try:
        result = db.run(sql)
        return f"📊 查询结果：\n{result}"
    except Exception as e:
        return f"❌ 执行失败：{e}"


# ========== 6. 交互循环 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🐾 宠物店数据查询（NL2SQL）")
    print("=" * 60)
    print("示例问题：")
    print("  - 一共有多少只狗？")
    print("  - 有哪些猫？它们的名字和品种？")
    print("  - 年龄最大的宠物是谁？")
    print("  - 身体不健康的宠物有哪些？")
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

        print(nl2sql(q))