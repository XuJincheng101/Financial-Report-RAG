import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
# 👇 核心变化：导入新的链构建工具
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 配置区域 ---
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
DB_PATH = "./chroma_db"

def test_rag():
    print("🚀 正在初始化 RAG 系统 (新版 LCEL 架构)...")

    # 1. 准备嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. 加载数据库
    if not os.path.exists(DB_PATH):
        print("❌ 错误：找不到数据库文件夹！请先运行步骤 3 构建数据库。")
        return

    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # 3. 准备大模型
    llm = Tongyi(model_name="qwen-turbo", temperature=0)

    # 4. 定义“提示词模板” (这是新版架构必须的，以前是自动隐藏的)
    # 告诉 AI：你是专家，只根据 Context 回答，不要瞎编。
    prompt = ChatPromptTemplate.from_template("""
    你是一个专业的金融分析师。请仔细阅读下面的背景信息（Context），并据此回答用户的问题。
    如果背景信息中没有答案，请直接说“根据年报无法找到相关信息”，不要编造。

    <context>
    {context}
    </context>

    用户问题: {input}
    """)

    # 5. 组装两条链
    # 第一条链：把文档塞进 Prompt 里 (Stuff Documents)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 第二条链：把“检索”和“回答”连起来 (Retrieval Chain)
    retrieval_chain = create_retrieval_chain(
        vectordb.as_retriever(search_kwargs={"k": 3}),  # 每次找最相关的3段
        combine_docs_chain
    )

    print("✅ 系统就绪！开始测试...\n")

    # --- 测试环节 ---
    my_question = "中国电信2024年的总资产是多少？"  # 或者是你下载的其他公司的具体问题

    print(f"❓ 提问：{my_question}")
    print("Thinking...")

    # 运行 (注意：新版用的参数名通常是 input)
    response = retrieval_chain.invoke({"input": my_question})

    # --- 展示结果 ---
    print("\n🤖 AI 回答：")
    print(response["answer"])  # 新版返回的结果 key 叫 'answer'

    print("\n📄 参考文档来源 (Evidence)：")
    # 新版返回的来源 key 叫 'context'
    for idx, doc in enumerate(response["context"]):
        source = doc.metadata.get('source_file', '未知文件')
        content = doc.page_content[0:100].replace('\n', '')
        print(f"  [{idx + 1}] 来源：{source} | 内容摘要：{content}...")

if __name__ == "__main__":
    test_rag()