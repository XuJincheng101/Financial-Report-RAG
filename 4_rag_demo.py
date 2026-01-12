import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
DB_PATH = "./chroma_db"
# 👇 必须和第3步保持一致
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def test_rag():
    print("🚀 正在初始化 RAG 系统 (中文增强版)...")

    # 1. 加载同样的 BGE 中文模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if not os.path.exists(DB_PATH):
        print("❌ 错误：数据库不存在，请重新运行 3_build_knowledge_base.py")
        return

    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    llm = Tongyi(model_name="qwen-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    你是一个金融专家。基于以下上下文回答问题。
    <context>
    {context}
    </context>
    用户问题: {input}
    """)

    # 2. 组装链
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        # 💡 这里把 k 改成 4，多找一份文档，提高准确率
        vectordb.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain
    )

    # --- 再次测试之前的那个“难题” ---
    my_question = "茅台2024年的总资产是多少？"

    print(f"❓ 提问：{my_question}")
    print("Thinking... ")

    response = retrieval_chain.invoke({"input": my_question})

    print("\n🤖 AI 回答：")
    print(response["answer"])

    print("\n📄 证据来源：")
    for doc in response["context"]:
        # 看看这次能不能出现 dianxin_2024.pdf
        print(f" - {doc.metadata.get('source_file')}")


if __name__ == "__main__":
    test_rag()