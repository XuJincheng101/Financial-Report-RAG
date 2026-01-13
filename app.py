import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. 基础配置 ---
st.set_page_config(page_title="金融年报智能问答", layout="wide")
st.title("🤖 金融年报 RAG 问答系统 (Pro版)")

# 加载环境变量
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 路径与模型配置 (必须与之前保持一致)
DB_PATH = "./chroma_db"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# --- 2. 核心逻辑 (带缓存，防止卡顿) ---
@st.cache_resource
def init_rag_chain():
    # A. 加载中文 Embedding 模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # B. 连接数据库
    if not os.path.exists(DB_PATH):
        return None

    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # C. 加载 LLM
    llm = Tongyi(model_name="qwen-turbo", temperature=0)

    # D. 定义提示词 (人设)
    prompt = ChatPromptTemplate.from_template("""
    你是一个专业的金融分析师。请基于以下检索到的年报片段回答用户问题。
    如果片段中没有答案，请直接说“根据现有年报无法找到相关数据”。

    <context>
    {context}
    </context>

    用户提问: {input}
    """)

    # E. 组装 RAG 链 (LCEL 架构)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vectordb.as_retriever(search_kwargs={"k": 4}),  # 检索最相关的4个片段
        combine_docs_chain
    )

    return retrieval_chain


# 初始化系统
with st.spinner("🚀 正在启动系统引擎 (加载模型与数据库)..."):
    rag_chain = init_rag_chain()

# --- 3. 侧边栏设计 ---
with st.sidebar:
    st.header("📊 系统状态面板")
    if rag_chain:
        st.success(f"✅ 知识库已挂载")
        st.info(f"🧠 模型: {MODEL_NAME}")
    else:
        st.error("❌ 数据库未找到，请先运行 build 脚本")

    st.markdown("---")
    st.markdown("### 💡 提问示例")
    st.code("中国电信2024年的经营收入是多少？")
    st.code("分析一下贵州茅台的利润分配方案")
    st.markdown("---")
    st.caption("Powered by LangChain & Streamlit")

# --- 4. 聊天窗口主逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您想咨询的年报问题..."):
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    with st.chat_message("assistant"):
        if rag_chain:
            message_placeholder = st.empty()
            with st.spinner("🔍 正在检索年报并生成答案..."):
                try:
                    # 调用链
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]

                    # 整理来源 (去重)
                    sources = set()
                    for doc in response["context"]:
                        # 只取文件名，去掉路径
                        fname = os.path.basename(doc.metadata.get('source_file', '未知文件'))
                        sources.add(fname)

                    # 拼接最终回复
                    final_text = f"{answer}\n\n---\n**📄 参考来源：** `{'`, `'.join(sources)}`"

                    message_placeholder.markdown(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})

                except Exception as e:
                    st.error(f"系统出错: {str(e)}")
        else:
            st.error("系统未就绪，请检查后台日志。")