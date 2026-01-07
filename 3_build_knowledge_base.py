import os

# -----------------------------------------------------------
# 👇 核心魔法：设置国内镜像加速 (必须放在最前面！)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# -----------------------------------------------------------

from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 👇 这里改用了最新的写法，消除红字警告
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 配置区域 ---
DATA_FOLDER = "./data"
DB_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_database():
    # 1. 扫描文件
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("❌ data 文件夹里没有 PDF！")
        return

    # 2. 读取与切割
    print(f"📦 正在处理 {len(pdf_files)} 份年报，这一步已经验证过没问题...")
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for filename in tqdm(pdf_files, desc="解析进度"):
        file_path = os.path.join(DATA_FOLDER, filename)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata['source_file'] = filename
            all_documents.extend(chunks)
        except Exception as e:
            print(f"跳过坏文件: {filename}")

    print(f"\n📊 累计获得 {len(all_documents)} 个知识片段。")

    # 3. 向量化（关键！）
    print("\n🚀 正在下载模型并存入数据库（开了加速，这次会很快）...")

    # 这里的模型名字不用改，有了上面的镜像配置，它会自动走国内通道
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print(f"\n🎉 大功告成！数据库已保存在：{DB_PATH}")


if __name__ == "__main__":
    build_database()