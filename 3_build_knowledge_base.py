import os

# 1. 必加：国内镜像加速，防止下载模型卡死
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 配置区域 ---
DATA_FOLDER = "./data"
DB_PATH = "./chroma_db"
# 👇 核心修改：换成了国产最强轻量级中文模型 (BGE)
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def build_database():
    # 1. 扫描文件
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("❌ data 文件夹里没有 PDF！")
        return

    # 2. 读取与切割
    print(f"📦 正在处理 {len(pdf_files)} 份年报...")
    all_documents = []
    # 这里把 chunk_size 稍微调大一点，BGE 模型支持长文本更好
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

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

    # 3. 向量化（核心升级）
    print(f"\n🚀 正在下载并加载中文模型: {MODEL_NAME} ...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        # 强制使用 CPU (防止部分同学电脑没显卡报错)，如果你有显卡想加速，把下面这行删掉即可
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # BGE 模型建议开启归一化
    )

    print("💾 正在初始化数据库...")
    # 先创建一个空的数据库连接
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    # --- 核心修改：分批写入 (Batching) ---
    print(f"🔄 开始分批写入数据 (总量: {len(all_documents)})...")

    BATCH_SIZE = 5000  # 每次只写 5000 条，绝对安全 (小于 5461)

    for i in range(0, len(all_documents), BATCH_SIZE):
        batch = all_documents[i: i + BATCH_SIZE]
        print(f"   - 正在写入第 {i} 到 {i + len(batch)} 条数据...")
        vectordb.add_documents(batch)

    print(f"\n🎉 升级完成！中文知识库已重建：{DB_PATH}")


if __name__ == "__main__":
    build_database()