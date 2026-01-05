import os
from langchain_community.document_loaders import PyPDFLoader

# 1. 找到你的 PDF 文件
pdf_path = "./data/maotai_2024.pdf"

print(f"正在读取文件：{pdf_path} ...")

# 2. 使用加载器读取
try:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print("读取成功！🎉")
    print(f"这份年报一共有 {len(pages)} 页。")

    print("\n--- 第一页内容展示 ---")
    print(pages[0].page_content[0:500]) # 只打印前500个字
    print("...\n---------------------")

except Exception as e:
    print(f"读取出错了：{e}")
    print("请检查：1. data文件夹里有没有这个文件？ 2. 文件名是不是写错了？")