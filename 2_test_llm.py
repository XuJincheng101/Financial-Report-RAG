import os
from dotenv import load_dotenv
import dashscope

# 加载 .env 文件里的密钥
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def test_llm():
    print("正在尝试连接阿里大模型...")
    response = dashscope.Generation.call(
        model='qwen-plus',
        prompt='你好，请用一句话介绍你自己。'
    )

    if response.status_code == 200:
        print("\n连接成功！🎉")
        print(f"模型回复：{response.output.text}")
    else:
        print(f"\n连接失败 ❌ 错误码：{response.code}")
        print(f"错误信息：{response.message}")

if __name__ == "__main__":
    test_llm()