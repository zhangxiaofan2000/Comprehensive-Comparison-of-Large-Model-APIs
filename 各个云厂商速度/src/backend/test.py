# simultaneous_requests.py
import threading
import time
from llm_client import UnifiedLLMClient

def request_provider(provider, question):
    client = UnifiedLLMClient()
    print(f"【{provider}】 请求开始...")
    try:
        # 调用流式接口
        stream = client.stream_chat(provider, [{"role": "user", "content": question}])
        for chunk in stream:
            # 每个 chunk 前面打印对应的厂商标识
            print(f"[{provider}] {chunk}")
    except Exception as e:
        print(f"[{provider}] 错误: {e}")
    print(f"【{provider}】 请求结束.")

if __name__ == "__main__":
    providers = ["deepseek", "aliyun", "tencent", "volcano", "siliconflow", "huawei", "baidu", "luchen"]
    question = "请用三句话介绍你自己"

    threads = []
    for p in providers:
        t = threading.Thread(target=request_provider, args=(p, question))
        threads.append(t)
        t.start()

    # 等待所有线程结束
    for t in threads:
        t.join()

    print("所有请求结束。")
