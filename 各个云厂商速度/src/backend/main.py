import os
import json
import asyncio
import threading
import time
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import FileResponse
from llm_client import UnifiedLLMClient
from queue import Queue, Empty

app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    return FileResponse(index_path)


class CompetitionConfig(BaseModel):
    questions: list = [
        "什么是人工智能？",
        "请解释机器学习与深度学习之间的区别。",
        "如何设计一个简单的神经网络来解决一个分类问题？",
        "请讨论生成对抗网络（GAN）的工作原理及其在图像生成中的应用。",
        "请分析当前AI系统在处理数据偏见与伦理问题上所面临的挑战，并提出可能的解决方案。"
        # 如需更多题目，可在此添加
    ]
    providers: list = [
        "深度求索", "阿里云", "腾讯云", "火山引擎",
        "硅基流动", "华为云", "百度云", "潞晨云", "天翼云"
    ]
    max_retries: int = 3


class ProviderStatus:
    def __init__(self):
        self.reset()

    def reset(self):
        # 每道题记录：开始时间、完成时间、首 token 时间、以及 tokens 数量
        self.start_time = [None] * 5
        self.complete_time = [None] * 5
        self.first_token = [None] * 5
        self.tokens = [{"completion": 0} for _ in range(5)]
        self.retries = [0] * 5
        self.errors = []  # 记录每题的错误信息


class CompetitionState:
    def __init__(self, config: CompetitionConfig):
        self.config = config
        self.status = {p: ProviderStatus() for p in config.providers}

    def reset_all(self):
        for p in self.config.providers:
            self.status[p].reset()


config = CompetitionConfig()
state = CompetitionState(config)
llm = UnifiedLLMClient()


@app.post("/start")
def start_competition():
    state.reset_all()
    return {"status": "ready"}


@app.get("/status")
def get_status():
    return {p: s.__dict__ for p, s in state.status.items()}


@app.get("/questions")
def get_questions():
    return config.questions


# 辅助函数：从队列中读取数据，超时返回 "TIMEOUT"
def get_chunk_from_queue(q):
    try:
        return q.get(timeout=10)
    except Empty:
        return "TIMEOUT"


@app.websocket("/ws/{provider}/{q_index}")
async def websocket_endpoint(websocket: WebSocket, provider: str, q_index: int):
    await websocket.accept()
    if provider not in config.providers or q_index >= len(config.questions):
        await websocket.send_text(json.dumps({"error": {"message": "Invalid request"}}))
        await websocket.close()
        return

    # 取当前题目内容
    question = config.questions[q_index]
    messages = [{"role": "user", "content": question}]
    p_status = state.status[provider]

    # 【关键修改】提前记录开始时间：在调用外部 API 前记录
    p_status.start_time[q_index] = time.perf_counter()

    # 调用多厂商 API 得到一个阻塞生成器
    stream = llm.stream_chat(provider, messages)
    first_token_sent = False

    # 使用队列配合后台线程异步读取生成器数据
    q = Queue()

    def reader_thread():
        for chunk in stream:
            q.put(chunk)
        q.put(None)  # 标记生成器结束

    threading.Thread(target=reader_thread, daemon=True).start()

    while True:
        chunk = await asyncio.to_thread(get_chunk_from_queue, q)
        if chunk == "TIMEOUT":
            error_message = "超时未收到输出"
            error_data = {"error": {"message": error_message}}
            await websocket.send_text(json.dumps(error_data))
            p_status.errors.append(f"Question {q_index}: {error_message}")
            break
        if chunk is None:
            break

        try:
            data = json.loads(chunk)
        except Exception as e:
            data = {"error": {"message": str(e)}}

        if "error" in data:
            p_status.errors.append(f"Question {q_index}: {data['error']['message']}")
            await websocket.send_text(chunk)
            break

        # 记录首次收到 token 的时延（单位：秒，前端可乘以 1000 展示为毫秒）
        if not first_token_sent and "choices" in data:
            delta = data["choices"][0].get("delta", {})
            if delta.get("content"):
                first_token_sent = True
                p_status.first_token[q_index] = time.perf_counter() - p_status.start_time[q_index]

        # 统计 tokens 总量，这里直接以字符数计（如需其它算法，可做调整）
        if "choices" in data:
            delta = data["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                p_status.tokens[q_index]["completion"] += len(content)

        await websocket.send_text(chunk)

        # 当收到 finish_reason 表明当前题结束
        if "choices" in data and data["choices"][0].get("finish_reason"):
            p_status.complete_time[q_index] = time.perf_counter() - p_status.start_time[q_index]
            break

    await websocket.close()


@app.get("/final_ranking")
def final_ranking():
    rankings = []
    for provider, status in state.status.items():
        # 获取所有已回答的题目索引（complete_time不为None）
        answered_q_indices = [q_index for q_index, ct in enumerate(status.complete_time) if ct is not None]

        # 只统计已回答的题目
        complete_times = [status.complete_time[q] for q in answered_q_indices]
        total_time = sum(complete_times) if complete_times else 0

        first_tokens = [status.first_token[q] for q in answered_q_indices if status.first_token[q] is not None]
        first_token = min(first_tokens) if first_tokens else None

        total_tokens = sum(status.tokens[q]["completion"] for q in answered_q_indices)
        answered_count = len(answered_q_indices)

        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        rankings.append({
            "provider": provider,
            "firstToken": first_token,
            "avgTokensPerSecond": round(avg_tokens_per_second, 2) if total_time > 0 else '-',
            "totalTime": total_time,
            "totalTokens": total_tokens,
            "totalCalls": len([t for t in status.start_time if t is not None]),
            "answeredCount": answered_count
        })

    # 按总耗时排序
    rankings.sort(key=lambda x: x["totalTime"] if x["totalTime"] else float('inf'))
    return {"rankings": rankings}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
