import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class UnifiedLLMClient:
    def __init__(self):
        """统一的多厂商 LLM 客户端。"""
        # 默认参数，天翼云也使用这些参数（可根据需要调整）
        self.default_params = {
            "temperature": 0.7,
        }
        self.provider_config = {
            "深度求索": {
                "client": self._init_openai_client,
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "model": "deepseek-reasoner",
                "default_params": {}
            },
            "阿里云": {
                "client": self._init_openai_client,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key_env": "ALIYUN_API_KEY",
                "model": "deepseek-r1",
                "default_params": {}
            },
            "腾讯云": {
                "client": self._init_openai_client,
                "base_url": "https://api.lkeap.cloud.tencent.com/v1",
                "api_key_env": "TENCENT_API_KEY",
                "model": "deepseek-r1",
                "default_params": {}
            },
            "火山引擎": {
                "client": self._init_openai_client,
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "api_key_env": "VOLC_API_KEY",
                "model": os.getenv("VOLC_ENDPOINT_ID"),
                "default_params": {}
            },
            "硅基流动": {
                "client": self._init_openai_client,
                "base_url": "https://api.siliconflow.cn/v1",
                "api_key_env": "SILICONFLOW_API_KEY",
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "default_params": {}
            },
            "百度云": {
                "client": self._init_openai_client,
                "base_url": "https://qianfan.baidubce.com/v2",
                "api_key_env": "BAIDU_API_KEY",
                "model": "deepseek-r1",
                "default_params": {}
            },
            "华为云": {
                "api_key_env": "HUAWEI_API_KEY",
                "url": "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/c3cfa9e2-40c9-485f-a747-caae405296ef/v1/chat/completions",
                "model": "DeepSeek-R1",
                "default_params": {}
            },
            "潞晨云": {
                "api_key_env": "LUCHEN_API_KEY",
                "url": "https://cloud.luchentech.com/api/maas/chat/completions",
                "model": "VIP/deepseek-ai/DeepSeek-R1",
                "default_params": {}
            },
            # 新增：天翼云（DeepSeek-R1-昇腾版 / DeepSeekPyTorch）
            "天翼云": {
                "api_key_env": "TIANYI_API_KEY",  # 请确保环境变量中配置了正确的 App Key
                "url": "https://wishub-x1.ctyun.cn/v1/chat/completions",
                "model": "4bd107bff85941239e27b1509eccfe98",
                "default_params": {}
            }
        }

    def _init_openai_client(self, config):
        return OpenAI(
            api_key=os.getenv(config["api_key_env"]),
            base_url=config["base_url"],
            timeout=30
        )

    def _handle_streaming_response_openai(self, response, model):
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = str(getattr(delta, "content", "") or "")
            reasoning = str(getattr(delta, "reasoning_content", "") or "")
            full_text = f"{reasoning}{content}"
            json_obj = {
                "id": "chatcmpl-simulated",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "delta": {"content": full_text},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield json.dumps(json_obj)
        finish_obj = {
            "id": "chatcmpl-simulated",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        yield json.dumps(finish_obj)

    def _handle_sse_streaming_response(self, provider, response, model):
        if response.status_code != 200:
            error_dict = {"error": {"message": f"请求失败({response.status_code}): {response.text}"}}
            yield json.dumps(error_dict)
            return
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                chunk = line[6:].strip().decode("utf-8")
                if chunk == "[DONE]":
                    finish_obj = {
                        "id": "chatcmpl-simulated",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield json.dumps(finish_obj)
                    break
                try:
                    json_data = json.loads(chunk)
                    result = {
                        "id": "chatcmpl-simulated",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    delta = {}
                    for choice in json_data.get("choices", []):
                        delta.update(choice.get("delta", {}))
                    if delta:
                        result["choices"][0]["delta"] = delta
                    yield json.dumps(result)
                except json.JSONDecodeError:
                    continue

    def _handle_tianyi_streaming_response(self, provider, response, model):
        if response.status_code != 200:
            yield json.dumps({"error": {"message": f"请求失败({response.status_code}): {response.text}"}})
            return

        for line in response.iter_lines():
            if not line:
                continue

            try:
                line = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            if line.startswith('data:'):
                line = line[5:].strip()

            if line == '[DONE]':
                finish_obj = {
                    "id": "chatcmpl-simulated",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }]
                }
                yield json.dumps(finish_obj)
                break

            try:
                # 处理可能出现的多个 JSON 合并情况
                json_objects = line.replace('}{', '}\n{').splitlines()
                for json_str in json_objects:
                    data = json.loads(json_str)
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        chunk_obj = {
                            "id": "chatcmpl-simulated",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "delta": {"content": content},
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield json.dumps(chunk_obj)
            except Exception as e:
                yield json.dumps({"error": {"message": f"[解析错误: {str(e)}]"}})

    def stream_chat(self, provider, messages):
        if provider not in self.provider_config:
            raise ValueError(f"不支持的厂商：{provider}")
        config = self.provider_config[provider]
        # 合并默认参数
        params = {**self.default_params, **config.get("default_params", {})}
        model = config["model"]
        try:
            if "client" in config:
                return self.stream_chat_openai(provider, messages, params, model)
            else:
                return self.stream_chat_http(provider, messages, params, model)
        except Exception as e:
            def error_gen():
                error_dict = {"error": {"message": f"{provider} API错误：{str(e)}"}}
                yield json.dumps(error_dict)
            return error_gen()

    def stream_chat_openai(self, provider, messages, params, model):
        config = self.provider_config[provider]
        try:
            client = config["client"](config)
            req_params = {
                "model": model,
                "messages": messages,
                "stream": True,
                **params
            }
            stream = client.chat.completions.create(**req_params)
            return self._handle_streaming_response_openai(stream, model)
        except Exception as e:
            def error_gen():
                error_dict = {"error": {"message": f"{provider} API错误：{str(e)}"}}
                yield json.dumps(error_dict)
            return error_gen()

    def stream_chat_http(self, provider, messages, params, model):
        config = self.provider_config[provider]
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv(config['api_key_env'])}"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if provider == "华为云":
                payload["stream_options"] = {"include_usage": True}
                response = requests.post(
                    config["url"],
                    headers=headers,
                    json=payload,
                    stream=True,
                    verify=False,
                    timeout=30
                )
            else:
                response = requests.post(
                    config["url"],
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=30
                )
            # 针对天翼云使用专用处理函数
            if provider == "天翼云":
                return self._handle_tianyi_streaming_response(provider, response, model)
            else:
                return self._handle_sse_streaming_response(provider, response, model)
        except Exception as e:
            def error_gen():
                error_dict = {"error": {"message": f"{provider} API错误：{str(e)}"}}
                yield json.dumps(error_dict)
            return error_gen()

if __name__ == "__main__":
    llm = UnifiedLLMClient()
    test_message = "简单介绍你自己"
    # 这里仅测试天翼云，可根据需要修改 provider 列表
    # providers = ["深度求索", "阿里云", "腾讯云", "火山引擎", "硅基流动", "华为云", "百度云", "潞晨云", "天翼云"]

    providers = ["天翼云"]
    for provider in providers:
        print(f"\n=== {provider.upper()} 测试 ===")
        try:
            for chunk in llm.stream_chat(provider, [{"role": "user", "content": test_message}]):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"错误：{str(e)}")
