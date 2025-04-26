"""
主服务器文件 - 使用模块化结构重新实现服务器功能
"""
import fastapi
import logging
import ssl
import json
import time
from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions, AdditionalOutputs
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from backend.config import Config
from backend.modules.speech_processor import SpeechProcessor

# 配置日志级别
logging.basicConfig(level=logging.INFO)

# 加载系统提示词
sys_prompt = """
你是一位专业的儿童英语单词学习助手"智多星单词伙伴"。你的任务是帮助孩子们系统地学习英语单词，遵循以下学习流程：

## 初始互动
1. 欢迎孩子进入智多星单词世界，简短介绍自己
2. 询问孩子想学习哪个分类的单词
3. 如果孩子没有明确选择，你随机选择3类供孩子们选择：
[...省略部分系统提示...]
请记住，你的目标是让孩子在轻松愉快的氛围中有效地学习和记忆英语单词。
"""

# 打印配置信息
Config.print_config()

# 初始化消息历史
messages = [{"role": "system", "content": sys_prompt}]

# 创建语音处理器
speech_processor = SpeechProcessor(Config, messages, sys_prompt)

def echo(audio):
    """
    处理音频输入并产生回复
    
    Args:
        audio: 输入的音频数据
        
    Returns:
        生成器，产生语音和文本输出
    """
    try:
        # 处理音频并产生响应
        for output in speech_processor.process_audio(audio):
            yield output
    except GeneratorExit:
        # 捕获生成器被关闭的情况
        speech_processor.set_interrupted(True)
        logging.info("Generator was closed externally (likely due to interruption)")

# 使用配置创建 ReplyOnPause 实例
reply_on_pause = ReplyOnPause(
    echo,
    can_interrupt=Config.PAUSE_CAN_INTERRUPT,
    algo_options=AlgoOptions(
        audio_chunk_duration=Config.PAUSE_AUDIO_CHUNK_DURATION,
        started_talking_threshold=Config.PAUSE_STARTED_TALKING_THRESHOLD,
        speech_threshold=Config.PAUSE_SPEECH_THRESHOLD,
    ),
    model_options=SileroVadOptions(
        threshold=Config.VAD_THRESHOLD,
        min_speech_duration_ms=Config.VAD_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=Config.VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=Config.VAD_SPEECH_PAD_MS,
        max_speech_duration_s=Config.VAD_MAX_SPEECH_DURATION_S,
    ),
)

# 创建流服务
stream = Stream(
    reply_on_pause,
    modality="audio",
    mode="send-receive",
    concurrency_limit=20,
)

# 创建 FastAPI 应用
app = fastapi.FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义WebRTC请求体模型
class OfferBody(BaseModel):
    sdp: str | None = None
    candidate: dict | None = None
    type: str
    webrtc_id: str

# 处理WebRTC请求
@app.post("/webrtc/offer")
async def webrtc_offer(body: OfferBody):
    # 处理WebRTC请求并返回响应
    response = await stream.offer(body)
    return response

# 添加SSE端点来获取LLM文本块
@app.get("/llm_chunks")
async def llm_chunks_stream(webrtc_id: str):
    """
    使用Server-Sent Events流式获取LLM文本块
    """
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            if output.args:
                llm_data = output.args[0]
                if isinstance(llm_data, dict) and llm_data.get("type") == "llm_chunk":
                    # 返回SSE格式的数据
                    yield f"data: {json.dumps(llm_data)}\n\n"
    
    return StreamingResponse(
        output_stream(),
        media_type="text/event-stream"
    )

# 重置对话
@app.get("/reset")
async def reset():
    # 重置语音处理器的消息历史
    speech_processor.reset()
    return {"status": "success"}

# 挂载流服务
stream.mount(app)

# 主入口点
if __name__ == "__main__":
    import uvicorn
    
    # 加载SSL证书和密钥
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(Config.SSL_CERT_FILE, Config.SSL_KEY_FILE)
    
    # 运行服务器
    uvicorn.run(
        app, 
        host=Config.SERVER_HOST, 
        port=Config.SERVER_PORT, 
        ssl_certfile=Config.SSL_CERT_FILE, 
        ssl_keyfile=Config.SSL_KEY_FILE
    )
