"""
异步服务器 - 使用Python asyncio协调LLM生成和TTS处理的完全异步实现版本
"""
import fastapi
import logging
import ssl
import json
import time
import asyncio
from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions, AdditionalOutputs
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from backend.config import Config
from backend.modules.async_speech_processor import AsyncSpeechProcessor

# 配置日志级别
logging.basicConfig(level=logging.INFO)

# 使用配置中的系统提示词
# 打印配置信息
Config().print_config()

# 初始化消息历史
messages = [{"role": "system", "content": Config().SYS_PROMPT}]

# 创建语音处理器
speech_processor = AsyncSpeechProcessor(Config(), messages, Config().SYS_PROMPT)

# 处理协程队列
import queue
import threading
import concurrent.futures

# 全局共享队列和线程池
response_queue = queue.Queue()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_async_in_thread(coro, *args, **kwargs):
    """在线程中运行异步函数，并将结果放入队列"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro(*args, **kwargs))
        response_queue.put(("result", result))
    except Exception as e:
        logging.error(f"异步函数执行错误: {e}", exc_info=True)
        response_queue.put(("error", e))
    finally:
        loop.close()

def process_audio_generator(audio):
    """处理音频并生成输出的同步生成器包装器"""
    # 创建处理队列
    output_queue = queue.Queue()
    end_marker = object()  # 结束标记
    
    # 在单独线程中处理音频
    def process_thread():
        async def collect_outputs():
            results = []
            try:
                # 使用异步方法处理音频
                async for output in speech_processor.process_audio_async(audio):
                    # 将输出放入队列
                    output_queue.put(output)
            except Exception as e:
                logging.error(f"处理音频时发生错误: {e}", exc_info=True)
            finally:
                # 添加结束标记
                output_queue.put(end_marker)
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 运行异步代码
            loop.run_until_complete(collect_outputs())
        finally:
            logging.info("音频处理线程结束")
            loop.close()
    
    # 启动处理线程
    logging.info("音频处理线程启动")
    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()
    
    # 从队列中读取结果
    try:
        while True:
            item = output_queue.get()
            if item is end_marker:
                break
            yield item
    except GeneratorExit:
        # 处理生成器关闭事件
        speech_processor.set_interrupted(True)
        logging.info("生成器被外部关闭")
    finally:
        # 等待线程结束
        thread.join(timeout=0.5)

# 封装echo函数，用于处理音频输入
def echo(audio):
    """
    处理音频输入并产生回复
    
    Args:
        audio: 输入的音频数据
        
    Returns:
        生成器，产生语音和文本输出
    """
    try:
        logging.info("enter echo ")
        # 使用处理器生成输出
        for output in process_audio_generator(audio):
            yield output
    except GeneratorExit:
        # 捕获生成器被关闭的情况
        speech_processor.set_interrupted(True)
        logging.info("Generator was closed externally (likely due to interruption)")

# 使用配置创建 ReplyOnPause 实例
reply_on_pause = ReplyOnPause(
    echo,
    can_interrupt=Config().PAUSE_CAN_INTERRUPT,
    algo_options=AlgoOptions(
        audio_chunk_duration=Config().PAUSE_AUDIO_CHUNK_DURATION,
        started_talking_threshold=Config().PAUSE_STARTED_TALKING_THRESHOLD,
        speech_threshold=Config().PAUSE_SPEECH_THRESHOLD,
    ),
    model_options=SileroVadOptions(
        threshold=Config().VAD_THRESHOLD,
        min_speech_duration_ms=Config().VAD_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=Config().VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=Config().VAD_SPEECH_PAD_MS,
        max_speech_duration_s=Config().VAD_MAX_SPEECH_DURATION_S,
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

@app.get("/reload_env")
async def reload_env():
    """重新加载.env配置文件并刷新系统配置"""
    from dotenv import load_dotenv
    import importlib
    from pathlib import Path
    
    try:
        # 获取.env文件路径（位于项目根目录）
        env_path = Path(__file__).parent.parent / '.env'
        logging.info(f"正在重新加载.env文件，路径: {env_path.absolute()}")
        
        # 重新加载.env文件
        load_dotenv(dotenv_path=env_path, override=True)
        
        # 重新导入Config模块
        import backend.config
        importlib.reload(backend.config)
        
        # 重新导入Config类并创建新实例
        global Config
        from backend.config import Config as ConfigClass
        # 使用类而不是实例
        Config = ConfigClass
        
        # 创建Config实例用于更新配置
        config_instance = Config()
        
        # 记录日志表明配置已更新
        logging.info("语音处理器配置和系统提示词已更新")
        
        # 重新初始化消息历史
        global messages
        messages = [{"role": "system", "content": config_instance.SYS_PROMPT}]
        
        # 更新语音处理器的配置
        speech_processor.update_config(config_instance)
        
        # 重新配置ReplyOnPause实例
        global reply_on_pause
        reply_on_pause = ReplyOnPause(
            echo,
            can_interrupt=config_instance.PAUSE_CAN_INTERRUPT,
            algo_options=AlgoOptions(
                audio_chunk_duration=config_instance.PAUSE_AUDIO_CHUNK_DURATION,
                started_talking_threshold=config_instance.PAUSE_STARTED_TALKING_THRESHOLD,
                speech_threshold=config_instance.PAUSE_SPEECH_THRESHOLD,
            ),
            model_options=SileroVadOptions(
                threshold=config_instance.VAD_THRESHOLD,
                min_speech_duration_ms=config_instance.VAD_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=config_instance.VAD_MIN_SILENCE_DURATION_MS,
                speech_pad_ms=config_instance.VAD_SPEECH_PAD_MS,
                max_speech_duration_s=config_instance.VAD_MAX_SPEECH_DURATION_S,
            ),
        )
        
        # 更新stream配置
        global stream
        stream = Stream(
            reply_on_pause,
            modality="audio",
            mode="send-receive",
            concurrency_limit=20,
        )
        
        # 打印更新后的配置
        config_instance.print_config()
        
        logging.info("环境变量已重新加载")
        return {"status": "success", "message": "环境变量已成功重新加载"}
    except Exception as e:
        logging.error(f"重新加载环境变量失败: {str(e)}")
        return {"status": "error", "message": f"重新加载环境变量失败: {str(e)}"}, 500

# 挂载流服务
stream.mount(app)

# 在应用程序结束时关闭线程池
@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=False)
    logging.info("线程池已关闭")

# 主入口点
if __name__ == "__main__":
    import uvicorn
    
    # 加载SSL证书和密钥
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(Config().SSL_CERT_FILE, Config().SSL_KEY_FILE)
    
    # 运行服务器
    uvicorn.run(
        app, 
        host=Config().SERVER_HOST,
        port=Config().SERVER_PORT,
        ssl_certfile=Config().SSL_CERT_FILE,
        ssl_keyfile=Config().SSL_KEY_FILE
    )
