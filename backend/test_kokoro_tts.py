"""
测试Kokoro TTS流式语音合成
该脚本用于测试Kokoro TTS的PCM流式输出功能
只记录音频数据流信息，不播放音频
"""
import os
import sys
import logging
from pathlib import Path
from typing import AsyncIterator, Optional, Dict, Any, Union
import asyncio
import time

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入配置
from backend.config import Config
config = Config()

# 导入OpenAI
from openai import OpenAI, AsyncOpenAI


# 定义异步测试函数
async def test_kokoro_tts_async():
    """异步测试Kokoro TTS的流式响应功能"""
    
    # 测试文本
    test_text = """这是一个测试文本，用于验证Kokoro TTS的PCM流式输出功能。
                如果您能听到这段话，说明我们的修改已经成功实现了。
                现在，我们来测试一些中英文混合的内容：
                Hello world! 你好，世界！
                Thank you for using our service. 感谢您使用我们的服务。"""
    
    logger.info(f"测试文本: {test_text}")
    logger.info(f"配置的TTS URL: {config.KOKORO_TTS_URL}")
    logger.info(f"配置的TTS模型: {config.KOKORO_TTS_MODEL}")
    logger.info(f"配置的TTS声音: {config.KOKORO_TTS_VOICE}")
    
    try:
        # 创建异步OpenAI客户端
        client = AsyncOpenAI(
            base_url=config.KOKORO_TTS_URL,
            api_key=config.KOKORO_TTS_API_KEY
        )
        
        logger.info("创建Kokoro TTS流式响应...")
        # 创建语音合成请求，指定PCM格式
        async with client.audio.speech.with_streaming_response.create(
            model=config.KOKORO_TTS_MODEL,
            voice=config.KOKORO_TTS_VOICE,
            input=test_text,
            response_format="pcm"
        ) as response:
            # 只记录音频数据流信息，不播放
            logger.info("开始接收音频数据流...")
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            async for chunk in response.aiter_bytes(chunk_size=1024):
                if not chunk:
                    continue
                chunk_count += 1
                logger.info(f"接收到音频块 #{chunk_count}，大小: {len(chunk)} 字节")
            
            end_time = asyncio.get_event_loop().time()
            logger.info(f"音频数据流接收完成，总共接收 {chunk_count} 个块，耗时 {end_time - start_time:.2f} 秒")
    
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)


# 同步版本的测试函数
def test_kokoro_tts_sync():
    """同步测试Kokoro TTS的流式响应功能"""
    
    # 测试文本
    test_text = """这是一个测试文本，用于验证Kokoro TTS的PCM流式输出功能。
                如果您能听到这段话，说明我们的修改已经成功实现了。
                现在，我们来测试一些中英文混合的内容：
                Hello world! 你好，世界！
                Thank you for using our service. 感谢您使用我们的服务。"""
    
    logger.info(f"测试文本: {test_text}")
    logger.info(f"配置的TTS URL: {config.KOKORO_TTS_URL}")
    logger.info(f"配置的TTS模型: {config.KOKORO_TTS_MODEL}")
    logger.info(f"配置的TTS声音: {config.KOKORO_TTS_VOICE}")
    
    try:
        # 创建同步OpenAI客户端
        client = OpenAI(
            base_url=config.KOKORO_TTS_URL,
            api_key=config.KOKORO_TTS_API_KEY
        )
        
        logger.info("创建Kokoro TTS流式响应...")
        # 创建语音合成请求，指定PCM格式
        with client.audio.speech.with_streaming_response.create(
            model=config.KOKORO_TTS_MODEL,
            voice=config.KOKORO_TTS_VOICE,
            input=test_text,
            response_format="pcm"
        ) as response:
            # 只记录音频数据流信息，不播放
            logger.info("开始接收音频数据流...")
            chunk_count = 0
            start_time = time.time()
            
            for chunk in response.iter_bytes(chunk_size=1024*10):
                if not chunk:
                    continue
                chunk_count += 1
                logger.info(f"接收到音频块 #{chunk_count}，大小: {len(chunk)} 字节")
            
            end_time = time.time()
            logger.info(f"音频数据流接收完成，总共接收 {chunk_count} 个块，耗时 {end_time - start_time:.2f} 秒")
    
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)


# OpenAI API测试函数
def test_openai_tts():
    """测试使用OpenAI API的TTS流式响应功能"""
    
    # 测试文本
    text_message = """这是一个使用OpenAI API的测试文本，用于验证TTS流式输出功能。
                如果您能听到这段话，说明流式输出功能正常工作。
                现在，我们来测试一些中英文混合的内容：
                Hello world! 你好，世界！
                Thank you for using our service. 感谢您使用我们的服务。"""
    
    logger.info(f"测试文本: {text_message}")
    
    try:
        # 创建OpenAI客户端
        openai_client = OpenAI(
            api_key=config.OPENAI_API_KEY
        )
        
        logger.info("创建OpenAI TTS流式响应...")
        # 创建语音合成请求，使用示例中的配置
        with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            response_format="mp3",  # 使用MP3格式
            input=text_message,
        ) as response:
            # 只记录音频数据流信息，不播放
            logger.info("开始接收音频数据流...")
            chunk_count = 0
            start_time = time.time()
            total_bytes = 0
            
            for chunk in response.iter_bytes(chunk_size=1024):
                if not chunk:
                    continue
                chunk_count += 1
                total_bytes += len(chunk)
                logger.info(f"接收到音频块 #{chunk_count}，大小: {len(chunk)} 字节")
            
            end_time = time.time()
            logger.info(f"音频数据流接收完成，总共接收 {chunk_count} 个块，总大小: {total_bytes} 字节，耗时 {end_time - start_time:.2f} 秒")
    
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    # 检查当前TTS类型是否为Kokoro
    if config.TTS_TYPE.lower() != "kokoro":
        logger.warning(f"当前配置的TTS类型为 {config.TTS_TYPE}，而非 'kokoro'")
        logger.info("继续测试，但请确保环境变量 TTS_TYPE=kokoro")
    
    logger.info("开始测试Kokoro TTS流式语音合成...")
    
    # 运行Kokoro TTS测试
    test_kokoro_tts_sync()
    
    # 运行OpenAI API测试
    # test_openai_tts()  # 取消注释以运行OpenAI API测试
