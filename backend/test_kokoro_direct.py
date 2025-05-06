"""
使用requests库直接测试Kokoro TTS流式语音合成
避开OpenAI客户端可能存在的流结束检测问题
"""
import os
import sys
import logging
import json
import time
import pyaudio
import requests
from pathlib import Path

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入配置
from backend.config import Config
config = Config()


def test_kokoro_tts_direct():
    """直接使用requests库测试Kokoro TTS"""
    
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
    
    # 准备请求参数
    url = f"{config.KOKORO_TTS_URL}/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.KOKORO_TTS_API_KEY}"
    }
    payload = {
        "model": config.KOKORO_TTS_MODEL,
        "voice": config.KOKORO_TTS_VOICE,
        "input": test_text,
        "response_format": "pcm"
    }
    
    try:
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        sample_rate = config.AUDIO_SAMPLE_RATE
        
        # 打开音频流
        stream = p.open(
            format=pyaudio.paInt16,  # PCM格式为16位整型
            channels=1,               # 单声道
            rate=sample_rate,         # 采样率
            output=True               # 输出模式
        )
        
        logger.info("发送请求到Kokoro TTS服务...")
        start_time = time.time()
        
        # 发送请求并流式处理响应
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True  # 启用流式处理
        )
        
        if response.status_code == 200:
            logger.info(f"请求成功，状态码: {response.status_code}")
            
            # 设置块大小和计数器
            chunk_size = 1024
            chunk_count = 0
            
            # 处理响应流
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    logger.info("接收到空块，可能表示流结束")
                    continue
                
                chunk_count += 1
                if chunk_count % 10 == 0:
                    logger.info(f"已处理 {chunk_count} 个音频块，当前块大小: {len(chunk)} 字节")
                
                stream.write(chunk)
            
            end_time = time.time()
            logger.info(f"音频播放完成，总共处理了 {chunk_count} 个音频块，耗时 {end_time - start_time:.2f} 秒")
            logger.info("流已正常结束")
            
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
            if response.text:
                logger.error(f"错误信息: {response.text}")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
    
    finally:
        # 关闭音频流和PyAudio
        logger.info("清理资源...")
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()
        logger.info("测试结束")


if __name__ == "__main__":
    # 检查当前TTS类型是否为Kokoro
    if config.TTS_TYPE.lower() != "kokoro":
        logger.warning(f"当前配置的TTS类型为 {config.TTS_TYPE}，而非 'kokoro'")
        logger.info("继续测试，但请确保环境变量 TTS_TYPE=kokoro")
    
    logger.info("开始直接测试Kokoro TTS流式语音合成...")
    test_kokoro_tts_direct()
