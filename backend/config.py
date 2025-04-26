import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置管理类，统一处理环境变量和默认值"""
    
    # 服务器配置
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "3000"))
    
    # SSL配置
    SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", "./backend/localhost+2.pem")
    SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", "./backend/localhost+2-key.pem")
    
    # API密钥
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    
    # 语言模型配置
    LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:27b")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://192.168.8.107:3000/v1")
    
    # ASR配置
    CUSTOM_ASR_URL = os.getenv("CUSTOM_ASR_URL", "http://192.168.8.107:50000/api/v1/asr")
    
    # TTS配置
    TTS_TYPE = os.getenv("TTS_TYPE", "local") # "local"、"siliconflow" 或 "fast"
    
    # 本地TTS配置
    TTS_SERVER = os.getenv("TTS_SERVER", "https://localhost:1234") 
    TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "/tts/stream")
    TTS_PROMPT_TEXT = os.getenv("TTS_PROMPT_TEXT", "希望你以后能够做的比我还好呦。")
    
    # 硅基TTS配置
    SILICONFLOW_TTS_URL = os.getenv("SILICONFLOW_TTS_URL", "https://api.siliconflow.cn/v1/audio/speech")
    SILICONFLOW_TTS_TOKEN = os.getenv("SILICONFLOW_TTS_TOKEN", "")
    SILICONFLOW_TTS_SAMPLE_RATE = int(os.getenv("SILICONFLOW_TTS_SAMPLE_RATE", "24000"))
    SILICONFLOW_TTS_SPEED = float(os.getenv("SILICONFLOW_TTS_SPEED", "1.0"))
    SILICONFLOW_TTS_GAIN = float(os.getenv("SILICONFLOW_TTS_GAIN", "0.0"))
    SILICONFLOW_TTS_MODEL = os.getenv("SILICONFLOW_TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B")
    SILICONFLOW_TTS_VOICE = os.getenv("SILICONFLOW_TTS_VOICE", "FunAudioLLM/CosyVoice2-0.5B:alex")
    SILICONFLOW_TTS_FORMAT = os.getenv("SILICONFLOW_TTS_FORMAT", "pcm")
    SILICONFLOW_TTS_PROMPT = os.getenv("SILICONFLOW_TTS_PROMPT", "<|endofprompt|>")
    
    # Fast TTS配置
    FAST_TTS_URL = os.getenv("FAST_TTS_URL", "http://localhost:8088/tts/stream")
    FAST_TTS_SAMPLE_RATE = int(os.getenv("FAST_TTS_SAMPLE_RATE", "24000"))
    FAST_TTS_SPEED = float(os.getenv("FAST_TTS_SPEED", "1.0"))
    FAST_TTS_CFG_STRENGTH = float(os.getenv("FAST_TTS_CFG_STRENGTH", "2.0"))
    FAST_TTS_NFE_STEP = int(os.getenv("FAST_TTS_NFE_STEP", "32"))
    FAST_TTS_PROMPT_AUDIO = os.getenv("FAST_TTS_PROMPT_AUDIO", "./src/f5_tts/infer/examples/basic/basic_ref_zh.wav")
    
    # 音频处理配置
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "24000"))
    BUFFER_SIZE_FACTOR = float(os.getenv("BUFFER_SIZE_FACTOR", "0.5"))
    
    # ReplyOnPause参数配置
    PAUSE_CAN_INTERRUPT = os.getenv("PAUSE_CAN_INTERRUPT", "true").lower() == "true"
    
    # AlgoOptions参数
    PAUSE_AUDIO_CHUNK_DURATION = float(os.getenv("PAUSE_AUDIO_CHUNK_DURATION", "1.2"))
    PAUSE_STARTED_TALKING_THRESHOLD = float(os.getenv("PAUSE_STARTED_TALKING_THRESHOLD", "0.55"))
    PAUSE_SPEECH_THRESHOLD = float(os.getenv("PAUSE_SPEECH_THRESHOLD", "0.3"))
    
    # SileroVadOptions参数
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.75"))
    VAD_MIN_SPEECH_DURATION_MS = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "550"))
    VAD_MIN_SILENCE_DURATION_MS = int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "1000"))
    VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "400")) 
    VAD_MAX_SPEECH_DURATION_S = int(os.getenv("VAD_MAX_SPEECH_DURATION_S", "30"))
    
    @classmethod
    def print_config(cls):
        """打印不敏感的配置参数"""
        logging.info("======== 当前配置参数 ========")
        # 服务器配置
        logging.info(f"服务器配置: {cls.SERVER_HOST}:{cls.SERVER_PORT}")
        
        # 模型配置
        logging.info(f"语言模型: {cls.LLM_MODEL}")
        logging.info(f"API基础URL: {cls.OPENAI_BASE_URL}")
        
        # ASR配置
        logging.info(f"语音识别服务: {cls.CUSTOM_ASR_URL}")
        
        # TTS配置
        logging.info(f"TTS类型: {cls.TTS_TYPE}")
        if cls.TTS_TYPE == "local":
            logging.info(f"本地语音合成服务: {cls.TTS_SERVER}{cls.TTS_ENDPOINT}")
        elif cls.TTS_TYPE == "siliconflow":
            logging.info(f"硅基语音合成服务: {cls.SILICONFLOW_TTS_URL}")
            logging.info(f"硅基TTS模型: {cls.SILICONFLOW_TTS_MODEL}")
            logging.info(f"硅基TTS格式: {cls.SILICONFLOW_TTS_FORMAT}")
            logging.info(f"硅基TTS采样率: {cls.SILICONFLOW_TTS_SAMPLE_RATE}")
            logging.info(f"硅基TTS速度: {cls.SILICONFLOW_TTS_SPEED}")
        elif cls.TTS_TYPE == "fast":
            logging.info(f"Fast语音合成服务: {cls.FAST_TTS_URL}")
            logging.info(f"Fast TTS采样率: {cls.FAST_TTS_SAMPLE_RATE}")
            logging.info(f"Fast TTS速度: {cls.FAST_TTS_SPEED}")
            logging.info(f"Fast TTS提示音频: {cls.FAST_TTS_PROMPT_AUDIO}")
        
        # 音频配置
        logging.info(f"音频采样率: {cls.AUDIO_SAMPLE_RATE}")
        logging.info(f"缓冲区大小因子: {cls.BUFFER_SIZE_FACTOR}")
        
        # ReplyOnPause配置
        logging.info("暂停检测配置:")
        logging.info(f"  - 是否可中断: {cls.PAUSE_CAN_INTERRUPT}")
        logging.info(f"  - 音频块持续时间: {cls.PAUSE_AUDIO_CHUNK_DURATION}秒")
        logging.info(f"  - 开始说话阈值: {cls.PAUSE_STARTED_TALKING_THRESHOLD}")
        logging.info(f"  - 语音阈值: {cls.PAUSE_SPEECH_THRESHOLD}")
        
        # VAD配置
        logging.info("语音活动检测配置:")
        logging.info(f"  - 检测阈值: {cls.VAD_THRESHOLD}")
        logging.info(f"  - 最小语音持续时间: {cls.VAD_MIN_SPEECH_DURATION_MS}ms")
        logging.info(f"  - 最小静音持续时间: {cls.VAD_MIN_SILENCE_DURATION_MS}ms")
        logging.info(f"  - 语音填充时间: {cls.VAD_SPEECH_PAD_MS}ms")
        logging.info(f"  - 最大语音持续时间: {cls.VAD_MAX_SPEECH_DURATION_S}秒")
        logging.info("==============================")
