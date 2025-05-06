import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
try:
    env_path = Path(__file__).parent.parent / '.env'
    print(f"Loading .env from: {env_path.absolute()}")  # 打印绝对路径
    load_dotenv(dotenv_path=env_path, verbose=True,override=True)
except FileNotFoundError:
    print("Error: .env file not found at expected location")
    raise

# 环境变量调试开关
debug_value = os.getenv("DEBUG_ENV")
print(f"debug value: {debug_value}")  # 打印绝对路径

DEBUG_ENV = debug_value.lower() == "true" if debug_value else False

class Config:
    """配置管理类，统一处理环境变量和默认值，使用动态属性获取最新环境变量"""
    
    # 默认值配置字典
    _defaults = {
        # 系统提示词配置
        "SYS_PROMPT": """
你是一位专业的儿童英语单词学习助手"智多星单词伙伴"。你的任务是帮助孩子们系统地学习英语单词，遵循以下学习流程：

## 初始互动
1. 欢迎孩子进入智多星单词世界，简短介绍自己
2. 询问孩子想学习哪个分类的单词
3. 如果孩子没有明确选择，你随机选择3类供孩子们选择：



## 学习对话要求
let's the each conversation in a few words
""",

        # 服务器配置
        "SERVER_HOST": "0.0.0.0",
        "SERVER_PORT": 3000,
        
        # SSL配置
        "SSL_CERT_FILE": "./backend/localhost+2.pem",
        "SSL_KEY_FILE": "./backend/localhost+2-key.pem",
        
        # API密钥
        "LLM_API_KEY": "",
        "ELEVENLABS_API_KEY": "",
        
        # 语言模型配置
        "LLM_MODEL": "gemma3:27b",
        "OPENAI_BASE_URL": "https://192.168.8.107:3000/v1",
        
        # ASR配置
        "CUSTOM_ASR_URL": "http://192.168.8.107:50000/api/v1/asr",
        
        # TTS配置
        "TTS_TYPE": "local",  # "local"、"siliconflow"、"fast" 或 "kokoro"
        
        # 本地TTS配置
        "TTS_SERVER": "https://localhost:1234",
        "TTS_ENDPOINT": "/tts/stream",
        "TTS_PROMPT_TEXT": "希望你以后能够做的比我还好呦。",
        
        # Kokoro TTS配置
        "KOKORO_TTS_URL": "http://192.168.8.107:8880/v1",
        "KOKORO_TTS_API_KEY": "not-needed",
        "KOKORO_TTS_MODEL": "kokoro",
        "KOKORO_TTS_VOICE": "af_sky+af_bella",
        
        # 硅基TTS配置
        "SILICONFLOW_TTS_URL": "https://api.siliconflow.cn/v1/audio/speech",
        "SILICONFLOW_TTS_TOKEN": "",
        "SILICONFLOW_TTS_SAMPLE_RATE": 24000,
        "SILICONFLOW_TTS_SPEED": 1.0,
        "SILICONFLOW_TTS_GAIN": 0.0,
        "SILICONFLOW_TTS_MODEL": "FunAudioLLM/CosyVoice2-0.5B",
        "SILICONFLOW_TTS_VOICE": "FunAudioLLM/CosyVoice2-0.5B:alex",
        "SILICONFLOW_TTS_FORMAT": "pcm",

        
        # Fast TTS配置
        "FAST_TTS_URL": "http://localhost:8088/tts/stream",
        "FAST_TTS_SAMPLE_RATE": 24000,
        "FAST_TTS_SPEED": 1.0,
        "FAST_TTS_CFG_STRENGTH": 2.0,
        "FAST_TTS_NFE_STEP": 32,
        "FAST_TTS_PROMPT_AUDIO": "./src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
        
        # 音频处理配置
        "AUDIO_SAMPLE_RATE": 24000,
        "BUFFER_SIZE_FACTOR": 0.5,
        
        # ReplyOnPause参数配置
        "PAUSE_CAN_INTERRUPT": True,
        
        # AlgoOptions参数
        "PAUSE_AUDIO_CHUNK_DURATION": 1.2,
        "PAUSE_STARTED_TALKING_THRESHOLD": 0.55,
        "PAUSE_SPEECH_THRESHOLD": 0.3,
        
        # SileroVadOptions参数
        "VAD_THRESHOLD": 0.3,  # 默认值改为0.25
        "VAD_MIN_SPEECH_DURATION_MS": 550,
        "VAD_MIN_SILENCE_DURATION_MS": 1000,
        "VAD_SPEECH_PAD_MS": 400,
        "VAD_MAX_SPEECH_DURATION_S": 30,
    }
    
    @classmethod
    def _get_value(cls, key, default_value=None, value_type=str):
        """从环境变量中获取值，如果不存在则使用默认值"""
        env_value = os.getenv(key)
        
        if env_value is None:
            return default_value
            
        # 根据类型转换值
        if value_type == bool:
            return env_value.lower() == "true"
        elif value_type == int:
            return int(env_value)
        elif value_type == float:
            return float(env_value)
        else:
            return env_value
    
    @classmethod
    def get(cls, key):
        """获取配置项值，每次都从环境变量中读取最新值"""
        default = cls._defaults.get(key)
        
        # 根据默认值的类型确定转换类型
        if isinstance(default, bool):
            return cls._get_value(key, default, bool)
        elif isinstance(default, int):
            return cls._get_value(key, default, int)
        elif isinstance(default, float):
            return cls._get_value(key, default, float)
        else:
            return cls._get_value(key, default, str)
            
    # 使用属性描述符动态获取配置
    def __getattr__(self, name):
        """改进属性访问逻辑，优先从环境变量获取，最后尝试默认值"""
        try:
            # 优先使用环境变量
            if name in os.environ:
                return self.get(name)
                
            # 其次使用默认值
            if name in self._defaults:
                return self.get(name)
                
            # 都不存在时抛出异常
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        except Exception as e:
            logging.error(f"获取配置属性 {name} 时发生错误: {str(e)}")
            raise
            
    # 创建类实例以支持属性访问
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
            
    @classmethod
    def get_instance(cls):
        return cls._instance
    
    @classmethod
    def print_config(cls):
        """打印不敏感的配置参数"""
        logging.info("======== 当前配置参数 ========")
        # 获取实例
        config = cls.get_instance()
        
        # 服务器配置
        logging.info(f"服务器配置: {config.SERVER_HOST}:{config.SERVER_PORT}")
        
        # 模型配置
        logging.info(f"语言模型: {config.LLM_MODEL}")
        logging.info(f"API基础URL: {config.OPENAI_BASE_URL}")
        
        # ASR配置
        logging.info(f"语音识别服务: {config.CUSTOM_ASR_URL}")
        
        # TTS配置
        logging.info(f"TTS类型: {config.TTS_TYPE}")
        if config.TTS_TYPE == "local":
            logging.info(f"本地语音合成服务: {config.TTS_SERVER}{config.TTS_ENDPOINT}")
        elif config.TTS_TYPE == "siliconflow":
            logging.info(f"硅基语音合成服务: {config.SILICONFLOW_TTS_URL}")
            logging.info(f"硅基TTS模型: {config.SILICONFLOW_TTS_MODEL}")
            logging.info(f"硅基TTS格式: {config.SILICONFLOW_TTS_FORMAT}")
            logging.info(f"硅基TTS采样率: {config.SILICONFLOW_TTS_SAMPLE_RATE}")
            logging.info(f"硅基TTS速度: {config.SILICONFLOW_TTS_SPEED}")
        elif config.TTS_TYPE == "fast":
            logging.info(f"Fast语音合成服务: {config.FAST_TTS_URL}")
            logging.info(f"Fast TTS采样率: {config.FAST_TTS_SAMPLE_RATE}")
            logging.info(f"Fast TTS速度: {config.FAST_TTS_SPEED}")
            logging.info(f"Fast TTS提示音频: {config.FAST_TTS_PROMPT_AUDIO}")
        elif config.TTS_TYPE == "kokoro":
            logging.info(f"Kokoro语音合成服务: {config.KOKORO_TTS_URL}")
            logging.info(f"Kokoro TTS模型: {config.KOKORO_TTS_MODEL}")
            logging.info(f"Kokoro TTS声音: {config.KOKORO_TTS_VOICE}")
        
        # 音频配置
        logging.info(f"音频采样率: {config.AUDIO_SAMPLE_RATE}")
        logging.info(f"缓冲区大小因子: {config.BUFFER_SIZE_FACTOR}")
        
        # ReplyOnPause配置
        logging.info("暂停检测配置:")
        logging.info(f"  - 是否可中断: {config.PAUSE_CAN_INTERRUPT}")
        logging.info(f"  - 音频块持续时间: {config.PAUSE_AUDIO_CHUNK_DURATION}秒")
        logging.info(f"  - 开始说话阈值: {config.PAUSE_STARTED_TALKING_THRESHOLD}")
        logging.info(f"  - 语音阈值: {config.PAUSE_SPEECH_THRESHOLD}")
        
        # VAD配置
        logging.info("语音活动检测配置:")
        logging.info(f"  - 检测阈值: {config.VAD_THRESHOLD}")
        logging.info(f"  - 最小语音持续时间: {config.VAD_MIN_SPEECH_DURATION_MS}ms")
        logging.info(f"  - 最小静音持续时间: {config.VAD_MIN_SILENCE_DURATION_MS}ms")
        logging.info(f"  - 语音填充时间: {config.VAD_SPEECH_PAD_MS}ms")
        logging.info(f"  - 最大语音持续时间: {config.VAD_MAX_SPEECH_DURATION_S}秒")
        logging.info("==============================")


# 在Config类定义之后，使用环境变量调试开关
if DEBUG_ENV:
    print("\n====== 环境变量调试输出 ======")
    # 输出所有与项目相关的环境变量
    for key in os.environ:
        if key in Config._defaults or key.startswith(("SERVER_", "SSL_", "LLM_", "TTS_", "OPENAI_", 
                                                     "CUSTOM_", "SILICONFLOW_", "FAST_", "AUDIO_", 
                                                     "PAUSE_", "VAD_", "ELEVENLABS_", "SYS_PROMPT")):
            # 如果是密钥类的敏感数据，则不显示具体值
            if "KEY" in key or "TOKEN" in key or "SECRET" in key:
                value = "******" if os.environ[key] else "[未设置]"
            else:
                value = os.environ[key]
            print(f"{key} = {value}")
    print("==============================\n")
