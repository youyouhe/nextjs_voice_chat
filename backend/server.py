import fastapi
from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions,AdditionalOutputs
from fastrtc.utils import audio_to_bytes, audio_to_float32, create_message
from openai import OpenAI
import logging
import uuid
import time
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import httpx
import requests # Added for custom ASR
import io       # Added for custom ASR
import os
import ssl
from backend.config import Config

# 配置日志级别
logging.basicConfig(level=logging.INFO)

# 从配置模块获取环境变量
LLM_API_KEY = Config.LLM_API_KEY
ELEVENLABS_API_KEY = Config.ELEVENLABS_API_KEY
import json
from typing import Iterator, Tuple

sys_prompt = """
你是一位专业的儿童英语单词学习助手"智多星单词伙伴"。你的任务是帮助孩子们系统地学习英语单词，遵循以下学习流程：

## 初始互动
1. 欢迎孩子进入智多星单词世界，简短介绍自己
2. 询问孩子想学习哪个分类的单词
3. 如果孩子没有明确选择，你随机选择3类供孩子们选择：

Section 1 基本常识
- Part 1 时间与方位：时间、日期、方向与位置
- Part 2 数字与度量衡：数字、数字相关、度量衡
- Part 3 货币常识：各种货币、货币兑换
- Part 4 自然常识：天气、颜色、环境、物质与材料
- Part 5 应急常识：呼救、救助

Section 2 居家生活
- Part 1 日常饮食：早餐、正餐、烹调手法、厨房用具、调料与香料、味觉与嗅觉、餐具、饮料、茶、酒类、饼干与甜点、乳制品、谷物、豆类、坚果与干果
- Part 2 水果、蔬菜与肉类：水果、蔬菜、肉类、海鲜
- Part 3 日常家务：常用工具、做家务、维修除虫、清洁用品、缝纫与洗涤

Section 3 日常生活
- Part 1 城市：城市街景、街道、日常标志、城市建筑、城市广场、店铺与商厦
- Part 2 生活：在银行、在邮局、在图书馆、在美发店、在幼儿园、在洗衣店、在加油站、在心理咨询中心、在法庭上、在警察局
- Part 3 购物：购物、砍价与付款、在超市、在百货店、在便利店、在药店、在糖果店、在面包店、在花店、在手表店、在珠宝店、在电器行、在体育用品店、在玩具店、在礼品店、在宠物店
- Part 4 餐饮：在中餐馆、中餐常见食品与菜品、在西餐厅、西餐常见食品与菜品、在快餐店、在咖啡馆

Section 4 电话与电脑
- Part 1：手机、打电话、发电子邮件

Section 5 人物
- Part 1 个人信息：个人情况、国籍、生肖与星座、性格、消遣、爱好、职业
- Part 2 人体：人体、上肢与下肢、头发、眼睛、性与生育、出生、成长与死亡
- Part 3 感觉、思想与动作：感官与感觉、情感、动作、思想、想象与意愿、能力与行为、动作与活动

Section 6 人与人的关系
- Part 1 家庭关系：家庭、人生轨迹、家庭与私人生活、婚礼、离婚、友情
- Part 2 社会结构与群体：社会结构与群体、感官与感觉、不同年龄的群体、一般的社会行为、正面的社会行为、负面的社会行为

Section 7 外表
- Part 1 服装鞋帽：服装

Section 10 运动健身
- Part 1 体育运动：体育运动相关、体育比赛、休闲运动、室内运动、户外运动、冬季运动、奥运会、田径运动、游泳池、游泳、潜水、体操、滑雪、瑜伽、武术、在健身房
- Part 2 球类运动：打篮球、踢足球、打羽毛球、打高尔夫球、打网球、打棒球

Section 11 健康养生
- Part 1 医院：医院、挂号处、医院科室、门诊室、手术室、医疗中心、病房、药房、产房
- Part 2 疾病与诊疗：疾病与症状、牙科、儿科、眼科、精神科、健康状况、疾病状况、各种疼痛与精神紊乱、医疗诊断与护理、怀孕与分娩

Section 12 职场工作
- Part 1 公司构成：公司与法人、成立与解散公司、组织结构、商务头衔与部门、公司部门与头衔
- Part 2 公司事务：求职面试、一般工作内容、并购与收购、问题与决策、会计术语、财务报表、工作评估与晋升、薪酬与福利
- Part 3 日常工作：办公室、会议室、电脑

Section 13 教育天地
- Part 1 学校：学校、学习科目、教室、宿舍、考场、常用文具
- Part 2 教育相关：文化教育、品行教育
- Part 3 初中级教育：幼儿园、小学、中学、教育分类及学校、课程、学习、测试
- Part 4 高等教育：大学、高等教育、学分与学位、高校科目、实验与论文、高等教育学科
- Part 5 知识学习：电脑、数学、几何、化学、物理、地理、生物、历史、美术

Section 14 政治军事
- Part 1 行政事务：行政与宪法、政治体制、选举、政体与政党、预算与财政
- Part 2 外交与战争：外交、国际事务、战争与军事、军队、现代战争

Section 15 节庆假日
- Part 1 节假日：假日与节庆

Section 16 旅游观光
- Part 1 出行准备：旅行用品、出行准备、在机场、在飞机上、入境与转机
- Part 2 餐饮与住宿：餐饮、在酒店

Section 17 自然天地
- Part 1 宇宙与地球：宇宙、天体、地理地貌、地理景观
- Part 2 地球生物：鸟类、昆虫、海底生物、丛林生物、极地、草原、热带雨林、沙漠
- Part 3 自然灾害：地震、台风、暴风雪、火山爆发、火灾、其他灾害

## 单词学习流程
1. 每次展示一个单词，简洁呈现:
   - 英文单词（单词前后不添加任何修饰符号*#，）
   - 询问孩子是否需要例句
   - 如需例句，提供英文例句并询问是否需要中文翻译
   - 如需翻译，提供单词和例句的中文含义

2. 学习节奏控制:
   - 询问孩子是否需要重复几遍单词
   - 确认孩子已掌握后再进入下一个单词
   - 每学完5个单词进行一次小组复习
   - 每学完5组(25个单词)进行一次全面复习

3. 复习方式:
   - 提供中文释义，让孩子说出对应的英文单词
   - 孩子回答正确，给予积极鼓励并继续下一个
   - 孩子回答错误，询问是否需要提示
   - 提供提示后让孩子再次尝试
   - 所有单词复习完成后，给予总体评价和鼓励

## 学习体验增强
1. 使用友好、鼓励的语气，适合儿童学习
2. 根据孩子的学习情况调整难度和速度
3. 适时给予积极反馈和鼓励
4. 使用简单有趣的方式解释单词
5. 在复习环节可以设计简单的游戏元素增加趣味性

请记住，你的目标是让孩子在轻松愉快的氛围中有效地学习和记忆英语单词。
"""

from openai.types.chat import ChatCompletionMessageParam
# 打印配置信息
Config.print_config()
messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": sys_prompt}]
# 创建一个禁用证书验证的传输层
transport = httpx.HTTPTransport(verify=False)
# 使用自定义传输层初始化 OpenAI 客户端
OPENAI_BASE_URL = Config.OPENAI_BASE_URL
openai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=LLM_API_KEY,
    http_client=httpx.Client(transport=transport)
)
#elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

logging.basicConfig(level=logging.INFO)


def echo(audio):
    # 用于检查是否被中断的函数
    # 在 FastRTC 框架中，中断会导致生成器被关闭
    # 我们可以使用一个简单的标志来跟踪中断状态
    interrupted = False
    def check_interrupt():
        return interrupted    

    stt_time = time.time()

    logging.info("Performing STT")

    # Comment out ElevenLabs ASR
    # transcription = elevenlabs_client.speech_to_text.convert(
    #     file=audio_to_bytes(audio),
    #     model_id="scribe_v1",
    #     tag_audio_events=False,
    #     language_code="eng",
    #     diarize=False,
    # )
    # prompt = transcription.text

    # Use custom ASR service
    try:
        audio_bytes = audio_to_bytes(audio)
        # Check if audio_bytes is empty before proceeding
        if not audio_bytes:
            logging.warning("audio_to_bytes returned empty bytes, skipping ASR.")
            return

        audio_bytes_io = io.BytesIO(audio_bytes)
        CUSTOM_ASR_URL = Config.CUSTOM_ASR_URL
        # Ensure filename is provided for multipart upload
        files = {'files': ('audio.mp3', audio_bytes_io, 'audio/mp3')}
        # 'keys' and 'lang' seem to be required based on the curl example
        data = {'keys': 'string', 'lang': 'auto'}
        headers = {'accept': 'application/json'}

        logging.info(f"Sending {len(audio_bytes)} bytes to ASR: {CUSTOM_ASR_URL}")
        response = requests.post(CUSTOM_ASR_URL, files=files, data=data, headers=headers, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        asr_result = response.json()
        logging.info(f"Custom ASR raw response: {asr_result}") # Log the full response

        # --- Updated logic to handle the specific ASR response structure ---
        prompt = '' # Default to empty string
        if isinstance(asr_result, dict) and 'result' in asr_result:
            result_list = asr_result['result']
            if isinstance(result_list, list) and len(result_list) > 0:
                first_result = result_list[0]
                if isinstance(first_result, dict) and 'text' in first_result:
                    prompt = first_result.get('text', '') # Use .get for safety
                else:
                    logging.warning(f"First item in 'result' list is not a dict or lacks 'text' key: {first_result}")
            else:
                logging.warning(f"'result' key is not a non-empty list: {result_list}")
        else:
            logging.warning(f"ASR response is not a dict or lacks 'result' key: {asr_result}")

    except requests.exceptions.Timeout:
        logging.error("Custom ASR service timed out.")
        prompt = ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling custom ASR service: {e}")
        prompt = "" # Handle error case
    except Exception as e:
        logging.error(f"Error processing custom ASR: {e}", exc_info=True) # Log traceback
        prompt = "" # Handle other potential errors


    if not prompt: # Check if prompt is empty or None after ASR call
        logging.info("ASR returned empty string or failed")
        return
    logging.info(f"ASR response: {prompt}") # Keep this log for consistency

    messages.append({"role": "user", "content": prompt})

    logging.info(f"STT took {time.time() - stt_time} seconds")

    llm_time = time.time()

    def text_stream():
        global full_response
        full_response = ""

        # Renamed 'stream' to 'completion_stream' to avoid name conflict
        LLM_MODEL = Config.LLM_MODEL
        completion_stream = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True
        )
        for chunk in completion_stream:
            # 检查是否被中断
            if check_interrupt():
                logging.info("LLM processing interrupted")
                break            

            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                logging.info(f"LLM chunk: {content}")  # Log the content
                full_response += content
                yield content
            if chunk.choices[0].finish_reason == "stop":
                break
    # audio_stream = elevenlabs_client.generate(
    #     text=text_stream(),
    #     voice="Rachel",  # Cassidy is also really good
    #     voice_settings=VoiceSettings(
    #         similarity_boost=0.9, stability=0.6, style=0.4, speed=1
    #     ),
    #     model="eleven_multilingual_v2",
    #     output_format="pcm_24000",
    #     stream=True,
    # )
    def audio_to_float32(audio_data: np.ndarray) -> np.ndarray:
        """将音频数据转换为float32格式，范围在[-1.0, 1.0]之间"""
        if audio_data.dtype == np.int16:
            return audio_data.astype(np.float32) / 32768.0
        return audio_data

    def generate_audio_stream(text_iterator: Iterator[str]) -> Iterator[bytes]:
        """
        从文本流生成音频流
        
        Args:
            text_iterator: 文本迭代器
        
        Returns:
            迭代器，每次返回原始音频字节数据
        """
        # 音频参数
        SAMPLE_RATE = Config.AUDIO_SAMPLE_RATE
        BIT_DEPTH = 16
        CHANNELS = 1
        # 计算缓冲区大小
        BUFFER_SIZE = int(SAMPLE_RATE * (BIT_DEPTH//8) * CHANNELS * Config.BUFFER_SIZE_FACTOR)
        
        # 缓存文本，当达到一定长度或遇到标点符号时发送请求
        text_buffer = ""
        sentence_end_marks = [',','.','，', '!', '?', '。', '！', '？', '\n',':','：']
        # 频繁检查中断的时间控制
        last_interrupt_check = time.time()        
        CHECK_INTERVAL = 0.02  # 每20ms检查一次
        
        # 根据TTS类型选择处理逻辑
        tts_type = Config.TTS_TYPE.lower()
        
        # 定义Fast TTS处理函数
        def process_with_fast_tts(text: str) -> Iterator[bytes]:
            """使用Fast TTS处理文本并生成音频"""
            if not text.strip():
                return
                
            logging.info(f"使用Fast TTS处理文本: {text}")
            
            # 准备请求数据
            payload = {
                "text": text,
                "prompt_audio_path": Config.FAST_TTS_PROMPT_AUDIO,
                "speed": Config.FAST_TTS_SPEED,
                "cfg_strength": Config.FAST_TTS_CFG_STRENGTH,
                "nfe_step": Config.FAST_TTS_NFE_STEP,
                "stream": True
            }
            
            try:
                response = requests.post(
                    Config.FAST_TTS_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True
                )
                
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=BUFFER_SIZE):
                        if check_interrupt():
                            logging.info("Fast TTS流处理被中断")
                            return
                        if chunk:
                            logging.debug(f"Fast TTS:输出音频块，大小={len(chunk)}")
                            yield chunk
                else:
                    logging.error(f"Fast TTS请求失败: {response.status_code} - {response.text}")
            
            except Exception as e:
                logging.error(f"Fast TTS请求异常: {str(e)}")
                
        # 定义硅基TTS处理函数
        def process_with_siliconflow_tts(text: str) -> Iterator[bytes]:
            """使用硅基TTS处理文本并生成音频"""
            if not text.strip():
                return
                
            logging.info(f"使用硅基TTS处理文本: {text}")
            
            # 准备请求数据：添加情感提示和模型选择
            # 硅基TTS需要格式：情感提示 <|endofprompt|> 实际文本
            input_text = f"{Config.SILICONFLOW_TTS_PROMPT}{text}"
            
            payload = {
                "input": input_text,
                "response_format": Config.SILICONFLOW_TTS_FORMAT,
                "sample_rate": Config.SILICONFLOW_TTS_SAMPLE_RATE,
                "stream": True,
                "speed": Config.SILICONFLOW_TTS_SPEED,
                "gain": Config.SILICONFLOW_TTS_GAIN,
                "model": Config.SILICONFLOW_TTS_MODEL,
                "voice": Config.SILICONFLOW_TTS_VOICE
            }
            
            headers = {
                "Authorization": f"Bearer {Config.SILICONFLOW_TTS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(
                    Config.SILICONFLOW_TTS_URL,
                    json=payload,
                    headers=headers,
                    stream=True
                )
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('Content-Length', 0))
                    offset = 0
                    
                    # 用于存储上一个奇数大小的 chunk
                    previous_chunk = None
                    
                    for chunk in response.iter_content(chunk_size=48000):
                        # logging.info(f"硅基TTS音频块: 大小={len(chunk)}, 类型={type(chunk)}, 偏移量={offset}, 总大小={total_size}")
                        offset += len(chunk)
                        
                        if check_interrupt():
                            logging.info("硅基TTS流处理被中断")
                            return
                        
                        if chunk:
                            chunk_length = len(chunk)
                            
                            if previous_chunk is not None:
                                # 将当前 chunk 与上一个奇数大小的 chunk 合并
                                chunk = previous_chunk + chunk
                                previous_chunk = None
                                chunk_length = len(chunk)
                                # logging.info(f"合并后的硅基TTS音频块: 大小={len(chunk)}, 类型={type(chunk)}")
                            
                            if chunk_length % 2 != 0:
                                # 如果当前 chunk 的大小是奇数，则将其存储起来，等待与下一个 chunk 合并
                                previous_chunk = chunk
                                continue
                            
                            yield chunk
                    
                    if previous_chunk is not None:
                        # 如果最后一个 chunk 的大小是奇数，则丢弃它
                        logging.warning("丢弃最后一个奇数大小的硅基TTS音频块")
                        
                else:
                    logging.error(f"硅基TTS请求失败: {response.status_code} - {response.text}")
            
            except Exception as e:
                logging.error(f"硅基TTS请求异常: {str(e)}")
        
        # 定义本地TTS处理函数
        def process_with_local_tts(text: str) -> Iterator[bytes]:
            """使用本地TTS服务处理文本并生成音频"""
            if not text.strip():
                return
                
            logging.info(f"使用本地TTS处理文本: {text}")
            
            # 本地TTS服务器配置
            TTS_SERVER = Config.TTS_SERVER
            ENDPOINT = Config.TTS_ENDPOINT
            
            # 准备请求数据
            payload = {
                "text": text,
                "prompt_text": Config.TTS_PROMPT_TEXT
            }
            
            try:
                response = requests.post(
                    f"{TTS_SERVER}{ENDPOINT}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    verify=False  # 如果使用自签名证书，需要设置为False
                )
                
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=BUFFER_SIZE):
                        logging.info(f"本地TTS音频块: 大小={len(chunk)}, 类型={type(chunk)}")
                        if check_interrupt():
                            logging.info("本地TTS流处理被中断")
                            return
                        if chunk:
                            logging.debug("本地TTS:输出音频块")
                            yield chunk
                else:
                    logging.error(f"本地TTS请求失败: {response.status_code} - {response.text}")
            
            except Exception as e:
                logging.error(f"本地TTS请求异常: {str(e)}")
        
        # 选择合适的TTS处理函数
        if tts_type == "siliconflow":
            process_tts = process_with_siliconflow_tts
            logging.info("使用硅基TTS服务")
        elif tts_type == "fast":
            process_tts = process_with_fast_tts
            logging.info("使用Fast TTS服务")
        else:  # 默认使用本地TTS
            process_tts = process_with_local_tts
            logging.info("使用本地TTS服务")
        
        try:
            for text_chunk in text_iterator:
                # 频繁检查中断
                current_time = time.time()
                if current_time - last_interrupt_check > CHECK_INTERVAL:
                    if check_interrupt():
                        logging.info("文本处理被中断")
                        return
                    last_interrupt_check = current_time
                    
                logging.info(f"Received text_chunk: {text_chunk}")
                text_buffer += text_chunk
                
                # 从左侧向右查找结束标记，一旦找到合适的就处理
                start_pos = 0
                while start_pos < len(text_buffer):
                    sent_end_pos = -1
                    found_mark = None

                    # 查找下一个最近的结束标记
                    for mark in sentence_end_marks:
                        pos = text_buffer.find(mark, start_pos)
                        if pos != -1 and (sent_end_pos == -1 or pos < sent_end_pos):
                            sent_end_pos = pos
                            found_mark = mark

                    # 如果找到了结束标记
                    if sent_end_pos != -1:
                        # 计算从开始到这个标记的文本段
                        segment = text_buffer[0:sent_end_pos + 1]
                        logging.info(f"Found mark: {found_mark}, index: {sent_end_pos}")
                        
                        # 判断当前文本段是否主要为中文，为其设置适当的阈值
                        segment_is_chinese = sum(1 for char in segment if '\u4e00' <= char <= '\u9fa5') > len(segment) / 2
                        logging.info(f"segment_is_chinese: {segment_is_chinese}")
                        segment_threshold = 10 if segment_is_chinese else 60
                        logging.info(f"segment_threshold: {segment_threshold}")
                        
                        # 如果这段文本长度超过适用于该段的阈值，处理它
                        if len(segment) >= segment_threshold:
                            # 再次检查中断
                            if check_interrupt():
                                return
                            
                            logging.info(f"Sending segment: {segment}")
                            # 使用选定的TTS处理函数处理文本
                            for audio_chunk in process_tts(segment):
                                yield audio_chunk
                            
                            # 更新缓冲区，移除已处理的部分
                            text_buffer = text_buffer[sent_end_pos + 1:]
                            logging.info(f"Updated text_buffer: {text_buffer}")
                            
                            # 重置开始位置，因为缓冲区已经更新
                            start_pos = 0
                        else:
                            # 如果这段文本长度不够，继续查找下一个标记
                            start_pos = sent_end_pos + 1
                    else:
                        # 没有找到更多的标记，跳出循环
                        break
            
            # 处理剩余的文本
            if text_buffer.strip() and not check_interrupt():
                for audio_chunk in process_tts(text_buffer):
                    yield audio_chunk
        except Exception as e:
            logging.error(f"音频流生成异常: {str(e)}", exc_info=True)
            if not check_interrupt():  # 只在非中断情况下记录异常
                raise
                
        except GeneratorExit:
            # 捕获生成器被关闭的情况
            nonlocal interrupted
            interrupted = True
            logging.info("Generator was closed externally (likely due to interruption)")

    # 创建一个缓冲区来存储已经传递给TTS的文本
    llm_chunks = []  # 存储所有LLM生成的文本块
    last_processed_index = 0  # 跟踪最后处理到的文本块索引
    last_sent_chunk_size = 0  # 跟踪发送到前端的文本总量
    
    # 直接使用LLM文本流，不需要队列和线程
    # 因为此处text_stream()是同步的，不会有并发问题
    text_for_tts = []  # 存储要传递给TTS的文本块
    for chunk in text_stream():
        if isinstance(chunk, str):
            llm_chunks.append(chunk)
            text_for_tts.append(chunk)
            logging.info(f"Got LLM chunk: '{chunk}'")
    
    # 如果LLM没有生成任何文本，直接返回
    if not text_for_tts:
        logging.warning("LLM did not generate any text")
        return
        
    # 将文本块合并成完整的文本
    full_text_for_tts = "".join(text_for_tts)
    logging.info(f"Full text for TTS: '{full_text_for_tts}'")
    
    # 生成音频流
    audio_stream = generate_audio_stream([full_text_for_tts])
    
    # 处理音频流
    try:
        # 每次处理音频块时，发送相应的文本块
        for audio_chunk in audio_stream:
            # 发送尚未发送的文本块
            if last_sent_chunk_size < len(full_response):
                # 找到一个合适的分割点，比如句子结束
                end_marks = ['.', ',', '!', '?', '。', '，', '！', '？', '\n', ':', '：']
                
                # 计算还剩余多少文本未发送
                remaining_text = full_response[last_sent_chunk_size:]
                
                # 默认每次最多发送50个字符
                max_send_length = 50
                
                # 在剩余文本中找最近的结束标记
                next_end_pos = max_send_length
                for mark in end_marks:
                    pos = remaining_text.find(mark, 0, max_send_length)
                    if pos != -1:
                        next_end_pos = pos + 1  # 包括标记在内
                        break
                
                # 获取要发送的文本片段
                text_chunk = remaining_text[:next_end_pos]
                
                if text_chunk.strip():  # 确保不发送空白文本
                    # 更新已发送文本大小
                    last_sent_chunk_size += len(text_chunk)
                    
                    # 发送LLM文本块
                    llm_data = {
                        "type": "llm_chunk",
                        "content": text_chunk,
                        "sync_info": {
                            "is_synced": True
                        }
                    }
                    
                    yield AdditionalOutputs(llm_data)
                    logging.info(f"Sent text chunk to frontend: '{text_chunk}'")
                
            # 处理音频输出
            if check_interrupt():
                logging.info("Audio output interrupted")
                break
                
            audio_array = audio_to_float32(
                np.frombuffer(audio_chunk, dtype=np.int16)
            )
            yield (Config.AUDIO_SAMPLE_RATE, audio_array)
        
        # 只有在未中断的情况下才添加消息历史
        if not check_interrupt() and full_response:
            messages.append({"role": "assistant", "content": full_response + " "})
            logging.info(f"LLM response: {full_response}")
            logging.info(f"LLM took {time.time() - llm_time} seconds")
    except GeneratorExit:
        # 捕获生成器被关闭的情况
        interrupted = True
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

stream = Stream(
    reply_on_pause,
    modality="audio",
    mode="send-receive",
    concurrency_limit=20,
    # additional_outputs_handler=lambda a, b: b,
)


import ssl
app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 为WebRTC添加特殊处理
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

class OfferBody(BaseModel):
    sdp: str | None = None
    candidate: dict | None = None
    type: str
    webrtc_id: str

@app.post("/webrtc/offer")
async def webrtc_offer(body: OfferBody):
    # 将自定义消息发送功能添加到响应处理函数中
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

# Load SSL certificate and key
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(Config.SSL_CERT_FILE, Config.SSL_KEY_FILE)

stream.mount(app)


@app.get("/reset")
async def reset():
    global messages
    logging.info("Resetting chat")
    messages = [{"role": "system", "content": sys_prompt}]
    return {"status": "success"}

# Run the app with HTTPS
if __name__ == "__main__":
    import uvicorn
    
    # 打印配置信息
    Config.print_config()
    
    uvicorn.run(app, 
                host=Config.SERVER_HOST, 
                port=Config.SERVER_PORT, 
                ssl_certfile=Config.SSL_CERT_FILE, 
                ssl_keyfile=Config.SSL_KEY_FILE)
