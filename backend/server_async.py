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

# 加载系统提示词
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
   - 英文单词（单词前后不添加任何修饰符号比如星号，井号这类，）
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

# 打印配置信息
Config.print_config()

# 初始化消息历史
messages = [{"role": "system", "content": sys_prompt}]

# 创建语音处理器
speech_processor = AsyncSpeechProcessor(Config, messages, sys_prompt)

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
            loop.close()
    
    # 启动处理线程
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
    context.load_cert_chain(Config.SSL_CERT_FILE, Config.SSL_KEY_FILE)
    
    # 运行服务器
    uvicorn.run(
        app, 
        host=Config.SERVER_HOST, 
        port=Config.SERVER_PORT, 
        ssl_certfile=Config.SSL_CERT_FILE, 
        ssl_keyfile=Config.SSL_KEY_FILE
    )
