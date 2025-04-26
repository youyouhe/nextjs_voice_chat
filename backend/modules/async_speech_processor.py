"""
异步语音处理器 - 使用Python异步特性实现LLM和TTS的并行处理和精确同步
"""
import logging
import time
import asyncio
import numpy as np
from fastrtc import AdditionalOutputs
from typing import Iterator, Tuple, Any, List, Dict, Optional, Coroutine

from backend.modules.asr import process_asr
from backend.modules.tts import TTSProcessor, audio_to_float32

class AsyncSpeechProcessor:
    """
    异步语音处理器，利用Python异步机制协调LLM文本生成和TTS音频合成
    实现非阻塞并行处理，同时保证文本和语音的同步
    """
    
    def __init__(self, config, messages, system_prompt):
        """
        初始化异步语音处理器
        
        Args:
            config: 配置对象 
            messages: 消息历史列表
            system_prompt: 系统提示词
        """
        self.config = config
        self.messages = messages
        self.system_prompt = system_prompt
        self.interrupted = False
        
        # 用于存储LLM处理过程中积累的文本
        self.text_buffer = ""
        # 段落结束标记
        self.sentence_end_marks = ['。', '！', '？', '!', '?', '.', '\n']
        # 设定中文和英文的最小段落长度阈值
        self.cn_min_length = 10  # 中文段落最小长度
        self.en_min_length = 40  # 英文段落最小长度
        
    def check_interrupt(self) -> bool:
        """检查是否被中断"""
        return self.interrupted
    
    def set_interrupted(self, value: bool):
        """设置中断状态"""
        self.interrupted = value
    
    async def process_llm_stream(self, client, prompt):
        """
        异步处理LLM文本流
        
        Args:
            client: OpenAI客户端
            prompt: 用户输入
            
        Yields:
            文本段落，当判断文本达到可处理条件时
        """
        # 添加用户消息到历史
        self.messages.append({"role": "user", "content": prompt})
        
        # 创建流式请求
        stream = await client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=self.messages,
            stream=True
        )
        
        full_response = ""
        segment_buffer = ""
        
        try:
            # 处理流式返回
            async for chunk in stream:
                if self.check_interrupt():
                    logging.info("LLM处理被中断")
                    break
                
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    segment_buffer += content
                    logging.info(f"LLM chunk: {content}")
                    
                    # 检查当前缓冲的文本是否可以作为一个段落发送
                    ready_segment = self._check_segment_ready(segment_buffer)
                    if ready_segment:
                        # 找到合适的分割点
                        cut_index = self._find_segment_break(segment_buffer)
                        if cut_index > 0:
                            # 提取段落并清空缓冲
                            current_segment = segment_buffer[:cut_index].strip()
                            segment_buffer = segment_buffer[cut_index:].strip()
                            
                            # 确保段落非空
                            if current_segment:
                                current_segment = current_segment.replace("*", " ")
                                logging.info(f"发现可发送段落: {current_segment}")
                                yield current_segment
                
                if chunk.choices[0].finish_reason == "stop":
                    # 流结束，处理剩余文本
                    if segment_buffer.strip():
                        logging.info(f"处理剩余段落: {segment_buffer}")
                        yield segment_buffer.strip()
                    break
            
            # 保存完整响应
            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})
                logging.info(f"LLM full response: {full_response}")
        except Exception as e:
            logging.error(f"处理LLM流出错: {str(e)}", exc_info=True)
    
    def _check_segment_ready(self, text: str) -> bool:
        """
        检查文本是否已经形成一个可发送的段落
        
        Args:
            text: 当前缓冲的文本
            
        Returns:
            布尔值，指示文本是否可以作为段落发送
        """
        # 检查文本是否为空
        if not text.strip():
            return False
        
        # 检查是否包含段落结束标记
        has_end_mark = any(mark in text for mark in self.sentence_end_marks)
        if not has_end_mark:
            return False
        
        # 检查文本长度是否达到阈值
        is_chinese = sum(1 for char in text if '\u4e00' <= char <= '\u9fa5') > len(text) / 2
        min_length = self.cn_min_length if is_chinese else self.en_min_length
        
        return len(text) >= min_length
    
    def _find_segment_break(self, text: str) -> int:
        """
        在文本中找到合适的分割点
        
        Args:
            text: 当前缓冲的文本
            
        Returns:
            分割点的索引
        """
        # 从右向左查找最近的句子结束标记
        for mark in self.sentence_end_marks:
            index = text.rfind(mark)
            if index != -1:
                return index + 1  # 包含结束标记
        
        # 如果没有找到，返回0表示没有合适的分割点
        return 0
    
    async def process_tts_for_segment(self, tts_processor, segment: str):
        """
        异步处理单个文本段的TTS生成
        
        Args:
            tts_processor: TTS处理器
            segment: 文本段落
            
        Yields:
            生成的音频数据
        """
        try:
            # 先收集所有音频块以计算总播放时间
            audio_chunks = list(tts_processor.process_tts(segment))
            if not audio_chunks:
                logging.warning(f"文本段 '{segment}' 未产生任何音频")
                return
                
            # 计算总音频长度和估计的播放时间
            total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
            bytes_per_second = self.config.AUDIO_SAMPLE_RATE * 2  # 16位音频，每采样2字节
            estimated_duration = total_audio_bytes / bytes_per_second
            
            logging.info(f"音频长度: {total_audio_bytes}字节, 估计播放时间: {estimated_duration:.2f}秒")
            
            # 播放音频前，先记录开始时间
            segment_start_time = time.time()
            
            # 播放所有音频块
            for audio_chunk in audio_chunks:
                if self.check_interrupt():
                    return
                
                # 转换音频格式
                audio_array = audio_to_float32(
                    np.frombuffer(audio_chunk, dtype=np.int16)
                )
                
                # 返回音频数据
                yield (self.config.AUDIO_SAMPLE_RATE, audio_array)
                
                # 适当让出控制权，允许其他协程执行
                await asyncio.sleep(0.01)
            
            # 计算已经播放的时间
            elapsed_time = time.time() - segment_start_time
            
            # 如果实际播放时间小于估计时间，进行等待使两者一致
            # 我们只等待估计时间的一定比例，以避免过长等待
            wait_factor = 0.95  # 等待估计播放时间的80%
            wait_time = max(0, (estimated_duration * wait_factor) - elapsed_time)
            
            if wait_time > 0:
                logging.info(f"等待中: {wait_time:.2f}秒，以同步文本和语音")
                
                # 分段小批次等待，以便能够响应中断
                wait_increment = 0.1  # 每次等待0.1秒
                waited_time = 0
                while waited_time < wait_time:
                    if self.check_interrupt():
                        logging.info("等待过程被中断")
                        break
                    
                    sleep_time = min(wait_increment, wait_time - waited_time)
                    await asyncio.sleep(sleep_time)
                    waited_time += sleep_time
            
            logging.info(f"段落处理完成, 总时长: {time.time() - segment_start_time:.2f}秒")
                
        except Exception as e:
            logging.error(f"处理TTS段落时出错: {str(e)}", exc_info=True)
    
    async def process_audio_async(self, audio):
        """
        异步处理音频输入，执行完整的语音交互流程
        
        Args:
            audio: 输入音频数据
            
        Yields:
            音频数据或文本数据
        """
        # 重置中断状态
        self.interrupted = False
        
        # 1. ASR语音识别 (同步调用，因为这是流程的开始)
        stt_time = time.time()
        prompt = process_asr(audio, self.config)
        
        if not prompt:
            logging.info("ASR返回空文本或失败，终止处理")
            return
            
        logging.info(f"STT耗时: {time.time() - stt_time:.2f}秒")
        
        # 2. 创建OpenAI客户端(异步)
        from openai import AsyncOpenAI
        import httpx
        
        # 创建一个异步传输层
        transport = httpx.AsyncHTTPTransport(verify=False)
        
        # 使用异步客户端
        async_client = AsyncOpenAI(
            base_url=self.config.OPENAI_BASE_URL,
            api_key=self.config.LLM_API_KEY,
            http_client=httpx.AsyncClient(transport=transport)
        )
        
        # 3. 创建TTS处理器
        tts_processor = TTSProcessor(self.config, self.check_interrupt)
        
        # 4. 开始处理LLM流和TTS流
        segment_index = 0
        total_segments = 0  # 将在处理过程中更新
        
        # 创建两个队列，用于LLM和TTS之间的通信
        segment_queue = asyncio.Queue()
        
        # 创建任务来处理LLM流
        async def llm_processor():
            try:
                async for segment in self.process_llm_stream(async_client, prompt):
                    if self.check_interrupt():
                        break
                        
                    # 将段落放入队列
                    await segment_queue.put(segment)
                    
                # 添加结束标记
                await segment_queue.put(None)
            except Exception as e:
                logging.error(f"LLM处理器异常: {e}", exc_info=True)
                await segment_queue.put(None)  # 确保在异常情况下也放入结束标记
        
        # 创建任务来处理TTS流
        async def tts_processor_task():
            nonlocal segment_index
            try:
                # 收集所有段落以计算总数
                segments = []
                while True:
                    segment = await segment_queue.get()
                    if segment is None:
                        break
                    segments.append(segment)
                
                # 设置总段落数
                nonlocal total_segments
                total_segments = len(segments)
                logging.info(f"总共收集了 {total_segments} 个文本段落")
                
                # 处理每个段落
                for i, segment in enumerate(segments):
                    if self.check_interrupt():
                        break
                    
                    # 发送当前文本段
                    llm_data = {
                        "type": "llm_chunk",
                        "content": segment,
                        "sync_info": {
                            "is_synced": True,
                            "segment_index": i,
                            "total_segments": total_segments,
                            "progress": round(i * 100 / total_segments if total_segments > 0 else 0)
                        }
                    }
                    yield AdditionalOutputs(llm_data)
                    logging.info(f"发送文本段[{i+1}/{total_segments}]到前端: '{segment}'")
                    
                    # 生成并播放TTS
                    audio_count = 0
                    async for audio_data in self.process_tts_for_segment(tts_processor, segment):
                        if self.check_interrupt():
                            break
                        yield audio_data
                        audio_count += 1
                    
                    logging.info(f"段落[{i+1}/{total_segments}]播放完成: {audio_count}个音频块")
                    segment_index = i + 1
                    
                    # 段落间添加很短的停顿
                    if i < total_segments - 1:
                        await asyncio.sleep(0.05)
                
                logging.info("所有文本段和音频处理完成")
            except Exception as e:
                logging.error(f"TTS处理器异常: {e}", exc_info=True)
        
        # 启动LLM处理任务
        llm_task = asyncio.create_task(llm_processor())
        
        # 开始TTS处理
        async for output in tts_processor_task():
            yield output
        
        # 等待LLM任务完成
        await llm_task
        
    def reset(self):
        """重置处理器，清除消息历史"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logging.info("语音处理器已重置")
