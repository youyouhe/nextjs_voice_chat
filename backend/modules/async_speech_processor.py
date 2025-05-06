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
        # self.sentence_end_marks = ['。', '！', '？', '!', '?', '.', '\n']
        self.sentence_end_marks = [',','.','，', '!', '?', '。', '！', '？', '\n',':','：']
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
                    logging.info("LLM处理被中断，主动关闭流")
                    # 使用AsyncOpenAI客户端API取消请求（如果支持）
                    try:
                        await stream.aclose()  # 假设存在此方法
                        logging.info("LLM close，主动关闭流")
                    except:
                        logging.info("LLM close，出现异常")
                        pass
                    break
                
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    segment_buffer += content
                    # logging.info(f"LLM chunk: {full_response}")
                    
                    # 检查当前缓冲的文本是否可以作为一个段落发送
                    ready_segment = self._check_segment_ready(segment_buffer)
                    if ready_segment:
                        # 找到合适的分割点
                        cut_index = self._find_segment_break(segment_buffer)
                        if cut_index > 4:
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
                    logging.info("audio_chunk close，主动关闭流")
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
    
    # 清空队列的辅助方法
    async def _clear_queue(self, queue):
        """清空队列中的所有内容"""
        try:
            logging.info(f"开始清空队列，当前内容数量: {queue.qsize()}")
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            logging.info("队列已清空")
        except Exception as e:
            logging.error(f"清空队列时出错: {e}")
    
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
        
        # 4. 实时并行处理LLM和TTS流
        segment_index = 0  # 当前处理的段落索引
        received_segments = []  # 已接收的段落列表，用于计算进度
        is_llm_complete = False  # 标记LLM是否已完成生成
        
        # 创建队列，用于LLM和TTS之间的实时通信
        segment_queue = asyncio.Queue()
        
        # 中断处理器 - 监控中断标志并清理资源
        async def interrupt_handler(check_interval=0.5):
            while not is_llm_complete:
                if self.check_interrupt():
                    logging.info("检测到中断，清理队列和取消任务...")
                    # 清空队列中的所有段落
                    await self._clear_queue(segment_queue)
                    # 添加结束标记到队列，通知TTS处理器退出
                    await segment_queue.put(None)
                    return True
                await asyncio.sleep(check_interval)
            return False
                
        # LLM处理器和TTS处理器作为两个并行任务运行
        async def llm_processor():
            nonlocal is_llm_complete
            try:
                async for segment in self.process_llm_stream(async_client, prompt):
                    if self.check_interrupt():
                        logging.info("LLM处理中断，停止生成")
                        break
                    
                    # 将段落放入队列，立即可用于TTS处理
                    await segment_queue.put(segment)
                    # 添加到已接收段落列表，用于计算进度
                    received_segments.append(segment)
                    logging.info(f"已放入队列段落数：{len(received_segments)}")
                
                # 标记LLM处理完成
                is_llm_complete = True
                logging.info("LLM处理完成，标记结束")
                # 添加结束标记到队列
                if not self.check_interrupt():  # 只有在非中断状态下才添加结束标记
                    await segment_queue.put(None)
            except Exception as e:
                logging.error(f"LLM处理器异常: {e}", exc_info=True)
                is_llm_complete = True
                await segment_queue.put(None)  # 确保在异常情况下也放入结束标记
        
        # TTS处理器 - 处理队列中的段落
        async def tts_processor_task():
            nonlocal segment_index
            try:
                while True:
                    # 如果已中断，不再处理新的段落
                    if self.check_interrupt():
                        logging.info("TTS处理被中断，停止处理队列")
                        # 清空队列中的所有段落
                        await self._clear_queue(segment_queue)                        
                        break
                    
                    # 从队列获取下一个段落
                    segment = await segment_queue.get()
                    
                    # 检查是否为结束标记
                    if segment is None:
                        logging.info("收到队列结束标记，TTS处理完成")
                        break
                    
                    # 再次检查中断，以防在获取段落后被中断
                    if self.check_interrupt():
                        logging.info("段落获取后检测到中断，跳过处理")
                        segment_queue.task_done()
                        break
                    
                    # 计算当前段落索引和总进度
                    current_index = segment_index
                    segment_index += 1
                    total_received = len(received_segments)
                    
                    # 基于已知的段落数量计算进度，如果LLM尚未完成，则进度估计为已处理/已接收
                    progress_percentage = round(current_index * 100 / total_received) if total_received > 0 else 0
                    
                    # 发送当前文本段到前端
                    llm_data = {
                        "type": "llm_chunk",
                        "content": segment,
                        "sync_info": {
                            "is_synced": True,
                            "segment_index": current_index,
                            "total_segments_so_far": total_received,
                            "llm_complete": is_llm_complete,
                            "progress": progress_percentage
                        }
                    }
                    
                    yield AdditionalOutputs(llm_data)
                    # logging.info(f"发送文本段[{current_index+1}/{total_received}+]到前端: '{segment}'")
                    
                    # 立即生成并播放TTS音频
                    audio_count = 0
                    was_interrupted = False
                    async for audio_data in self.process_tts_for_segment(tts_processor, segment):
                        if self.check_interrupt():
                            logging.info("TTS音频生成过程中被中断")
                            was_interrupted = True
                            break
                        yield audio_data
                        audio_count += 1
                    
                    # 检查是否在音频生成过程中被中断
                    if was_interrupted:
                        logging.info("音频生成被中断，清空剩余队列")
                        # 标记当前任务已完成
                        segment_queue.task_done()
                        # 清空整个队列，放弃剩余内容
                        await self._clear_queue(segment_queue)
                        # 跳出主循环，准备新会话
                        break
                    else:
                        # 只有在成功完成时才记录
                        # logging.info(f"段落[{current_index+1}/{total_received}+]播放完成: {audio_count}个音频块")
                        # 标记任务已完成
                        segment_queue.task_done()
                    
                    # 段落间添加很短的停顿，但不会阻塞下一段的处理
                    if (not is_llm_complete or current_index < total_received - 1) and not self.check_interrupt():
                        await asyncio.sleep(0.05)
                
                if not self.check_interrupt():
                    logging.info(f"所有文本段和音频处理完成，共处理了 {segment_index} 个段落")
            
            except Exception as e:
                logging.error(f"TTS处理器异常: {e}", exc_info=True)
        
        # 启动LLM处理任务和中断监控任务
        llm_task = asyncio.create_task(llm_processor())
        interrupt_task = asyncio.create_task(interrupt_handler())
        
        # 直接在主协程中运行TTS处理任务，避免嵌套异步生成器的问题
        try:
            # 运行TTS处理任务并转发所有输出
            async for output in tts_processor_task():
                yield output
                
            # 确保所有任务完成，但不要阻塞主流程
            pending_tasks = []
            if not llm_task.done():
                logging.info("等待LLM任务完成...")
                pending_tasks.append(llm_task)
            
            if not interrupt_task.done():
                # 取消中断监控任务，因为主处理已完成
                interrupt_task.cancel()
            
            # 等待剩余任务完成
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                
        except Exception as e:
            logging.error(f"主处理流程异常: {e}", exc_info=True)
            # 确保取消所有任务
            if not llm_task.done():
                llm_task.cancel()
            if not interrupt_task.done():
                interrupt_task.cancel()
        
    def reset(self):
        """重置处理器，清除消息历史"""
        # 使用配置中的最新系统提示词
        self.system_prompt = self.config.SYS_PROMPT
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logging.info("语音处理器已重置，使用最新系统提示词")
        
    def update_config(self, new_config):
        """更新处理器的配置
        
        Args:
            new_config: 新的配置对象
        """
        self.config = new_config
        # 同时更新系统提示词
        self.system_prompt = self.config.SYS_PROMPT
        logging.info("语音处理器配置和系统提示词已更新")
