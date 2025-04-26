"""
语音处理器模块 - 整合ASR、LLM和TTS的完整语音处理流程
"""
import logging
import time
import numpy as np
from fastrtc import AdditionalOutputs
from typing import Iterator, Tuple, Any, List, Dict, Optional

from backend.modules.asr import process_asr
from backend.modules.llm import LLMProcessor
from backend.modules.tts import TTSProcessor, audio_to_float32

class SpeechProcessor:
    """语音处理器，协调ASR、LLM和TTS的完整处理流程"""
    
    def __init__(self, config, messages, system_prompt):
        """
        初始化语音处理器
        
        Args:
            config: 配置对象 
            messages: 消息历史列表
            system_prompt: 系统提示词
        """
        self.config = config
        self.messages = messages
        self.system_prompt = system_prompt
        self.interrupted = False
    
    def check_interrupt(self) -> bool:
        """
        检查是否被中断
        
        Returns:
            如果被中断则返回True
        """
        return self.interrupted
    
    def set_interrupted(self, value: bool):
        """设置中断状态"""
        self.interrupted = value
        
    def process_audio(self, audio):
        """
        处理音频输入，执行完整的语音交互流程
        
        Args:
            audio: 输入音频数据
            
        Returns:
            迭代器，产生音频数据和文本数据
        """
        # 重置中断状态
        self.interrupted = False
        
        # 1. ASR语音识别
        stt_time = time.time()
        prompt = process_asr(audio, self.config)
        
        if not prompt:
            logging.info("ASR返回空文本或失败，终止处理")
            return
            
        logging.info(f"STT耗时: {time.time() - stt_time:.2f}秒")
        
        # 2. LLM处理
        llm_processor = LLMProcessor(self.config, self.messages, self.check_interrupt)
        llm_processor.add_message("user", prompt)
        
        # 3. 收集LLM输出
        text_for_tts = []
        for text_chunk in llm_processor.generate_text_stream():
            text_for_tts.append(text_chunk)
        
        # 如果LLM没有生成文本或者处理被中断，退出
        if not text_for_tts or self.check_interrupt():
            return
            
        # 合并文本
        full_text = "".join(text_for_tts)
        logging.info(f"LLM完整响应: {full_text}")
        
        # 4. 创建TTS处理器
        tts_processor = TTSProcessor(self.config, self.check_interrupt)
        
        # 5. 创建文本同步器
        text_sync = llm_processor.get_text_synchronizer()
        
        # 6. 处理TTS音频流
        audio_stream = tts_processor.generate_audio_stream([full_text])
        last_sent_chunk_size = 0
        
        # 将文本按语音单位分段
        text_segments = self._segment_text_for_tts(full_text)
        logging.info(f"文本已分段为{len(text_segments)}个段落用于TTS处理")
        
        # 优化：创建可视化进度条
        total_segments = len(text_segments)
        current_segment_index = 0
        
        # 处理并生成音频流 - 即时处理，无需预生成全部音频
        try:
            # 逐段处理，实现同步处理
            for i, segment in enumerate(text_segments):
                if self.check_interrupt():
                    logging.info("处理被中断")
                    break
                
                # 先发送文本段
                llm_data = {
                    "type": "llm_chunk",
                    "content": segment,
                    "sync_info": {
                        "is_synced": True,
                        "segment_index": i,
                        "total_segments": total_segments,
                        "progress": round(i * 100 / total_segments)
                    }
                }
                yield AdditionalOutputs(llm_data)
                logging.info(f"发送文本段[{i+1}/{total_segments}]到前端: '{segment}'")
                
                # 首先收集所有TTS音频块以计算总播放时间
                audio_chunks = list(tts_processor.process_tts(segment))
                if not audio_chunks:
                    logging.warning(f"文本段 '{segment}' 未产生任何音频，跳过")
                    continue
                
                # 计算总音频长度和估计的播放时间
                total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
                bytes_per_second = self.config.AUDIO_SAMPLE_RATE * 2  # 16位音频，每采样2字节
                estimated_duration = total_audio_bytes / bytes_per_second
                
                logging.info(f"段落[{i+1}/{total_segments}]音频长度: {total_audio_bytes}字节, " 
                           f"估计播放时间: {estimated_duration:.2f}秒")
                
                # 播放音频前，先记录开始时间
                segment_start_time = time.time()
                
                # 播放音频
                for audio_chunk in audio_chunks:
                    if self.check_interrupt():
                        break
                        
                    audio_array = audio_to_float32(
                        np.frombuffer(audio_chunk, dtype=np.int16)
                    )
                    yield (self.config.AUDIO_SAMPLE_RATE, audio_array)
                
                # 计算已经播放的时间
                elapsed_time = time.time() - segment_start_time
                
                # 如果实际播放时间小于估计时间，进行等待使两者一致
                # 我们只等待估计时间的一定比例，以避免过长等待
                wait_factor = 0.8  # 等待估计播放时间的80%
                wait_time = max(0, (estimated_duration * wait_factor) - elapsed_time)
                
                if wait_time > 0:
                    logging.info(f"段落[{i+1}/{total_segments}]等待中: {wait_time:.2f}秒，"
                               f"以同步文本和语音")
                    
                    # 分段小批次等待，以便能够响应中断
                    wait_increment = 0.1  # 每次等待0.1秒
                    waited = 0
                    while waited < wait_time:
                        if self.check_interrupt():
                            logging.info("等待过程被中断")
                            break
                        
                        sleep_time = min(wait_increment, wait_time - waited)
                        time.sleep(sleep_time)
                        waited += sleep_time
                
                logging.info(f"段落[{i+1}/{total_segments}]处理完成, "
                           f"总时长: {time.time() - segment_start_time:.2f}秒")
                
                # 段落之间添加短暂停顿
                if i < total_segments - 1 and not self.check_interrupt():
                    time.sleep(0.1)
        
            # 所有处理完成
            logging.info("所有文本段和音频处理完成")
            
        except Exception as e:
            logging.error(f"处理音频流时发生异常: {str(e)}", exc_info=True)
    
    def _segment_text_for_tts(self, text: str) -> list:
        """
        将文本切分成适合TTS处理的段落
        
        Args:
            text: 输入的完整文本
            
        Returns:
            文本段落列表
        """
        segments = []
        # 主要分割标记：句号、问号、感叹号、换行符
        major_marks = ['。', '！', '？', '!', '?', '.\n', '!\n', '?\n']
        # 次要分割标记：逗号、分号等
        minor_marks = ['，', ',', '、', '；', ';', ':']
        
        # 先按主要标记分割
        temp_segments = []
        last_pos = 0
        
        # 查找所有主要标记的位置
        for i, char in enumerate(text):
            if any(text[i:i+len(mark)] == mark for mark in major_marks):
                if i + 1 > last_pos:  # 确保有内容添加
                    segment = text[last_pos:i+1].strip()
                    if segment:  # 确保不添加空段
                        temp_segments.append(segment)
                    last_pos = i + 1
        
        # 添加最后一段（如果有）
        if last_pos < len(text):
            segment = text[last_pos:].strip()
            if segment:
                temp_segments.append(segment)
        
        # 对较长的段落进一步按次要标记分割
        for segment in temp_segments:
            if len(segment) <= 50:  # 短段落直接添加
                segments.append(segment)
            else:
                # 按次要标记分割长段落
                sub_last_pos = 0
                found_minor_mark = False
                
                for i, char in enumerate(segment):
                    if i - sub_last_pos > 30:  # 至少30个字符再考虑分割
                        if any(segment[i:i+len(mark)] == mark for mark in minor_marks):
                            sub_segment = segment[sub_last_pos:i+1].strip()
                            if sub_segment:
                                segments.append(sub_segment)
                            sub_last_pos = i + 1
                            found_minor_mark = True
                
                # 添加最后一个子段
                if sub_last_pos < len(segment):
                    sub_segment = segment[sub_last_pos:].strip()
                    if sub_segment:
                        segments.append(sub_segment)
                elif not found_minor_mark:
                    # 如果没找到次要标记，添加整个段落
                    segments.append(segment)
        
        return segments
        
    def reset(self):
        """重置处理器，清除消息历史"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logging.info("语音处理器已重置")
