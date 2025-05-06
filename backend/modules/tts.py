"""
文本到语音模块 - 负责将文本转换为音频流
"""
import logging
import requests
import time
import os
import numpy as np
from typing import Iterator, List, Optional
from openai import OpenAI

def audio_to_float32(audio_data: np.ndarray) -> np.ndarray:
    """将音频数据转换为float32格式，范围在[-1.0, 1.0]之间"""
    if audio_data.dtype == np.int16:
        return audio_data.astype(np.float32) / 32768.0
    return audio_data

class TTSProcessor:
    """TTS处理器，用于处理文本到语音的转换"""
    
    def __init__(self, config, interrupt_checker):
        """
        初始化TTS处理器
        
        Args:
            config: 配置对象
            interrupt_checker: 检查是否中断的函数
        """
        self.config = config
        self.check_interrupt = interrupt_checker
        self.buffer_size = int(config.AUDIO_SAMPLE_RATE * (16//8) * 1 * config.BUFFER_SIZE_FACTOR)
        self.sentence_end_marks = [',','.','，', '!', '?', '。', '！', '？', '\n',':','：']
        
        # 根据配置选择TTS处理函数
        tts_type = config.TTS_TYPE.lower()
        if tts_type == "siliconflow":
            self._process_func = self._process_with_siliconflow_tts
            logging.info("使用硅基TTS服务")
        elif tts_type == "fast":
            self._process_func = self._process_with_fast_tts
            logging.info("使用Fast TTS服务")
        elif tts_type == "kokoro":
            self._process_func = self._process_with_kokoro_tts
            logging.info("使用Kokoro TTS服务")
        else:  # 默认使用本地TTS
            self._process_func = self._process_with_local_tts
            logging.info("使用本地TTS服务")
    
    def process_tts(self, text: str) -> Iterator[bytes]:
        """直接处理单个文本段，返回音频数据"""
        if not text or not text.strip():
            logging.warning("尝试处理空文本")
            return []
            
        return self._process_func(text)
        
    def generate_audio_stream(self, text_iterator: Iterator[str]) -> Iterator[bytes]:
        """
        从文本流生成音频流
        
        Args:
            text_iterator: 文本迭代器
        
        Returns:
            迭代器，每次返回原始音频字节数据
        """
        # 缓存文本，当达到一定长度或遇到标点符号时发送请求
        text_buffer = ""
        # 频繁检查中断的时间控制
        last_interrupt_check = time.time()        
        CHECK_INTERVAL = 0.02  # 每20ms检查一次
        
        try:
            for text_chunk in text_iterator:
                # 频繁检查中断
                current_time = time.time()
                if current_time - last_interrupt_check > CHECK_INTERVAL:
                    if self.check_interrupt():
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
                    for mark in self.sentence_end_marks:
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
                            if self.check_interrupt():
                                return
                            
                            logging.info(f"Sending segment: {segment}")
                            # 使用选定的TTS处理函数处理文本
                            for audio_chunk in self.process_tts(segment):
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
            if text_buffer.strip() and not self.check_interrupt():
                for audio_chunk in self.process_tts(text_buffer):
                    yield audio_chunk
        except Exception as e:
            logging.error(f"音频流生成异常: {str(e)}", exc_info=True)
            if not self.check_interrupt():  # 只在非中断情况下记录异常
                raise
    
    def _process_with_fast_tts(self, text: str) -> Iterator[bytes]:
        """使用Fast TTS处理文本并生成音频"""
        if not text.strip():
            return
            
        logging.info(f"使用Fast TTS处理文本: {text}")
        
        # 准备请求数据
        payload = {
            "text": text,
            "prompt_audio_path": self.config.FAST_TTS_PROMPT_AUDIO,
            "speed": self.config.FAST_TTS_SPEED,
            "cfg_strength": self.config.FAST_TTS_CFG_STRENGTH,
            "nfe_step": self.config.FAST_TTS_NFE_STEP,
            "stream": True
        }
        
        try:
            response = requests.post(
                self.config.FAST_TTS_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=self.buffer_size):
                    if self.check_interrupt():
                        logging.info("Fast TTS流处理被中断")
                        # 主动关闭连接
                        if hasattr(response, 'close'):
                            response.close()
                        return                        
                    if chunk:
                        logging.debug(f"Fast TTS:输出音频块，大小={len(chunk)}")
                        yield chunk
            else:
                logging.error(f"Fast TTS请求失败: {response.status_code} - {response.text}")
        
        except Exception as e:
            logging.error(f"Fast TTS请求异常: {str(e)}")
        finally:
            # 确保连接被关闭
            if response and hasattr(response, 'close'):
                response.close()            
    
    def _process_with_siliconflow_tts(self, text: str) -> Iterator[bytes]:
        """使用硅基TTS处理文本并生成音频"""
        if not text.strip():
            return
            
        logging.info(f"使用硅基TTS处理文本: {text}")
        
        # 准备请求数据：添加情感提示和模型选择
        # 硅基TTS需要格式：情感提示 <|endofprompt|> 实际文本
        input_text = f"{text}"
        # input_text = f"{self.config.SILICONFLOW_TTS_PROMPT}{text}"
        
        payload = {
            "input": input_text,
            "response_format": self.config.SILICONFLOW_TTS_FORMAT,
            "sample_rate": self.config.SILICONFLOW_TTS_SAMPLE_RATE,
            "stream": True,
            "speed": self.config.SILICONFLOW_TTS_SPEED,
            "gain": self.config.SILICONFLOW_TTS_GAIN,
            "model": self.config.SILICONFLOW_TTS_MODEL,
            "voice": self.config.SILICONFLOW_TTS_VOICE
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.SILICONFLOW_TTS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.config.SILICONFLOW_TTS_URL,
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
                    offset += len(chunk)
                    
                    if self.check_interrupt():
                        logging.info("硅基TTS流处理被中断")
                        return
                    
                    if chunk:
                        chunk_length = len(chunk)
                        
                        if previous_chunk is not None:
                            # 将当前 chunk 与上一个奇数大小的 chunk 合并
                            chunk = previous_chunk + chunk
                            previous_chunk = None
                            chunk_length = len(chunk)
                        
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
    
    def _process_with_kokoro_tts(self, text: str) -> Iterator[bytes]:
        """使用Kokoro TTS处理文本并生成音频"""
        if not text.strip():
            return
            
        # logging.info(f"使用Kokoro TTS处理文本: {text}")
        
        # 创建OpenAI客户端，配置基础URL和API密钥
        try:
            client = OpenAI(
                base_url=self.config.KOKORO_TTS_URL,
                api_key=self.config.KOKORO_TTS_API_KEY
            )
            
            # 使用流式响应API创建语音，明确指定PCM格式支持
            # logging.info(f"创建Kokoro TTS流式响应，模型={self.config.KOKORO_TTS_MODEL}, 声音={self.config.KOKORO_TTS_VOICE}, 格式=pcm")
            
            # 使用with语句确保资源被正确释放
            with client.audio.speech.with_streaming_response.create(
                model=self.config.KOKORO_TTS_MODEL,
                voice=self.config.KOKORO_TTS_VOICE,
                input=text,
                response_format="pcm"  # 指定PCM格式输出
            ) as response:
                # 获取流式响应并逐块生成
                chunk_count = 0
                start_time = time.time()
                
                # 使用配置的buffer_size参数
                for chunk in response.iter_bytes(chunk_size=self.buffer_size):
                    if self.check_interrupt():
                        logging.info("Kokoro TTS流处理被中断")
                        return
                    
                    if not chunk:
                        continue
                    
                    chunk_count += 1
                    logging.debug(f"Kokoro TTS: 输出音频块 #{chunk_count}，大小={len(chunk)}字节")
                    yield chunk
                
                end_time = time.time()
                logging.debug(f"Kokoro TTS: 流处理完成，总共输出{chunk_count}个音频块，耗时{end_time-start_time:.2f}秒")
                    
        except Exception as e:
            logging.error(f"Kokoro TTS请求异常: {str(e)}")
            # 记录更详细的错误信息以便调试
            import traceback
            logging.error(traceback.format_exc())
    
    def _process_with_local_tts(self, text: str) -> Iterator[bytes]:
        """使用本地TTS服务处理文本并生成音频"""
        if not text.strip():
            return
            
        logging.info(f"使用本地TTS处理文本: {text}")
        
        # 本地TTS服务器配置
        TTS_SERVER = self.config.TTS_SERVER
        ENDPOINT = self.config.TTS_ENDPOINT
        
        # 准备请求数据
        payload = {
            "text": text,
            "prompt_text": self.config.TTS_PROMPT_TEXT
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
                for chunk in response.iter_content(chunk_size=self.buffer_size):
                    logging.info(f"本地TTS音频块: 大小={len(chunk)}, 类型={type(chunk)}")
                    if self.check_interrupt():
                        logging.info("本地TTS流处理被中断")
                        return
                    if chunk:
                        logging.debug("本地TTS:输出音频块")
                        yield chunk
            else:
                logging.error(f"本地TTS请求失败: {response.status_code} - {response.text}")
        
        except Exception as e:
            logging.error(f"本地TTS请求异常: {str(e)}")
