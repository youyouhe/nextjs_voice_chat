"""
LLM处理模块 - 负责文本生成和处理
"""
import logging
import time
import httpx
from openai import OpenAI
from typing import Iterator, List, Dict, Any, Optional

class LLMProcessor:
    """LLM处理器，用于处理LLM的文本生成"""
    
    def __init__(self, config, messages, interrupt_checker):
        """
        初始化LLM处理器
        
        Args:
            config: 配置对象
            messages: 消息历史列表
            interrupt_checker: 检查是否中断的函数
        """
        self.config = config
        self.messages = messages
        self.check_interrupt = interrupt_checker
        self.full_response = ""
        
        # 创建一个禁用证书验证的传输层
        transport = httpx.HTTPTransport(verify=False)
        # 使用自定义传输层初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.LLM_API_KEY,
            http_client=httpx.Client(transport=transport)
        )
    
    def get_full_response(self) -> str:
        """获取完整的响应文本"""
        return self.full_response
    
    def add_message(self, role: str, content: str):
        """添加消息到历史记录"""
        self.messages.append({"role": role, "content": content})
    
    def reset_messages(self, system_prompt: str):
        """重置消息历史，仅保留系统提示"""
        self.messages = [{"role": "system", "content": system_prompt}]
        logging.info("消息历史已重置")
    
    def generate_text_stream(self) -> Iterator[str]:
        """
        生成文本流
        
        Returns:
            文本流迭代器
        """
        self.full_response = ""
        llm_time = time.time()

        # 创建聊天完成流
        completion_stream = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=self.messages,
            stream=True
        )
        
        try:
            for chunk in completion_stream:
                # 检查是否被中断
                if self.check_interrupt():
                    logging.info("LLM processing interrupted")
                    break            

                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    logging.info(f"LLM chunk: {content}")
                    self.full_response += content
                    yield content
                
                if chunk.choices[0].finish_reason == "stop":
                    break
        except Exception as e:
            logging.error(f"LLM text generation error: {str(e)}", exc_info=True)
        finally:
            # 在生成完成后记录总响应和时间
            if self.full_response:
                logging.info(f"LLM full response: {self.full_response}")
                logging.info(f"LLM took {time.time() - llm_time:.2f} seconds")
                
                # 将助手回复添加到消息历史
                self.add_message("assistant", self.full_response)
        
    def process_one_time(self, user_input: str) -> str:
        """
        处理单次LLM调用，非流式
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            LLM的完整响应
        """
        # 添加用户消息
        self.add_message("user", user_input)
        
        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=self.messages,
        )
        
        content = response.choices[0].message.content
        self.full_response = content
        
        # 添加助手回复
        self.add_message("assistant", content)
        
        return content

    def get_text_synchronizer(self):
        """
        返回一个简化的文本同步器，用于处理文本的分段同步显示
        
        Returns:
            TextSynchronizer 实例
        """
        return TextSynchronizer(self.full_response)

class TextSynchronizer:
    """文本同步器，用于控制文本显示的速度与语音同步"""
    
    def __init__(self, full_text: str):
        """
        初始化文本同步器
        
        Args:
            full_text: 完整的文本内容
        """
        self.full_text = full_text
        self.last_sent_position = 0
        self.end_marks = ['.', ',', '!', '?', '。', '，', '！', '？', '\n', ':', '：']
    
    def get_next_segment(self, max_length: int = 50) -> str:
        """
        获取下一个文本段
        
        Args:
            max_length: 最大段落长度
            
        Returns:
            下一个文本段，如果没有更多文本则返回空字符串
        """
        if self.last_sent_position >= len(self.full_text):
            return ""
        
        # 计算剩余文本
        remaining_text = self.full_text[self.last_sent_position:]
        
        # 查找下一个句子结束标记
        next_end_pos = max_length
        for mark in self.end_marks:
            pos = remaining_text.find(mark, 0, max_length)
            if pos != -1:
                next_end_pos = pos + 1  # 包括标记
                break
        
        # 如果超出剩余文本长度，则取全部
        if next_end_pos > len(remaining_text):
            next_end_pos = len(remaining_text)
        
        # 提取文本段
        segment = remaining_text[:next_end_pos]
        
        # 更新已发送位置
        self.last_sent_position += len(segment)
        
        return segment
    
    def is_complete(self) -> bool:
        """
        检查是否已经发送完所有文本
        
        Returns:
            如果所有文本已发送则返回True
        """
        return self.last_sent_position >= len(self.full_text)
    
    def get_progress(self) -> float:
        """
        获取发送进度
        
        Returns:
            以百分比表示的进度 (0.0-1.0)
        """
        if not self.full_text:
            return 1.0
        return self.last_sent_position / len(self.full_text)
