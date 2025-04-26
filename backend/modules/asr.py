"""
语音识别模块 - 负责将音频转换为文本
"""
import logging
import io
import requests
from fastrtc.utils import audio_to_bytes

def process_asr(audio, config):
    """
    处理语音识别，将音频转为文本
    
    Args:
        audio: 输入的音频数据
        config: 配置对象
    
    Returns:
        str: 识别出的文本，如果识别失败则返回空字符串
    """
    logging.info("Performing STT")
    stt_start_time = 0
    
    try:
        audio_bytes = audio_to_bytes(audio)
        # 检查音频是否为空
        if not audio_bytes:
            logging.warning("audio_to_bytes returned empty bytes, skipping ASR.")
            return ""

        audio_bytes_io = io.BytesIO(audio_bytes)
        CUSTOM_ASR_URL = config.CUSTOM_ASR_URL
        
        # 准备请求参数
        files = {'files': ('audio.mp3', audio_bytes_io, 'audio/mp3')}
        data = {'keys': 'string', 'lang': 'auto'}
        headers = {'accept': 'application/json'}

        logging.info(f"Sending {len(audio_bytes)} bytes to ASR: {CUSTOM_ASR_URL}")
        response = requests.post(
            CUSTOM_ASR_URL, 
            files=files, 
            data=data, 
            headers=headers, 
            timeout=10
        )
        response.raise_for_status()

        asr_result = response.json()
        logging.info(f"Custom ASR raw response: {asr_result}")

        # 解析ASR响应
        prompt = ''
        if isinstance(asr_result, dict) and 'result' in asr_result:
            result_list = asr_result['result']
            if isinstance(result_list, list) and len(result_list) > 0:
                first_result = result_list[0]
                if isinstance(first_result, dict) and 'text' in first_result:
                    prompt = first_result.get('text', '')
                else:
                    logging.warning(f"First item in 'result' list is not a dict or lacks 'text' key: {first_result}")
            else:
                logging.warning(f"'result' key is not a non-empty list: {result_list}")
        else:
            logging.warning(f"ASR response is not a dict or lacks 'result' key: {asr_result}")

        if not prompt:
            logging.info("ASR returned empty string or failed")
            return ""
            
        logging.info(f"ASR response: {prompt}")
        return prompt

    except requests.exceptions.Timeout:
        logging.error("Custom ASR service timed out.")
        return ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling custom ASR service: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error processing custom ASR: {e}", exc_info=True)
        return ""
