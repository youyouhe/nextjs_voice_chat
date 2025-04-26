# 语音文本同步机制说明

## 问题描述

在实现语音聊天系统时，我们遇到一个典型问题：**文本显示速度远快于语音播放速度**。这会导致用户体验不佳，因为：
1. 用户看到的文本内容远超过当前正在播放的语音内容
2. 当语音仍在播放前面的内容时，用户已经看到后面的文本

## 根本原因

这个问题的根本原因在于：
- TTS生成的音频数据（例如5秒长度的语音）在网络传输中几乎是瞬时完成的（可能只需0.01秒）
- 系统没有考虑音频的实际播放时长，而是在发送完音频数据后立即继续下一段文本
- 缺少一种机制来同步文本显示和音频播放的进度

## 解决方案：基于音频时长的同步控制

我们实现了一种基于音频时长估算的同步控制机制，核心思路是：

1. **计算音频播放时长**：
   ```python
   total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
   bytes_per_second = self.config.AUDIO_SAMPLE_RATE * 2  # 16位音频，每采样2字节
   estimated_duration = total_audio_bytes / bytes_per_second
   ```

2. **添加等待时间以同步文本和语音**：
   ```python
   # 计算已经播放的时间（实际上是发送时间）
   elapsed_time = time.time() - segment_start_time
   
   # 如果实际播放时间小于估计时间，进行等待使两者一致
   wait_factor = 0.8  # 等待估计播放时间的80%
   wait_time = max(0, (estimated_duration * wait_factor) - elapsed_time)
   ```

3. **响应式等待，支持中断**：
   ```python
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
   ```

## 两种实现方案

### 1. 同步版本 (speech_processor.py)

同步版本使用标准的Python `time.sleep()` 来实现等待：

```python
# 段落处理流程
for i, segment in enumerate(text_segments):
    # 发送文本段
    yield AdditionalOutputs(llm_data)
    
    # 播放音频
    for audio_chunk in audio_chunks:
        yield (self.config.AUDIO_SAMPLE_RATE, audio_array)
    
    # 等待音频播放完成
    wait_time = max(0, (estimated_duration * wait_factor) - elapsed_time)
    # 循环小批次等待，支持中断
    while waited < wait_time:
        # ...小批次等待代码...
```

### 2. 异步版本 (async_speech_processor.py)

异步版本使用 `await asyncio.sleep()` 来实现非阻塞等待：

```python
async def process_tts_for_segment(self, tts_processor, segment: str):
    # 计算播放时间
    total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
    bytes_per_second = self.config.AUDIO_SAMPLE_RATE * 2
    estimated_duration = total_audio_bytes / bytes_per_second
    
    # 播放音频
    for audio_chunk in audio_chunks:
        yield (self.config.AUDIO_SAMPLE_RATE, audio_array)
        await asyncio.sleep(0.01)  # 让出控制权
    
    # 等待播放完成
    wait_time = max(0, (estimated_duration * wait_factor) - elapsed_time)
    while waited_time < wait_time:
        # ...异步小批次等待代码...
        await asyncio.sleep(sleep_time)
```

## 优化参数

1. **wait_factor = 0.8**：我们只等待估计播放时间的80%
   - 避免过长等待，因为估算可能有误差
   - 提供更自然的节奏感，句子之间有短暂重叠

2. **wait_increment = 0.1**：每次等待0.1秒
   - 足够小以实现快速响应中断
   - 足够大以减少CPU负担

## 使用建议

同步版本和异步版本在功能上是等效的，选择取决于项目需求：

- **简单项目**：使用同步版本 (server_new.py)
- **复杂项目**：使用异步版本 (server_async.py)，特别是需要处理大量并发请求时

## 日志监控

添加了详细的日志记录，可以监控同步过程：

```
INFO:root:段落[1/6]音频长度: 8192字节, 估计播放时间: 0.17秒
INFO:root:段落[1/6]等待中: 0.11秒，以同步文本和语音
INFO:root:段落[1/6]处理完成, 总时长: 0.17秒
```

这些日志可以帮助诊断和优化同步参数。
