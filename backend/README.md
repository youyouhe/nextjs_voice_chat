# 语音聊天服务器重构说明

## 项目结构

重构后的项目采用模块化设计，将之前的单文件结构拆分为多个专注于特定功能的模块：

```
backend/
├── config.py              # 配置文件
├── server.py              # 原始服务器文件(保留)
├── server_new.py          # 重构后的主服务器文件
└── modules/               # 功能模块目录
    ├── asr.py             # 语音识别模块
    ├── llm.py             # LLM处理模块
    ├── tts.py             # 文本到语音模块
    └── speech_processor.py # 语音处理集成模块
```

## 模块说明

### 1. ASR模块 (asr.py)

负责语音识别(Speech-to-Text)功能：
- 接收音频输入并转换为文本
- 处理ASR服务的请求和响应
- 错误处理和日志记录

### 2. LLM模块 (llm.py)

负责语言模型处理：
- 与OpenAI API交互
- 管理对话历史
- 提供文本生成流
- 添加文本同步器用于控制前端显示节奏

### 3. TTS模块 (tts.py)

负责文本到语音转换：
- 支持多种TTS服务(Fast TTS, 硅基TTS, 本地TTS)
- 文本分段处理
- 音频流生成与处理

### 4. 语音处理器 (speech_processor.py)

整合以上三个模块：
- 协调ASR-LLM-TTS完整流程
- 实现文本与语音的同步
- 管理中断和状态

### 5. 主服务器 (server_new.py)

负责HTTP和WebRTC通信：
- FastAPI服务器配置
- WebRTC处理
- 端点和路由定义

## 重构优势

1. **可维护性**：拆分后的代码更易于理解和维护
2. **可扩展性**：可以独立升级或替换各个模块
3. **可读性**：每个模块专注于一个功能，逻辑清晰
4. **可测试性**：可以单独测试每个模块
5. **解耦**：模块间通过清晰的接口交互，降低耦合

## 使用方法

重构后的服务使用方法与原服务相同，但运行新服务器文件：

```bash
python -m backend.server_new
```

或在开发和测试阶段，可以直接运行：

```bash
uvicorn backend.server_new:app --host 0.0.0.0 --port 8000 --ssl-keyfile backend/localhost+2-key.pem --ssl-certfile backend/localhost+2.pem
```

## 文本与语音同步机制

新版本实现了改进的文本-语音同步机制：

1. LLM生成完整文本后，创建文本同步器
2. TTS处理完整文本，生成音频流
3. 每生成一个音频块，发送对应的文本段
4. 文本段按自然语言边界(句子、短语等)分割
5. 前端收到的文本与当前播放的语音保持同步

这种机制确保了用户看到的文本与听到的语音完全匹配，提升了用户体验。
