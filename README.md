# FastRTC Voice Chat Application

This is a real-time voice chat application built with FastAPI and FastRTC (developed by [rohanprichard](https://github.com/rohanprichard)). The project combines a Next.js frontend with a FastAPI backend to deliver seamless integration of real-time voice interaction, speech synthesis (TTS), speech recognition (ASR), and large language models (LLM).

## Core Features

- **Real-time Voice Interaction**: Low-latency two-way voice communication using FastRTC
- **Automatic Speech Recognition (ASR)**: Support for custom ASR services to convert user speech to text
- **Large Language Model (LLM)**: Integration with OpenAI API or compatible alternatives for text processing and intelligent responses
- **Text-to-Speech (TTS)**: Multiple TTS service options:
  - Local TTS service
  - SiliconFlow TTS service
  - Fast TTS service
- **Voice Activity Detection (VAD)**: Precise detection of speech activity using the Silero VAD model
- **Advanced Audio Processing**: Rich configuration options for audio parameters
- **Text and Speech Synchronization**: Precise synchronization between text display and speech playback

## System Architecture

### Backend Architecture

The backend adopts a modular design with these main components:

1. **Server Modules**:
   - `server.py`: Main server entry point, handling WebRTC connections and request routing
   - `server_async.py`: Asynchronous processing version for more efficient concurrent handling

2. **Voice Processing Modules**:
   - `modules/asr.py`: Speech recognition module for converting audio to text
   - `modules/tts.py`: Text-to-speech module supporting various TTS services
   - `modules/llm.py`: Large language model interaction module
   - `modules/async_speech_processor.py`: Asynchronous speech processor coordinating parallel execution of ASR, LLM, and TTS
   - `modules/speech_processor.py`: Synchronous version of the speech processor

3. **Configuration System**:
   - `config.py`: Unified configuration management for environment variables and default values
   - `env.py`: Helper functions for environment variable processing

### Voice Processing Flow

1. User speech is transmitted to the backend via WebRTC
2. Voice Activity Detection (VAD) identifies valid speech segments
3. Speech Recognition (ASR) converts speech to text
4. Large Language Model (LLM) processes the text and generates responses
5. Text-to-Speech (TTS) converts the response to speech
6. Speech is transmitted back to the user in real-time through WebRTC
7. Text is simultaneously displayed in real-time on the frontend interface

### Frontend Architecture

The frontend is based on the Next.js framework and is responsible for:
- User interface rendering
- WebRTC connection management
- Voice input control
- Real-time LLM text display
- Audio level visualization

## Configuration Details

The project provides extensive configuration options through environment variables or a `.env` file:

### Server Configuration
- `SERVER_HOST`: Server host address
- `SERVER_PORT`: Server port

### Model Configuration
- `LLM_MODEL`: Large language model name
- `LLM_API_KEY`: API key
- `OPENAI_BASE_URL`: API base URL

### Speech Recognition Configuration
- `CUSTOM_ASR_URL`: Custom ASR service URL

### Speech Synthesis Configuration
- `TTS_TYPE`: Choose TTS service type ("local", "siliconflow", or "fast")
- Various parameters specific to each TTS service

### Audio Processing Configuration
- `AUDIO_SAMPLE_RATE`: Audio sampling rate
- `BUFFER_SIZE_FACTOR`: Buffer size factor

### Voice Activity Detection Configuration
- `VAD_THRESHOLD`: Voice detection threshold
- `VAD_MIN_SPEECH_DURATION_MS`: Minimum speech duration
- `VAD_MIN_SILENCE_DURATION_MS`: Minimum silence duration
- Other VAD-related parameters

## Installation and Running

1. Copy the `.env.example` file to `.env` and set the necessary API keys and configuration
2. Create a Python virtual environment and install dependencies:
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

3. Run the backend service:
    ```bash
    ./run.sh
    ```

4. In another terminal, start the frontend:
    ```bash
    cd frontend/fastrtc-demo
    npm install
    npm run dev
    ```

5. Open your browser, click the microphone icon to start chatting
6. Click the trash bin icon in the bottom right to reset the conversation

## Tuning Recommendations

### Audio Parameter Adjustments

#### AlgoOptions Parameters

- **audio_chunk_duration**: Duration of audio chunks in seconds
- **started_talking_threshold**: Threshold for detecting the start of speech
- **speech_threshold**: Threshold for speech detection

#### SileroVadOptions Parameters

- **threshold**: Speech probability threshold (0.0-1.0)
- **min_speech_duration_ms**: Minimum speech duration in milliseconds
- **min_silence_duration_ms**: Minimum silence duration in milliseconds
- **speech_pad_ms**: Speech padding time in milliseconds
- **max_speech_duration_s**: Maximum speech duration in seconds

### Common Adjustments

- If the AI interrupts too early:
  - Increase `min_silence_duration_ms`
  - Increase `speech_threshold`
  - Increase `speech_pad_ms`

- If the AI response is slow:
  - Decrease `min_silence_duration_ms`
  - Decrease `speech_threshold`

- If the system fails to detect some speech:
  - Lower the `threshold` value
  - Lower the `started_talking_threshold`

## Development Notes

- The `.env` file contains sensitive information and is added to `.gitignore`
- You can modify the system prompt in `backend/server.py` according to your needs
- The project uses self-signed SSL certificates for local development; these should be replaced with valid certificates in production

## Technical Acknowledgements

- FastRTC: Providing a Python implementation of WebRTC
- FastAPI: Building high-performance API servers
- Next.js: Frontend framework
- Silero VAD: Voice activity detection model
- UI components: Thanks to Shadcn, Aceternity UI, and Kokonut UI for providing UI components
