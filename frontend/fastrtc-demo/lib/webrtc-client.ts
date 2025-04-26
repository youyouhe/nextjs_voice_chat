interface WebRTCClientOptions {
    onConnected?: () => void;
    onDisconnected?: () => void;
    onMessage?: (message: any) => void;
    onAudioStream?: (stream: MediaStream) => void;
    onAudioLevel?: (level: number) => void;
}

export class WebRTCClient {
    private peerConnection: RTCPeerConnection | null = null;
    private mediaStream: MediaStream | null = null;
    private dataChannel: RTCDataChannel | null = null;
    private options: WebRTCClientOptions;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private dataArray: Uint8Array | null = null;
    private animationFrameId: number | null = null;
    private connectionId: string | null = null; // 保存连接ID

    constructor(options: WebRTCClientOptions = {}) {
        this.options = options;
    }

    async connect() {
        try {
            this.peerConnection = new RTCPeerConnection();
            
            // Get user media
            try {
                this.mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: true
                });
            } catch (mediaError: any) {
                console.error('Media error:', mediaError);
                if (mediaError.name === 'NotAllowedError') {
                    throw new Error('Microphone access denied. Please allow microphone access and try again.');
                } else if (mediaError.name === 'NotFoundError') {
                    throw new Error('No microphone detected. Please connect a microphone and try again.');
                } else {
                    throw mediaError;
                }
            }
            
            this.setupAudioAnalysis();
            
            this.mediaStream.getTracks().forEach(track => {
                if (this.peerConnection) {
                    this.peerConnection.addTrack(track, this.mediaStream!);
                }
            });
            
            if (this.peerConnection) {
                this.peerConnection.addEventListener('track', (event) => {
                    if (this.options.onAudioStream) {
                        this.options.onAudioStream(event.streams[0]);
                    }
                });
            } else {
                console.warn('Peer connection is null, cannot add track event listener');
            }
            
            if (!this.peerConnection) {
                console.error('Peer connection is null, cannot create data channel');
                throw new Error('Peer connection is null, cannot create data channel');
            }
            
            this.dataChannel = this.peerConnection.createDataChannel('text');
            
            if (!this.dataChannel) {
                console.error('Data channel creation failed');
                throw new Error('Data channel creation failed');
            }
            
            this.dataChannel.addEventListener('message', (event) => {
                try {
                    const message = JSON.parse(event.data);
                    // console.log('Received message:', message);
                    
                    // 处理特定类型的消息
                    const eventJson = message;
                    
                    if (eventJson.type === "error") {
                        console.log('Error message received:', eventJson.message);
                        // 可以添加错误处理逻辑
                    } else if (eventJson.type === "send_input") {
                        console.log('Send input event received:', eventJson);
                        // 这里可以添加相应的处理逻辑
                    } else if (eventJson.type === "log") {
                        console.log('Log event received:', eventJson.data);
                        if (eventJson.data === "pause_detected") {
                            console.log('Pause detected in speech');
                        } else if (eventJson.data === "response_starting") {
                            console.log('Response starting');
                        }
                    } else if(eventJson.type="llm_chunk"){
                        // console.log('llm_chunk received:',eventJson.data);
                    }
                    
                    if (this.options.onMessage) {
                        this.options.onMessage(message);
                    }
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            });
            
            // Create and send offer
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            
            // 使用相对路径或获取当前窗口的域名
            const serverURL = process.env.NEXT_PUBLIC_BACKEND_URL || 'https://localhost:8000';
            console.log(`连接到WebRTC服务器: ${serverURL}/webrtc/offer`);
            
            // 生成连接ID
            this.connectionId = Math.random().toString(36).substring(7);
            
            // 发送WebRTC请求
            const response = await fetch(`${serverURL}/webrtc/offer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                mode: 'cors', // Explicitly set CORS mode
                credentials: 'same-origin',
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    webrtc_id: this.connectionId
                })
            });
            
            const serverResponse = await response.json();
            console.log('Received server response:', serverResponse); // Add this line
            await this.peerConnection.setRemoteDescription(serverResponse);
            
            if (this.options.onConnected) {
                this.options.onConnected();
            }
        } catch (error) {
            console.error('Error connecting:', error);
            this.disconnect();
            throw error;
        }
    }

    private setupAudioAnalysis() {
        if (!this.mediaStream) return;
        
        try {
            this.audioContext = new AudioContext();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.analyser);
            
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
            this.startAnalysis();
        } catch (error) {
            console.error('Error setting up audio analysis:', error);
        }
    }

    private startAnalysis() {
        if (!this.analyser || !this.dataArray || !this.options.onAudioLevel) return;
        
        // Add throttling to prevent too many updates
        let lastUpdateTime = 0;
        const throttleInterval = 100; // Only update every 100ms
        
        const analyze = () => {
            this.analyser!.getByteFrequencyData(this.dataArray!);
            
            const currentTime = Date.now();
            // Only update if enough time has passed since last update
            if (currentTime - lastUpdateTime > throttleInterval) {
                // Calculate average volume level (0-1)
                let sum = 0;
                for (let i = 0; i < this.dataArray!.length; i++) {
                    sum += this.dataArray![i];
                }
                const average = sum / this.dataArray!.length / 255;
                
                this.options.onAudioLevel!(average);
                lastUpdateTime = currentTime;
            }
            
            this.animationFrameId = requestAnimationFrame(analyze);
        };
        
        this.animationFrameId = requestAnimationFrame(analyze);
    }

    private stopAnalysis() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.analyser = null;
        this.dataArray = null;
    }

    disconnect() {
        this.cleanupResources();
        
        // 调用断开连接的回调
        if (this.options.onDisconnected) {
            this.options.onDisconnected();
        }
    }
    
    /**
     * 清理资源但不触发回调，用于组件卸载时避免状态更新
     */
    cleanupWithoutCallbacks() {
        this.cleanupResources();
        // 不调用任何回调，以避免在组件卸载过程中触发状态更新
    }
    
    /**
     * 内部方法：清理所有资源
     */
    private cleanupResources() {
        this.stopAnalysis();
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        
        this.dataChannel = null;
        this.connectionId = null; // 重置连接ID
    }
    
    /**
     * 获取当前WebRTC连接ID
     * @returns 连接ID，如果未连接则返回null
     */
    getConnectionId(): string | null {
        return this.connectionId;
    }
}
