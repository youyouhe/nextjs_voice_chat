"use client"

import { useState, useEffect, useRef, useCallback } from "react";
import { BackgroundCircles } from "@/components/ui/background-circles";
import { AIVoiceInput } from "@/components/ui/ai-voice-input";
import { WebRTCClient } from "@/lib/webrtc-client";
import { SubtitleWrapper } from "@/components/ui/subtitle-wrapper";

interface LlmChunk {
  type: string;
  content: string;
}

export function BackgroundCircleProvider() {
    const [currentVariant, setCurrentVariant] = 
        useState<keyof typeof COLOR_VARIANTS>("octonary");
    const [isConnected, setIsConnected] = useState(false);
    const [audioLevel, setAudioLevel] = useState(0);
    const audioRef = useRef<HTMLAudioElement>(null);
    // 用于存储收到的LLM文本块
    const [llmChunks, setLlmChunks] = useState<string[]>([]);
    // 当前正在显示的文本
    const [displayText, setDisplayText] = useState("");
    // 打字机效果计时器与状态
    const typingTimerRef = useRef<NodeJS.Timeout | null>(null);
    const currentIndexRef = useRef<number>(0);
    const fullTextRef = useRef<string>("");
    // 存储WebRTC连接ID
    const [webrtcId, setWebrtcId] = useState<string | null>(null);
    // SSE事件源
    const eventSourceRef = useRef<EventSource | null>(null);
    // 使用useRef存储WebRTCClient实例，避免不必要的重新渲染
    const webrtcClientRef = useRef<WebRTCClient | null>(null);

    // Memoize callbacks to prevent recreation on each render
    const handleConnected = useCallback(() => {
        console.log("WebRTC connected");
        setIsConnected(true);
        
        // 获取WebRTC ID并存储
        const client = webrtcClientRef.current;
        if (client) {
            const id = client.getConnectionId();
            if (id) {
                console.log("Setting webrtc ID:", id);
                setWebrtcId(id);
                // 存储到会话存储
                sessionStorage.setItem('webrtc_id', id);
            }
        }
    }, []);
    
    // 注意：不要在这里直接操作eventSourceRef，避免循环依赖
    const handleDisconnected = useCallback(() => {
        // console.log("WebRTC disconnected");
        setIsConnected(false);
    }, []);
    
    const handleAudioStream = useCallback((stream: MediaStream) => {
        if (audioRef.current) {
            audioRef.current.srcObject = stream;
        }
    }, []);
    
    const handleAudioLevel = useCallback((level: number) => {
        // Apply some smoothing to the audio level
        setAudioLevel(prev => {
            const smoothedLevel = prev * 0.7 + level * 0.3;
            // Only update if the change is significant
            if (Math.abs(smoothedLevel - prev) > 0.001) {
                return smoothedLevel;
            }
            return prev;
        });
    }, []);

    // 打字机效果函数，定义在组件外部避免重复创建
    const typeWriter = useCallback(() => {
        if (currentIndexRef.current < fullTextRef.current.length) {
            // 使用函数式更新，避免依赖最新的state
            setDisplayText(fullTextRef.current.substring(0, currentIndexRef.current + 1));
            currentIndexRef.current++;
            // 设置下一个字符的延迟，速度随机变化以模拟真实打字感觉
            const delay = Math.random() * 30 + 20; // 20-50ms之间的随机延迟
            typingTimerRef.current = setTimeout(typeWriter, delay);
        }
    }, []);
    
    // 打字机效果实现
    useEffect(() => {
        // 清除之前的计时器
        if (typingTimerRef.current) {
            clearTimeout(typingTimerRef.current);
        }
        
        // 如果有新的文本块，启动打字机效果
        if (llmChunks.length > 0) {
            // 保存完整文本到ref
            fullTextRef.current = llmChunks[llmChunks.length - 1];
            // 重置索引
            currentIndexRef.current = 0;
            // 重置显示文本
            setDisplayText("");
            
            // 开始打字效果
            typeWriter();
        } else {
            // 没有文本时清空显示
            setDisplayText("");
            fullTextRef.current = "";
            currentIndexRef.current = 0;
        }
        
        // 组件卸载时清除计时器
        return () => {
            if (typingTimerRef.current) {
                clearTimeout(typingTimerRef.current);
            }
        };
    }, [llmChunks, typeWriter]); // 依赖于llmChunks的变化和typeWriter函数

    // 使用SSE设置LLM文本块
    useEffect(() => {
        // 防止重复创建SSE连接
        const shouldConnect = isConnected && webrtcId && !eventSourceRef.current;
        const shouldDisconnect = !isConnected && eventSourceRef.current;
        
        if (shouldConnect) {
            // 创建服务器端事件源
            console.log("创建SSE连接, webrtcId:", webrtcId);
            const serverURL = process.env.NEXT_PUBLIC_BACKEND_URL || window.location.origin;
            const eventSource = new EventSource(`${serverURL}/llm_chunks?webrtc_id=${webrtcId}`);
            eventSourceRef.current = eventSource;
            
            // 处理事件
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data) as LlmChunk;
                    console.log("收到LLM文本块 (SSE):", data);
                    
                    if (data.type === "llm_chunk") {
                        setLlmChunks(prev => [...prev, data.content]);
                    }
                } catch (error) {
                    console.error("解析SSE消息失败:", error);
                }
            };
            
            // 处理错误
            eventSource.onerror = (error) => {
                console.error("SSE连接错误:", error);
            };
        }
        
        if (shouldDisconnect && eventSourceRef.current) {
            console.log("关闭SSE连接");
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        
        // 清理函数
        return () => {
            if (eventSourceRef.current) {
                console.log("清理SSE连接");
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        };
    }, [isConnected, webrtcId]);

    // Get all available variants
    const variants = Object.keys(
        COLOR_VARIANTS
    ) as (keyof typeof COLOR_VARIANTS)[];

    // Function to change to the next color variant
    const changeVariant = () => {
        const currentIndex = variants.indexOf(currentVariant);
        const nextVariant = variants[(currentIndex + 1) % variants.length];
        setCurrentVariant(nextVariant);
    };

    // 初始化WebRTC客户端 - 避免在组件卸载时调用disconnect导致状态变化
    useEffect(() => {
        // 创建不会在组件卸载时触发状态更新的客户端
        const clientOptions = {
            onConnected: handleConnected, 
            onAudioStream: handleAudioStream,
            onAudioLevel: handleAudioLevel,
            onDisconnected: handleDisconnected // 在正常使用时可以使用断开连接回调
        };
        
        // 使用useRef存储客户端，而不是用useState
        const client = new WebRTCClient(clientOptions);
        webrtcClientRef.current = client;

        return () => {
            // 在卸载时不更新状态，直接调用底层方法清理资源
            if (webrtcClientRef.current) {
                webrtcClientRef.current.cleanupWithoutCallbacks();
            }
        };
    }, [handleConnected, handleDisconnected, handleAudioStream, handleAudioLevel]);

    const handleStart = () => {
        // 连接时清空文本块
        setLlmChunks([]);
        if (webrtcClientRef.current) {
            webrtcClientRef.current.connect();
        }
    };

    const handleStop = () => {
        if (webrtcClientRef.current) {
            webrtcClientRef.current.disconnect();
        }
    };

    // 重置LLM文本块
    const handleResetText = () => {
        setLlmChunks([]);
    };

    return (
        <div 
            className="relative w-full h-full"
            onClick={changeVariant} // Add click handler to change color
        >
            <BackgroundCircles 
                variant={currentVariant} 
                audioLevel={audioLevel}
                isActive={isConnected}
            />
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <AIVoiceInput 
                    onStart={handleStart}
                    onStop={handleStop}
                    isConnected={isConnected}
                />
            </div>
            
            {/* 使用Subtitle组件显示LLM文本块 */}
            {displayText && <SubtitleWrapper text={displayText} />}
            
            {/* 连接ID显示 - 底部 */}
            {webrtcId && isConnected && (
                <div className="absolute bottom-4 right-4 z-10">
                    <span className="bg-black/30 text-white/40 px-2 py-1 rounded-full text-xs">
                        ID: {webrtcId.substring(0, 6)}
                    </span>
                </div>
            )}
            
            {/* 重置按钮 */}
            {isConnected && (
                <div className="absolute top-4 right-4 z-10">
                    <button 
                        onClick={(e) => {
                            e.stopPropagation(); // 防止触发背景色变化
                            handleResetText();
                        }}
                        className="bg-black/30 hover:bg-black/40 px-3 py-1 rounded-full text-white/80 text-xs transition-colors"
                    >
                        清空文本
                    </button>
                </div>
            )}
            <audio ref={audioRef} autoPlay hidden />
        </div>
    );
}

export default { BackgroundCircleProvider }

const COLOR_VARIANTS = {
    primary: {
        border: [
            "border-emerald-500/60",
            "border-cyan-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-emerald-500/30",
    },
    secondary: {
        border: [
            "border-violet-500/60",
            "border-fuchsia-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-violet-500/30",
    },
    senary: {
        border: [
            "border-blue-500/60",
            "border-sky-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-blue-500/30",
    }, // blue
    octonary: {
        border: [
            "border-red-500/60",
            "border-rose-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-red-500/30",
    },
} as const;
