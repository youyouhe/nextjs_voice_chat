"use client";

import { useEffect, useState, useRef } from "react";
import { useSubtitle } from "@/context/subtitle-context";

interface SyncInfo {
  is_synced: boolean;
  segment_index: number;
  total_segments: number;
  progress: number;
}

interface LlmChunk {
  type: string;
  content: string;
  sync_info?: SyncInfo;
}

export function LlmTextReceiver() {
  const { updateSubtitle } = useSubtitle();
  const [messages, setMessages] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [webrtcId, setWebrtcId] = useState<string | null>(null);
  const [currentSegment, setCurrentSegment] = useState<number>(0);
  const [totalSegments, setTotalSegments] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  
  // 获取webrtcId的函数 - 可以从URL参数、全局状态或其他方式获取
  const getWebrtcId = () => {
    // 这里暂时使用一个简单的方法来获取或生成webrtcId
    // 在实际应用中，这应该与实际的WebRTC连接ID同步
    const storedId = sessionStorage.getItem('webrtc_id');
    if (storedId) return storedId;
    
    // 如果没有存储的ID，尝试从URL获取
    const urlParams = new URLSearchParams(window.location.search);
    const urlId = urlParams.get('webrtc_id');
    if (urlId) {
      sessionStorage.setItem('webrtc_id', urlId);
      return urlId;
    }
    
    // 如果都没有，生成一个新的（这在实际应用中可能不适用）
    const newId = Math.random().toString(36).substring(7);
    sessionStorage.setItem('webrtc_id', newId);
    return newId;
  };
  
  // 建立SSE连接获取LLM文本块
  useEffect(() => {
    const id = getWebrtcId();
    if (!id) return;
    
    setWebrtcId(id);
    
    // 创建SSE连接
    const serverURL = process.env.NEXT_PUBLIC_BACKEND_URL || window.location.origin;
    const eventSource = new EventSource(`${serverURL}/llm_chunks?webrtc_id=${id}`);
    eventSourceRef.current = eventSource;
    
    // 连接状态变化
    eventSource.onopen = () => {
      console.log("SSE连接已建立");
      setIsConnected(true);
    };
    
    // 接收消息
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as LlmChunk;
        console.log("收到LLM文本块:", data);
        
        if (data.type === "llm_chunk") {
          // 更新字幕显示 - 关键改进：只显示当前段落
          if (data.sync_info) {
            // 使用同步信息，只显示当前段落
            updateSubtitle(data.content);
            
            // 更新段落信息
            setCurrentSegment(data.sync_info.segment_index + 1);
            setTotalSegments(data.sync_info.total_segments);
            setProgress(data.sync_info.progress || 0);
            
            // 为了保存完整历史，仍然保存所有文本
            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[data.sync_info?.segment_index || 0] = data.content;
              return newMessages;
            });
          } else {
            // 没有同步信息的情况下保持原有逻辑
            updateSubtitle(data.content);
            setMessages(prev => [...prev, data.content]);
          }
        }
      } catch (error) {
        console.error("解析SSE消息失败:", error);
      }
    };
    
    // 错误处理
    eventSource.onerror = (error) => {
      console.error("SSE连接错误:", error);
      setIsConnected(false);
      
      // 尝试重新连接
      setTimeout(() => {
        eventSourceRef.current?.close();
        setWebrtcId(null);
        // 下次useEffect运行时会重新连接
      }, 5000);
    };
    
    return () => {
      // 组件卸载时关闭SSE连接
      eventSource.close();
      eventSourceRef.current = null;
    };
  }, [webrtcId, updateSubtitle]);
  
  // 重置文本块
  const handleReset = () => {
    setMessages([]);
    updateSubtitle('');
    setCurrentSegment(0);
    setTotalSegments(0);
    setProgress(0);
  };
  
  return (
    <div className="w-full max-w-2xl mx-auto p-4">
      <div className="bg-white/10 dark:bg-black/20 rounded-lg p-4 shadow-md">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-medium">LLM文本接收器</h3>
          <button 
            onClick={handleReset}
            className="text-xs bg-gray-500/20 hover:bg-gray-500/30 py-1 px-2 rounded"
          >
            重置
          </button>
        </div>
        
        <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          <div className="flex justify-between">
            <span>
              状态: {isConnected ? (
                <span className="text-green-500">已连接</span>
              ) : (
                <span className="text-red-500">未连接</span>
              )}
              {webrtcId && <span className="ml-2 opacity-50">ID: {webrtcId.substring(0, 4)}...</span>}
            </span>
            
            {totalSegments > 0 && (
              <span>段落: {currentSegment}/{totalSegments} ({progress}%)</span>
            )}
          </div>
          
          {/* 进度条显示 */}
          {totalSegments > 0 && (
            <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2 dark:bg-gray-700">
              <div 
                className="bg-blue-600 h-1.5 rounded-full transition-all duration-300" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          )}
        </div>
        
        <div className="bg-black/5 dark:bg-white/5 rounded p-3 h-[200px] overflow-y-auto">
          {messages.length > 0 ? (
            <div className="whitespace-pre-wrap">
              {/* 显示完整历史文本 */}
              {messages.join('')}
            </div>
          ) : (
            <div className="text-gray-500 dark:text-gray-400">
              等待接收LLM文本块...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
