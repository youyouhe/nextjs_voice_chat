"use client";

import { useEffect } from 'react';
import { SubtitleProvider, useSubtitle } from '@/context/subtitle-context';
import Subtitle from '@/components/ui/subtitle';

interface SubtitleContentProps {
  text: string;
  fadeState?: string; // 新增淡入淡出状态参数
}

// 控制字幕内容的组件
function SubtitleContent({ text, fadeState = "fade-in" }: SubtitleContentProps) {
  const { updateSubtitle, setFadeState } = useSubtitle();
  
  useEffect(() => {
    if (text) {
      updateSubtitle(text);
    }
    // 更新淡入淡出状态
    setFadeState(fadeState);
  }, [text, fadeState, updateSubtitle, setFadeState]);
  
  return <Subtitle />;
}

// 提供字幕上下文的包装组件
export function SubtitleWrapper({ text, fadeState = "fade-in" }: SubtitleContentProps) {
  return (
    <SubtitleProvider>
      <SubtitleContent text={text} />
    </SubtitleProvider>
  );
}
