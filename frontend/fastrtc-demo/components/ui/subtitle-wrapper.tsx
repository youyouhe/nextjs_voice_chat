"use client";

import { useEffect } from 'react';
import { SubtitleProvider, useSubtitle } from '@/context/subtitle-context';
import Subtitle from '@/components/ui/subtitle';

interface SubtitleContentProps {
  text: string;
}

// 控制字幕内容的组件
function SubtitleContent({ text }: SubtitleContentProps) {
  const { updateSubtitle } = useSubtitle();
  
  useEffect(() => {
    if (text) {
      updateSubtitle(text);
    }
  }, [text, updateSubtitle]);
  
  return <Subtitle />;
}

// 提供字幕上下文的包装组件
export function SubtitleWrapper({ text }: SubtitleContentProps) {
  return (
    <SubtitleProvider>
      <SubtitleContent text={text} />
    </SubtitleProvider>
  );
}
