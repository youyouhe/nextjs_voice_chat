"use client";

import { useEffect, useState } from 'react';
import { useSubtitle } from '@/context/subtitle-context';

export function useSubtitleDisplay() {
  const { currentText } = useSubtitle();
  const [subtitleText, setSubtitleText] = useState<string>('');
  const [isLoaded, setIsLoaded] = useState<boolean>(false);

  // 当文本变化时更新字幕显示
  useEffect(() => {
    if (currentText) {
      setSubtitleText(currentText);
    } else {
      setSubtitleText('');
    }
    
    // 确保组件加载完成
    if (!isLoaded) {
      setIsLoaded(true);
    }
  }, [currentText, isLoaded]);

  return {
    subtitleText,
    isLoaded,
  };
}
