"use client";

import { useEffect, useState } from 'react';
import { useSubtitle } from '@/context/subtitle-context';

export function useSubtitleDisplay() {
  const { currentText } = useSubtitle();
  const [subtitleText, setSubtitleText] = useState<string>('');
  const [isLoaded, setIsLoaded] = useState<boolean>(false);

  // 单独处理组件加载状态，只运行一次
  useEffect(() => {
    setIsLoaded(true);
  }, []);

  // 当文本变化时更新字幕显示
  useEffect(() => {
    if (currentText) {
      setSubtitleText(currentText);
    } else {
      setSubtitleText('');
    }
  }, [currentText]);

  return {
    subtitleText,
    isLoaded,
  };
}
