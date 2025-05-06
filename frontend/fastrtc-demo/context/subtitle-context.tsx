"use client"

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface SubtitleContextType {
  showSubtitle: boolean;
  setShowSubtitle: (show: boolean) => void;
  currentText: string;
  updateSubtitle: (text: string) => void;
  clearSubtitle: () => void;
  fadeState: string;
  setFadeState: (state: string) => void;
}

// 创建上下文
const SubtitleContext = createContext<SubtitleContextType | undefined>(undefined);

// 提供者组件
export function SubtitleProvider({ children }: { children: ReactNode }) {
  const [showSubtitle, setShowSubtitle] = useState<boolean>(true);
  const [currentText, setCurrentText] = useState<string>('');
  const [fadeState, setFadeState] = useState<string>('fade-in');

  const updateSubtitle = useCallback((text: string) => {
    setCurrentText(text);
  }, []);

  const clearSubtitle = useCallback(() => {
    setCurrentText('');
  }, []);

  return (
    <SubtitleContext.Provider
      value={{
        showSubtitle,
        setShowSubtitle,
        currentText,
        updateSubtitle,
        clearSubtitle,
        fadeState,
        setFadeState
      }}
    >
      {children}
    </SubtitleContext.Provider>
  );
}

// 自定义钩子，用于访问上下文
export function useSubtitle() {
  const context = useContext(SubtitleContext);
  
  if (context === undefined) {
    throw new Error('useSubtitle must be used within a SubtitleProvider');
  }
  
  return context;
}
