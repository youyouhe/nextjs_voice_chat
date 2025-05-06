import React, { memo } from 'react';
import { canvasStyles } from './canvas-styles';
import { useSubtitleDisplay } from '@/hooks/canvas/use-subtitle-display';
import { useSubtitle } from '@/context/subtitle-context';

// CSS for fade transitions
const fadeTransitionStyles = {
  base: "transition-opacity duration-300 ease-in-out",
  "fade-in": "opacity-100",
  "fade-out": "opacity-0",
};

// Type definitions
interface SubtitleTextProps {
  text: string;
  className?: string;
}

// Reusable components
const SubtitleText = memo(({ text, className = "" }: SubtitleTextProps) => (
  <span className={`text-white text-center block ${className}`} style={{ fontSize: "35px" }}>
    {text}
  </span>
));

SubtitleText.displayName = 'SubtitleText';

// Main component
const Subtitle = memo((): React.ReactNode => {
  const { subtitleText, isLoaded } = useSubtitleDisplay();
  const { showSubtitle, fadeState } = useSubtitle();
  
  // console.log("Subtitle component:", { showSubtitle, isLoaded, fadeState, hasText: !!subtitleText });
  if (!isLoaded || !subtitleText || !showSubtitle) return null;

  // 合并淡入淡出的CSS类
  const fadeClass = `${fadeTransitionStyles.base} ${fadeTransitionStyles[fadeState as keyof typeof fadeTransitionStyles] || ''}`;

  return (
    <div 
      className="absolute bottom-16 left-0 right-0 flex justify-center items-center z-30 px-4" 
    >
      <SubtitleText text={subtitleText} className={fadeClass} />
    </div>
  );
});

Subtitle.displayName = 'Subtitle';

export default Subtitle;
