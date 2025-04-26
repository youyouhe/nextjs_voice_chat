import React, { memo } from 'react';
import { canvasStyles } from './canvas-styles';
import { useSubtitleDisplay } from '@/hooks/canvas/use-subtitle-display';
import { useSubtitle } from '@/context/subtitle-context';

// Type definitions
interface SubtitleTextProps {
  text: string
}

// Reusable components
const SubtitleText = memo(({ text }: SubtitleTextProps) => (
  <span className="text-white text-center block" style={{ fontSize: "80px", fontWeight: "bold" }}>
    {text}
  </span>
));

SubtitleText.displayName = 'SubtitleText';

// Main component
const Subtitle = memo((): React.ReactNode => {
  const { subtitleText, isLoaded } = useSubtitleDisplay();
  const { showSubtitle } = useSubtitle();
  // console.log("Subtitle component:", { showSubtitle, isLoaded, hasText: !!subtitleText });
  if (!isLoaded || !subtitleText || !showSubtitle) return null;

  return (
    <div 
      className="absolute bottom-16 left-0 right-0 flex justify-center items-center z-30 px-4" 
    >
      <SubtitleText text={subtitleText} />
    </div>
  );
});

Subtitle.displayName = 'Subtitle';

export default Subtitle;
