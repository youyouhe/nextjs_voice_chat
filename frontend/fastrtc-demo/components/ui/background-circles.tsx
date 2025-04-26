"use client";

import { motion } from "framer-motion";
import clsx from "clsx";
import { useState, useEffect } from "react";

interface BackgroundCirclesProps {
    title?: string;
    description?: string;
    className?: string;
    variant?: keyof typeof COLOR_VARIANTS;
    audioLevel?: number;
    isActive?: boolean;
    isGridAnimating?: boolean;
}

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
    tertiary: {
        border: [
            "border-orange-500/60",
            "border-yellow-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-orange-500/30",
    },
    quaternary: {
        border: [
            "border-purple-500/60",
            "border-pink-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-purple-500/30",
    },
    quinary: {
        border: [
            "border-red-500/60",
            "border-rose-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-red-500/30",
    }, // red
    senary: {
        border: [
            "border-blue-500/60",
            "border-sky-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-blue-500/30",
    }, // blue
    septenary: {
        border: [
            "border-gray-500/60",
            "border-gray-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-gray-500/30",
    },
    octonary: {
        border: [
            "border-red-500/60",
            "border-rose-400/50",
            "border-slate-600/30",
        ],
        gradient: "from-red-500/30",
    },
} as const;

interface AnimatedGridProps {
    isAnimating: boolean;
}

const AnimatedGrid = ({ isAnimating }: AnimatedGridProps) => (
    <motion.div
        className="absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,transparent_30%,black)]"
        animate={{
            backgroundPosition: ["0% 0%", "100% 100%"],
        }}
        transition={{
            duration: 40,
            repeat: isAnimating ? Number.POSITIVE_INFINITY : 0,
            ease: "linear",
        }}
    >
        <div className="h-full w-full [background-image:repeating-linear-gradient(100deg,#64748B_0%,#64748B_1px,transparent_1px,transparent_4%)] opacity-20" />
    </motion.div>
);

export function BackgroundCircles({
    title = "",
    description = "",
    className,
    variant = "octonary",
    audioLevel = 0,
    isActive = false,
}: BackgroundCirclesProps) {
    const variantStyles = COLOR_VARIANTS[variant];
    const [animationParams, setAnimationParams] = useState({
        scale: 1,
        duration: 5,
        intensity: 0
    });
    const [isLoaded, setIsLoaded] = useState(false);
    
    // Initial page load animation
    useEffect(() => {
        // Small delay to ensure the black screen is visible first
        const timer = setTimeout(() => {
            setIsLoaded(true);
        }, 300);
        
        return () => clearTimeout(timer);
    }, []);
    
    // Update animation based on audio level
    useEffect(() => {
        // console.log('audioLevel changed:', audioLevel, 'isActive:', isActive);
        if (isActive && audioLevel > 0) {
            // Simple enhancement of audio level for more dramatic effect
            const enhancedLevel = Math.min(1, audioLevel * 2.0); // Increased amplification factor
            // console.log('enhancedLevel :', enhancedLevel, 'isActive:', isActive);
            setAnimationParams({
                scale: 1 + enhancedLevel * 0.3,
                duration: Math.max(2.0, 5 - enhancedLevel * 2.5), // Shorter duration, more sensitive to intensity
                intensity: enhancedLevel
            });
        } else if (animationParams.intensity > 0) {
            // Only reset if we need to (prevents unnecessary updates)
            const timer = setTimeout(() => {
                setAnimationParams(prevParams => {
                    // 只有在当前强度大于0时才重置，避免不必要的更新
                    if (prevParams.intensity > 0) {
                        return {
                            scale: 1,
                            duration: 5,
                            intensity: 0
                        };
                    }
                    return prevParams;
                });
            }, 300);
            
            return () => clearTimeout(timer);
        }
    }, [audioLevel, isActive]); // 移除 animationParams.intensity 依赖
    
    return (
        <>
            {/* Initial black overlay that fades out */}
            <motion.div 
                className="fixed inset-0 bg-black z-50"
                initial={{ opacity: 1 }}
                animate={{ opacity: isLoaded ? 0 : 1 }}
                transition={{ duration: 1.2, ease: "easeInOut" }}
                style={{ pointerEvents: isLoaded ? "none" : "auto" }}
            />
            
            <div
                className={clsx(
                    "relative flex h-screen w-full items-center justify-center overflow-hidden",
                    "bg-white dark:bg-black/5",
                    className
                )}
            >
                <AnimatedGrid isAnimating={false} />
                <motion.div
                    className="absolute h-[480px] w-[480px]"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ 
                        opacity: isLoaded ? 1 : 0,
                        scale: isLoaded ? 1 : 0.9
                    }}
                    transition={{ 
                        duration: 1.5, 
                        delay: 0.3,
                        ease: "easeInOut" 
                    }}
                >
                    {[0,1,2,3,4,5].map((i) => (
                        <motion.div
                            key={i}
                            className={clsx(
                                "absolute inset-0 rounded-full",
                                "border-2 bg-gradient-to-br to-transparent",
                                variantStyles.border[i],
                                variantStyles.gradient
                            )}
                            animate={{
                                rotate: 360,
                                scale: [
                                    0.7 + (i * 0.05),
                                    (0.7 + (i * 0.05)) * (1 + (isActive ? animationParams.intensity * 0.8 : 0.02)), // Increased intensity effect
                                    0.7 + (i * 0.05)
                                ],
                                opacity: [
                                    0.7 + (i * 0.1),
                                    0.8 + (i * 0.1) + (isActive ? animationParams.intensity * 0.4 : 0), // Increased intensity effect
                                    0.7 + (i * 0.1)
                                ]
                            }}
                            transition={{
                                duration: isActive ? animationParams.duration : 8 + (i * 2),
                                repeat: Number.POSITIVE_INFINITY,
                                ease: "easeOut",
                            }}
                        >
                            <div
                                className={clsx(
                                    "absolute inset-0 rounded-full mix-blend-screen",
                                    `bg-[radial-gradient(ellipse_at_center,${variantStyles.gradient.replace(
                                        "from-",
                                        ""
                                    )}/10%,transparent_70%)]`
                                )}
                            />
                        </motion.div>
                    ))}
                </motion.div>

                <div className="absolute inset-0 [mask-image:radial-gradient(90%_60%_at_50%_50%,#000_40%,transparent)]">
                    <motion.div 
                        className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,#0F766E/30%,transparent_70%)] blur-[120px]"
                        initial={{ opacity: 0 }}
                        animate={{
                            opacity: isLoaded ? 0.7 : 0,
                            scale: [1, 1 + (isActive ? animationParams.intensity * 0.6 : 0.02), 1], // Increased intensity effect
                        }}
                        transition={{
                            opacity: { duration: 1.8, delay: 0.5 },
                            scale: { 
                                duration: isActive ? 2 : 12,
                                repeat: Number.POSITIVE_INFINITY,
                                ease: "easeOut",
                            }
                        }}
                    />
                    <motion.div 
                        className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,#2DD4BF/15%,transparent)] blur-[80px]"
                        initial={{ opacity: 0 }}
                        animate={{
                            opacity: isLoaded ? 1 : 0,
                            scale: [1, 1 + (isActive ? animationParams.intensity * 0.8 : 0.03), 1] // Increased intensity effect
                        }}
                        transition={{
                            opacity: { duration: 2, delay: 0.7 },
                            scale: { 
                                duration: isActive ? 1.5 : 15,
                                repeat: Number.POSITIVE_INFINITY,
                                ease: "easeOut",
                            }
                        }}
                    />
                    
                    {/* Additional glow that appears only during high audio levels */}
                    {isActive && animationParams.intensity > 0.3 && (
                        <motion.div 
                            className={`absolute inset-0 bg-[radial-gradient(ellipse_at_center,${variantStyles.gradient.replace("from-", "")}/20%,transparent_70%)] blur-[60px]`}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{
                                opacity: [0, Math.min(1, animationParams.intensity * 0.8), 0], // Increased intensity effect, capped at 1
                                scale: [0.8, 1.1, 0.8],
                            }}
                            transition={{
                                duration: 0.8,
                                repeat: Number.POSITIVE_INFINITY,
                                ease: "easeOut",
                            }}
                        />
                    )}
                </div>
            </div>
        </>
    );
}

export function DemoCircles() {
    const [currentVariant, setCurrentVariant] =
        useState<keyof typeof COLOR_VARIANTS>("octonary");

    const variants = Object.keys(
        COLOR_VARIANTS
    ) as (keyof typeof COLOR_VARIANTS)[];

    function getNextVariant() {
        const currentIndex = variants.indexOf(currentVariant);
        const nextVariant = variants[(currentIndex + 1) % variants.length];
        return nextVariant;
    }

    return (
        <>
            <BackgroundCircles variant={currentVariant} />
            <div className="absolute top-12 right-12">
                <button
                    type="button"
                    className="bg-slate-950 dark:bg-white text-white dark:text-slate-950 px-4 py-1 rounded-md z-10 text-sm font-medium"
                    onClick={() => {
                        setCurrentVariant(getNextVariant());
                    }}
                >
                    Change Variant
                </button>
            </div>
        </>
    );
}
