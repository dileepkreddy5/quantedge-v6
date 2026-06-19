import { useState, useEffect } from 'react';

export type Breakpoint = 'phone' | 'tablet' | 'desktop';

export function useResponsive(): { bp: Breakpoint; isPhone: boolean; isTablet: boolean; isMobile: boolean } {
  const get = (): Breakpoint => {
    if (typeof window === 'undefined') return 'desktop';
    const w = window.innerWidth;
    if (w < 640) return 'phone';
    if (w < 1024) return 'tablet';
    return 'desktop';
  };
  const [bp, setBp] = useState<Breakpoint>(get);
  useEffect(() => {
    let raf = 0;
    const onResize = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => setBp(get())); };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); cancelAnimationFrame(raf); };
  }, []);
  return { bp, isPhone: bp === 'phone', isTablet: bp === 'tablet', isMobile: bp !== 'desktop' };
}

export function gridCols(bp: Breakpoint): string {
  if (bp === 'phone') return '1fr';
  if (bp === 'tablet') return '1fr 1fr';
  return '1fr 1fr 1fr';
}
