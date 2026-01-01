import React from 'react';
import { DisplayMode, PixelData } from '../types';

interface MatrixViewProps {
  data: PixelData[][]; // 10x10 grid of pixels
  mode: DisplayMode;
}

const MatrixView: React.FC<MatrixViewProps> = ({ data, mode }) => {
  const getCellContent = (pixel: PixelData) => {
    if (!pixel) return '';
    
    switch (mode) {
      case DisplayMode.RGB:
        return (
          <div className="flex flex-col text-[8px] leading-tight">
            <span className="text-red-400">{pixel.r}</span>
            <span className="text-green-400">{pixel.g}</span>
            <span className="text-blue-400">{pixel.b}</span>
          </div>
        );
      case DisplayMode.HEX:
        const toHex = (n: number) => n.toString(16).padStart(2, '0').toUpperCase();
        return <span className="text-[9px]">#{toHex(pixel.r)}{toHex(pixel.g)}{toHex(pixel.b)}</span>;
      case DisplayMode.GRAYSCALE:
        // Standard Rec. 601 luma
        const gray = Math.round(0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b);
        return <span className="text-xs font-bold">{gray}</span>;
    }
  };

  const getCellColor = (pixel: PixelData) => {
    if (!pixel) return 'transparent';
    if (mode === DisplayMode.GRAYSCALE) {
      const gray = Math.round(0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b);
      return `rgb(${gray}, ${gray}, ${gray})`;
    }
    return `rgb(${pixel.r}, ${pixel.g}, ${pixel.b})`;
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center p-4">
      <div className="grid grid-cols-10 gap-px bg-slate-700/50 p-1 rounded-lg border border-glassBorder shadow-2xl backdrop-blur-xl">
        {data.map((row, y) => 
          row.map((pixel, x) => (
            <div 
              key={`${y}-${x}`}
              className="w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center relative group transition-all duration-75"
              style={{
                backgroundColor: 'rgba(15, 23, 42, 0.8)', // Base dark
              }}
            >
              {/* Color indicator background opacity */}
              <div 
                className="absolute inset-0 opacity-20 group-hover:opacity-40 transition-opacity"
                style={{ backgroundColor: getCellColor(pixel) }}
              />
              
              <div className="z-10 text-slate-200 font-mono">
                {getCellContent(pixel)}
              </div>
              
              {/* Center Highlight */}
              {x === 4 && y === 4 && (
                <div className="absolute inset-0 border-2 border-neonCyan rounded-sm pointer-events-none shadow-[0_0_10px_rgba(6,182,212,0.5)]"></div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default MatrixView;