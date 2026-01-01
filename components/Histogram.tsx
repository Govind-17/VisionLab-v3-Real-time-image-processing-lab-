import React, { useRef, useEffect } from 'react';
import { HistogramData } from '../types';

interface HistogramProps {
  data: HistogramData | null;
}

const Histogram: React.FC<HistogramProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.width;
    const height = canvas.height;
    const barWidth = width / 256;

    // Helper to draw channel
    const drawChannel = (histData: Uint32Array, color: string, compositeOp: GlobalCompositeOperation = 'screen') => {
      ctx.globalCompositeOperation = compositeOp;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let i = 0; i < 256; i++) {
        const value = histData[i];
        const percent = value / data.max;
        const barHeight = percent * height * 0.9; // Scale to 90% height
        const x = i * barWidth;
        const y = height - barHeight;
        
        ctx.lineTo(x, y);
        ctx.lineTo(x + barWidth, y);
      }
      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.fill();
    };

    // Draw RGB
    drawChannel(data.r, 'rgba(239, 68, 68, 0.6)'); // Red-500
    drawChannel(data.g, 'rgba(34, 197, 94, 0.6)'); // Green-500
    drawChannel(data.b, 'rgba(59, 130, 246, 0.6)'); // Blue-500
    
    // Draw Luminance outline on top
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < 256; i++) {
      const percent = data.l[i] / data.max;
      const h = percent * height * 0.9;
      const x = i * barWidth + barWidth/2;
      const y = height - h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

  }, [data]);

  return (
    <div className="w-full h-full bg-slate-900/50 rounded-lg border border-glassBorder relative group overflow-hidden">
      <canvas 
        ref={canvasRef} 
        width={300} 
        height={100} 
        className="w-full h-full opacity-80 group-hover:opacity-100 transition-opacity"
      />
    </div>
  );
};

export default Histogram;