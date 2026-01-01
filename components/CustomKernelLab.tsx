import React, { useEffect, useState } from 'react';

interface CustomKernelLabProps {
  onKernelChange: (matrix: number[]) => void;
  active: boolean;
}

const CustomKernelLab: React.FC<CustomKernelLabProps> = ({ onKernelChange, active }) => {
  const [grid, setGrid] = useState<number[]>([
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
  ]);

  const handleChange = (index: number, val: string) => {
    const newGrid = [...grid];
    const parsed = parseFloat(val);
    newGrid[index] = isNaN(parsed) ? 0 : parsed;
    setGrid(newGrid);
  };

  useEffect(() => {
    if (active) {
      onKernelChange(grid);
    }
  }, [grid, active, onKernelChange]);

  return (
    <div className="mt-4 p-4 bg-glass rounded-xl border border-glassBorder">
      <h3 className="text-neonCyan font-mono text-sm mb-3 flex items-center gap-2">
        <span className="w-2 h-2 bg-neonCyan rounded-full animate-pulse"/>
        Custom Kernel Lab
      </h3>
      <div className="grid grid-cols-3 gap-2 mb-2">
        {grid.map((val, i) => (
          <input
            key={i}
            type="number"
            value={val}
            onChange={(e) => handleChange(i, e.target.value)}
            className="w-full bg-slate-900/80 border border-slate-600 rounded text-center text-xs py-2 text-white focus:border-neonIndigo focus:outline-none focus:ring-1 focus:ring-neonIndigo transition-all"
          />
        ))}
      </div>
      <p className="text-[10px] text-slate-400">
        Edit weights to create custom filters. Try edge detection or blurring.
      </p>
    </div>
  );
};

export default CustomKernelLab;