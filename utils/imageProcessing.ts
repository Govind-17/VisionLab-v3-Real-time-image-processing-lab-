import { Kernel, HistogramData, MorphologyType, ColorChannel } from '../types';

// --- KERNEL CONVOLUTION ---
export const convolve = (
  source: ImageData,
  kernel: Kernel
): ImageData => {
  const side = kernel.width;
  const halfSide = Math.floor(side / 2);
  const src = source.data;
  const sw = source.width;
  const sh = source.height;
  
  const output = new ImageData(sw, sh);
  const dst = output.data;
  
  const weights = kernel.matrix;
  const factor = kernel.factor ?? 1;
  const bias = kernel.bias ?? 0;

  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      let r = 0, g = 0, b = 0;

      for (let cy = 0; cy < side; cy++) {
        for (let cx = 0; cx < side; cx++) {
          const scy = y + cy - halfSide;
          const scx = x + cx - halfSide;

          if (scy >= 0 && scy < sh && scx >= 0 && scx < sw) {
            const srcOff = (scy * sw + scx) * 4;
            const wt = weights[cy * side + cx];
            r += src[srcOff] * wt;
            g += src[srcOff + 1] * wt;
            b += src[srcOff + 2] * wt;
          }
        }
      }

      const dstOff = (y * sw + x) * 4;
      dst[dstOff] = Math.min(255, Math.max(0, r * factor + bias));
      dst[dstOff + 1] = Math.min(255, Math.max(0, g * factor + bias));
      dst[dstOff + 2] = Math.min(255, Math.max(0, b * factor + bias));
      dst[dstOff + 3] = 255;
    }
  }
  return output;
};

// --- MORPHOLOGY (DILATION / EROSION) ---
export const applyMorphology = (
  source: ImageData,
  type: MorphologyType,
  radius: number
): ImageData => {
  const width = source.width;
  const height = source.height;
  const src = source.data;
  const output = new ImageData(width, height);
  const dst = output.data;

  // Optimizing simply for square kernel. 
  // For production, separable filters or histogram-based sliding windows are faster.
  const side = radius * 2 + 1;
  const half = radius;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let rVal = type === MorphologyType.EROSION ? 255 : 0;
      let gVal = type === MorphologyType.EROSION ? 255 : 0;
      let bVal = type === MorphologyType.EROSION ? 255 : 0;

      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const py = y + ky;
          const px = x + kx;

          if (px >= 0 && px < width && py >= 0 && py < height) {
            const idx = (py * width + px) * 4;
            if (type === MorphologyType.EROSION) {
              rVal = Math.min(rVal, src[idx]);
              gVal = Math.min(gVal, src[idx+1]);
              bVal = Math.min(bVal, src[idx+2]);
            } else {
              rVal = Math.max(rVal, src[idx]);
              gVal = Math.max(gVal, src[idx+1]);
              bVal = Math.max(bVal, src[idx+2]);
            }
          }
        }
      }

      const outIdx = (y * width + x) * 4;
      dst[outIdx] = rVal;
      dst[outIdx + 1] = gVal;
      dst[outIdx + 2] = bVal;
      dst[outIdx + 3] = 255;
    }
  }
  return output;
};

// --- BIT PLANE SLICING ---
export const sliceBitPlane = (source: ImageData, bit: number): ImageData => {
  const data = source.data;
  const len = data.length;
  const output = new ImageData(source.width, source.height);
  const dst = output.data;
  const mask = 1 << (bit - 1);

  for (let i = 0; i < len; i += 4) {
    // Convert to grayscale for structural visibility first, or apply to RGB channels?
    // Standard educational approach: Grayscale -> Slice.
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    const val = (Math.floor(gray) & mask) ? 255 : 0;
    
    dst[i] = val;
    dst[i+1] = val;
    dst[i+2] = val;
    dst[i+3] = 255;
  }
  return output;
};

// --- COLOR SPACE DECOMPOSITION ---
export const applyColorSpace = (source: ImageData, channel: ColorChannel): ImageData => {
  if (channel === ColorChannel.RGB) return source; // Copy needed? usually handled in app

  const data = source.data;
  const output = new ImageData(source.width, source.height);
  const dst = output.data;
  const len = data.length;

  for (let i = 0; i < len; i += 4) {
    const r = data[i];
    const g = data[i+1];
    const b = data[i+2];

    if (channel === ColorChannel.GRAYSCALE) {
      const val = 0.299 * r + 0.587 * g + 0.114 * b;
      dst[i] = dst[i+1] = dst[i+2] = val;
      dst[i+3] = 255;
      continue;
    }

    // RGB to HSV Conversion
    const rNorm = r / 255, gNorm = g / 255, bNorm = b / 255;
    const max = Math.max(rNorm, gNorm, bNorm);
    const min = Math.min(rNorm, gNorm, bNorm);
    const d = max - min;
    
    let h = 0;
    let s = (max === 0 ? 0 : d / max);
    let v = max;

    if (max !== min) {
      switch (max) {
        case rNorm: h = (gNorm - bNorm) + (gNorm < bNorm ? 6 : 0); h /= 6; break;
        case gNorm: h = (bNorm - rNorm) + 2; h /= 6; break;
        case bNorm: h = (rNorm - gNorm) + 4; h /= 6; break;
      }
    }

    let outVal = 0;
    switch (channel) {
      case ColorChannel.HSV_HUE: outVal = h * 255; break;
      case ColorChannel.HSV_SAT: outVal = s * 255; break;
      case ColorChannel.HSV_VAL: outVal = v * 255; break;
      default: outVal = 0;
    }

    dst[i] = dst[i+1] = dst[i+2] = outVal;
    dst[i+3] = 255;
  }
  return output;
};

// --- HISTOGRAM CALCULATION ---
export const computeHistogram = (imageData: ImageData): HistogramData => {
  const r = new Uint32Array(256);
  const g = new Uint32Array(256);
  const b = new Uint32Array(256);
  const l = new Uint32Array(256);
  
  const data = imageData.data;
  let max = 0;

  for (let i = 0; i < data.length; i += 4) {
    const rv = data[i];
    const gv = data[i + 1];
    const bv = data[i + 2];
    const lv = Math.round(0.299 * rv + 0.587 * gv + 0.114 * bv);

    r[rv]++;
    g[gv]++;
    b[bv]++;
    l[lv]++;
  }

  // Find max for normalization
  for(let i = 0; i < 256; i++) {
    if (r[i] > max) max = r[i];
    if (g[i] > max) max = g[i];
    if (b[i] > max) max = b[i];
    if (l[i] > max) max = l[i];
  }

  return { r, g, b, l, max };
};

// --- PIXEL PROBE ---
export const getPixelData = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number
) => {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  
  const startX = Math.floor(x - size / 2);
  const startY = Math.floor(y - size / 2);
  
  const safeX = Math.max(0, startX);
  const safeY = Math.max(0, startY);
  const safeW = Math.min(width - safeX, size);
  const safeH = Math.min(height - safeY, size);

  if (safeW <= 0 || safeH <= 0) return null;

  try {
    const imageData = ctx.getImageData(safeX, safeY, safeW, safeH);
    return { data: imageData, startX: safeX, startY: safeY };
  } catch (e) {
    return null;
  }
};