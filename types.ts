export interface Kernel {
  name: string;
  matrix: number[];
  width: number;
  height: number;
  description: string;
  formula?: string;
  factor?: number; // Normalization factor
  bias?: number;
}

export interface PixelData {
  r: number;
  g: number;
  b: number;
  a: number;
}

export enum DisplayMode {
  RGB = 'RGB',
  HEX = 'HEX',
  GRAYSCALE = 'GRAY',
}

export interface ProbeState {
  x: number;
  y: number;
  data: PixelData[][]; // 10x10 grid
}

// v2 Lab Modes (Preserved for Reference/Formula display)
export enum LabMode {
  KERNEL = 'KERNEL',
  MORPHOLOGY = 'MORPHOLOGY',
  BIT_PLANE = 'BIT_PLANE',
  COLOR_SPACE = 'COLOR_SPACE',
}

export enum MorphologyType {
  EROSION = 'EROSION',
  DILATION = 'DILATION',
}

export enum ColorChannel {
  RGB = 'RGB',
  GRAYSCALE = 'GRAYSCALE',
  HSV_HUE = 'HSV_HUE',
  HSV_SAT = 'HSV_SAT',
  HSV_VAL = 'HSV_VAL',
}

export interface HistogramData {
  r: Uint32Array;
  g: Uint32Array;
  b: Uint32Array;
  l: Uint32Array; // Luminance
  max: number;
}

// --- PHASE 3: WebGL Pipeline Types ---

export enum FilterType {
  KERNEL = 'KERNEL',
  MORPHOLOGY = 'MORPHOLOGY',
  BIT_PLANE = 'BIT_PLANE',
  COLOR_ADJUST = 'COLOR_ADJUST',
  CUSTOM_GLSL = 'CUSTOM_GLSL',
}

export interface FilterParams {
  // Kernel
  kernelMatrix?: number[];
  kernelWeight?: number;
  
  // Morphology
  morphType?: MorphologyType;
  morphRadius?: number;

  // Bit Plane
  bitPlane?: number;

  // GLSL
  customSource?: string;

  // Generic
  active?: boolean;
}

export interface FilterNode {
  id: string;
  name: string;
  type: FilterType;
  params: FilterParams;
  expanded?: boolean; // UI State
}
