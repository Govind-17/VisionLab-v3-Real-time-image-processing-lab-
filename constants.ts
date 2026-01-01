import { Kernel } from './types';

export const KERNELS: Record<string, Kernel> = {
  identity: {
    name: 'Identity',
    matrix: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    width: 3,
    height: 3,
    description: "Does not modify the image. The center pixel is kept as is.",
    formula: "I(x, y) = I(x, y)"
  },
  gaussianBlur3: {
    name: 'Gaussian Blur 3x3',
    matrix: [1, 2, 1, 2, 4, 2, 1, 2, 1],
    width: 3,
    height: 3,
    factor: 1 / 16,
    description: "Smooths the image by averaging pixels with a weighted Gaussian distribution.",
    formula: "G(x, y) = (1/16) * Σ (kernel * I)"
  },
  gaussianBlur5: {
    name: 'Gaussian Blur 5x5',
    matrix: [
      1, 4, 6, 4, 1,
      4, 16, 24, 16, 4,
      6, 24, 36, 24, 6,
      4, 16, 24, 16, 4,
      1, 4, 6, 4, 1
    ],
    width: 5,
    height: 5,
    factor: 1 / 256,
    description: "Stronger smoothing with a larger 5x5 kernel window.",
    formula: "G(x, y) = (1/256) * Σ (kernel * I)"
  },
  sharpen: {
    name: 'Sharpen',
    matrix: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    width: 3,
    height: 3,
    description: "Enhances edges by subtracting surrounding pixels from the center.",
    formula: "S(x, y) = 5*I(x,y) - neighbors"
  },
  laplacian: {
    name: 'Laplacian',
    matrix: [0, 1, 0, 1, -4, 1, 0, 1, 0],
    width: 3,
    height: 3,
    description: "Detects rapid changes (edges) regardless of orientation. Second derivative.",
    formula: "∇²f = ∂²f/∂x² + ∂²f/∂y²"
  },
  sobelX: {
    name: 'Sobel Horizontal',
    matrix: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    width: 3,
    height: 3,
    description: "Calculates the gradient approximation in the horizontal direction.",
    formula: "Gx = Kx * I"
  },
  sobelY: {
    name: 'Sobel Vertical',
    matrix: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    width: 3,
    height: 3,
    description: "Calculates the gradient approximation in the vertical direction.",
    formula: "Gy = Ky * I"
  },
  emboss: {
    name: 'Emboss',
    matrix: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
    width: 3,
    height: 3,
    description: "Creates a 3D shadow effect.",
    formula: "E(x, y) = I(x, y) + Shadow"
  }
};

export const INITIAL_CUSTOM_KERNEL = [0, 0, 0, 0, 1, 0, 0, 0, 0];
export const PROBE_SIZE = 10;
export const SAMPLE_IMAGE_URL = "https://picsum.photos/800/600";