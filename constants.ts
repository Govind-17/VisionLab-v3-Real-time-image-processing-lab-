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

// NEW Phase 4 Constants
export const DEFAULT_MOTION_THRESHOLD = 0.05; // 0.0 - 1.0 (normalized pixel difference)
export const MOTION_HEATMAP_FRAGMENT = `
precision mediump float;
uniform sampler2D u_image;          // Current texture in pipeline (output of previous filter)
uniform sampler2D u_prevRawFrame;   // Raw input frame from the previous cycle
uniform vec2 u_resolution;
uniform float u_motionThreshold;
varying vec2 v_texCoord;

void main() {
  vec4 currentFilteredPx = texture2D(u_image, v_texCoord); // The image as it stands in the pipeline
  vec4 prevRawPx = texture2D(u_prevRawFrame, v_texCoord);   // The raw image from last frame

  // Calculate difference in grayscale for robustness
  float currentGray = dot(currentFilteredPx.rgb, vec3(0.299, 0.587, 0.114));
  float prevGray = dot(prevRawPx.rgb, vec3(0.299, 0.587, 0.114));

  float diff = abs(currentGray - prevGray);
  
  if (diff > u_motionThreshold) {
    // Mix with neon green to highlight motion on top of the current filtered image
    gl_FragColor = mix(currentFilteredPx, vec4(0.2, 1.0, 0.4, 0.7), 0.5); // 50% blend with semi-transparent green
  } else {
    gl_FragColor = currentFilteredPx; // No motion, just pass through the current filtered image
  }
}
`;