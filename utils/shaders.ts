// Standard full-screen quad vertex shader
export const VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_texCoord;
void main() {
  gl_Position = vec4(a_position, 0, 1);
  v_texCoord = (a_position + 1.0) / 2.0;
  v_texCoord.y = 1.0 - v_texCoord.y; 
}
`;

export const PASSTHROUGH_FRAGMENT = `
precision mediump float;
uniform sampler2D u_image;
varying vec2 v_texCoord;
void main() {
  gl_FragColor = texture2D(u_image, v_texCoord);
}
`;

// Generic 3x3 Kernel Shader
export const KERNEL_FRAGMENT = `
precision mediump float;
uniform sampler2D u_image;
uniform vec2 u_resolution;
uniform float u_kernel[9];
uniform float u_kernelWeight;
varying vec2 v_texCoord;

void main() {
  vec2 onePixel = vec2(1.0, 1.0) / u_resolution;
  vec4 colorSum = vec4(0.0);
  
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2(-1, -1)) * u_kernel[0];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2( 0, -1)) * u_kernel[1];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2( 1, -1)) * u_kernel[2];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2(-1,  0)) * u_kernel[3];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2( 0,  0)) * u_kernel[4];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2( 1,  0)) * u_kernel[5];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2(-1,  1)) * u_kernel[6];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2( 0,  1)) * u_kernel[7];
  colorSum += texture2D(u_image, v_texCoord + onePixel * vec2( 1,  1)) * u_kernel[8];

  gl_FragColor = vec4((colorSum / u_kernelWeight).rgb, 1.0);
}
`;

// Morphology (Erosion/Dilation)
// Implementing a simple 3x3 or 5x5 loop for GPU
export const MORPHOLOGY_FRAGMENT = `
precision mediump float;
uniform sampler2D u_image;
uniform vec2 u_resolution;
uniform int u_morphType; // 0 = Erosion (Min), 1 = Dilation (Max)
uniform int u_radius; // Supports up to 5
varying vec2 v_texCoord;

void main() {
  vec2 onePixel = vec2(1.0, 1.0) / u_resolution;
  vec4 val = texture2D(u_image, v_texCoord);
  
  // Optimization: In a real app we might use separable filters, 
  // but for a lab, a nested loop is fine.
  for (int x = -5; x <= 5; x++) {
    for (int y = -5; y <= 5; y++) {
      // Corrected: Cast x, y, and u_radius to float for abs and comparison
      if (abs(float(x)) > float(u_radius) || abs(float(y)) > float(u_radius)) continue;
      
      vec4 neighbor = texture2D(u_image, v_texCoord + vec2(float(x), float(y)) * onePixel);
      
      if (u_morphType == 0) { // Erosion (Min)
        val = min(val, neighbor);
      } else { // Dilation (Max)
        val = max(val, neighbor);
      }
    }
  }
  gl_FragColor = val;
}
`;

// Bit Plane Slicing
export const BIT_PLANE_FRAGMENT = `
precision mediump float;
uniform sampler2D u_image;
uniform float u_bitPlane; // 1.0 to 8.0
varying vec2 v_texCoord;

void main() {
  vec4 color = texture2D(u_image, v_texCoord);
  float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
  
  // Convert 0-1 float to 0-255 integer approximation
  float val = gray * 255.0;
  
  // Power of 2 for the specific bit (0..7)
  float bitVal = pow(2.0, u_bitPlane - 1.0);
  
  // Check if bit is set using mod/floor logic adapted for float
  // (val / bitVal) % 2
  float sliced = mod(floor(val / bitVal), 2.0);
  
  // If sliced is 1, output white, else black
  // Threshold slightly to avoid precision errors
  if (sliced >= 0.5) {
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
  } else {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
  }
}
`;

// Default Hello World for the GLSL Editor
export const DEFAULT_CUSTOM_SHADER = `precision mediump float;
uniform sampler2D u_image;
uniform vec2 u_resolution;
uniform float u_time; // Added u_time for consistency with WebGLEngine
varying vec2 v_texCoord;

void main() {
  vec4 color = texture2D(u_image, v_texCoord);
  
  // Example: Chromatic Aberration
  float shift = sin(u_time * 2.0) * 0.005;
  float r = texture2D(u_image, v_texCoord + vec2(shift, 0.0)).r;
  float b = texture2D(u_image, v_texCoord - vec2(shift, 0.0)).b;
  
  gl_FragColor = vec4(r, color.g, b, 1.0);
}`;