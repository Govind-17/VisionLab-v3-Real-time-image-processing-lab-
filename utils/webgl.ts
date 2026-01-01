import { FilterNode, FilterType, MorphologyType } from '../types';
import { 
  VERTEX_SHADER, 
  PASSTHROUGH_FRAGMENT, 
  KERNEL_FRAGMENT, 
  MORPHOLOGY_FRAGMENT, 
  BIT_PLANE_FRAGMENT 
} from './shaders';
import { MOTION_HEATMAP_FRAGMENT, DEFAULT_MOTION_THRESHOLD } from '../constants';

export class WebGLEngine {
  gl: WebGLRenderingContext;
  programCache: Map<string, WebGLProgram> = new Map();
  positionBuffer: WebGLBuffer;
  
  // Dedicated texture for the initial input image/video frame
  sourceTexture: WebGLTexture;

  // Ping-Pong Buffers: Used for intermediate filter outputs
  pingPongFBOs: [WebGLFramebuffer, WebGLFramebuffer]; 
  pingPongTextures: [WebGLTexture, WebGLTexture];
  
  // Dedicated texture for storing the raw input image from the *previous* frame for motion detection
  prevRawFBO: WebGLFramebuffer;
  prevRawTexture: WebGLTexture;

  width: number = 0;
  height: number = 0;
  startTime: number = Date.now();

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', { 
      preserveDrawingBuffer: true, // Needed for readPixels to work correctly
      premultipliedAlpha: false // Standard for image processing
    });
    if (!gl) throw new Error("WebGL not supported");
    this.gl = gl;

    // Full screen quad setup
    this.positionBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    // Initialize all textures and FBOs
    this.sourceTexture = gl.createTexture()!;
    this.pingPongFBOs = [gl.createFramebuffer()!, gl.createFramebuffer()!];
    this.pingPongTextures = [gl.createTexture()!, gl.createTexture()!];
    this.prevRawFBO = gl.createFramebuffer()!;
    this.prevRawTexture = gl.createTexture()!;
  }

  resize(width: number, height: number) {
    if (this.width === width && this.height === height) return;
    this.width = width;
    this.height = height;
    this.gl.canvas.width = width;
    this.gl.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
    this.initTextures(width, height);
  }

  initTextures(width: number, height: number) {
    const gl = this.gl;

    // Helper to create and bind texture parameters
    const createAndBindTexture = (texture: WebGLTexture) => {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    };

    // Source Texture
    createAndBindTexture(this.sourceTexture);

    // Ping-pong textures and FBOs
    [0, 1].forEach(i => {
      createAndBindTexture(this.pingPongTextures[i]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pingPongFBOs[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.pingPongTextures[i], 0);
    });

    // Previous Raw Frame Texture and FBO
    createAndBindTexture(this.prevRawTexture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.prevRawFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.prevRawTexture, 0);
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Unbind FBO
  }

  createShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${info}`);
    }
    return shader;
  }

  getProgram(fsSource: string): WebGLProgram {
    if (this.programCache.has(fsSource)) return this.programCache.get(fsSource)!;

    try {
      const gl = this.gl;
      const vs = this.createShader(gl.VERTEX_SHADER, VERTEX_SHADER);
      const fs = this.createShader(gl.FRAGMENT_SHADER, fsSource);
      const program = gl.createProgram()!;
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error("Program link error");
      }
      this.programCache.set(fsSource, program);
      return program;
    } catch (e) {
      console.error(e);
      // Fallback to passthrough if custom shader fails
      return this.getProgram(PASSTHROUGH_FRAGMENT);
    }
  }

  // Load source image into initial sourceTexture
  loadImage(source: TexImageSource) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.sourceTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  // Helper to copy content from one texture to another
  private copyTexture(sourceTexture: WebGLTexture, destinationTexture: WebGLTexture, destinationFBO: WebGLFramebuffer) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, destinationFBO);
    gl.viewport(0, 0, this.width, this.height);
    
    const program = this.getProgram(PASSTHROUGH_FRAGMENT);
    gl.useProgram(program);

    const posLoc = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const imgLoc = gl.getUniformLocation(program, "u_image");
    gl.uniform1i(imgLoc, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
    
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Unbind FBO
  }


  renderPipeline(pipeline: FilterNode[]) {
    const gl = this.gl;
    
    // Copy the current raw source image to prevRawTexture for the *next* frame's motion detection
    // This must happen *before* the pipeline for the current frame is processed.
    this.copyTexture(this.sourceTexture, this.prevRawTexture, this.prevRawFBO);

    // Initial state: first filter takes input from sourceTexture.
    // Result goes to pingPongTextures[0].
    this.copyTexture(this.sourceTexture, this.pingPongTextures[0], this.pingPongFBOs[0]);
    let currentInputTexture = this.pingPongTextures[0]; // The texture that holds the current image state for the next filter
    let count = 0; // Tracks which pingPongTexture is the *output* of the last filter (0 or 1)

    // Iterate through filter chain
    pipeline.forEach(node => {
      // If node is disabled, it means we effectively pass the current texture to the next step.
      if (!node.params.active) return;

      const sourceTextureForFilter = currentInputTexture;
      const destIdx = (count + 1) % 2; // Write to the other ping-pong buffer
      const destinationFBO = this.pingPongFBOs[destIdx];
      const destinationTexture = this.pingPongTextures[destIdx];

      // Draw into the 'Next' framebuffer
      gl.bindFramebuffer(gl.FRAMEBUFFER, destinationFBO);
      gl.viewport(0, 0, this.width, this.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Select Shader
      let fs = PASSTHROUGH_FRAGMENT;
      if (node.type === FilterType.KERNEL) fs = KERNEL_FRAGMENT;
      if (node.type === FilterType.MORPHOLOGY) fs = MORPHOLOGY_FRAGMENT;
      if (node.type === FilterType.BIT_PLANE) fs = BIT_PLANE_FRAGMENT;
      if (node.type === FilterType.MOTION_HEATMAP) fs = MOTION_HEATMAP_FRAGMENT; // NEW Phase 4
      if (node.type === FilterType.CUSTOM_GLSL && node.params.customSource) fs = node.params.customSource;

      const program = this.getProgram(fs);
      gl.useProgram(program);

      // Bind Geometry
      const posLoc = gl.getAttribLocation(program, "a_position");
      gl.enableVertexAttribArray(posLoc);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

      // Bind Uniforms
      const imgLoc = gl.getUniformLocation(program, "u_image");
      const resLoc = gl.getUniformLocation(program, "u_resolution");
      const timeLoc = gl.getUniformLocation(program, "u_time");
      
      gl.uniform1i(imgLoc, 0); // u_image will be texture unit 0
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, sourceTextureForFilter); // This is the *current* state of the image in the pipeline
      
      if(resLoc) gl.uniform2f(resLoc, this.width, this.height);
      if(timeLoc) gl.uniform1f(timeLoc, (Date.now() - this.startTime) / 1000);

      // Node Specific Uniforms
      if (node.type === FilterType.KERNEL && node.params.kernelMatrix) {
        const kLoc = gl.getUniformLocation(program, "u_kernel");
        const wLoc = gl.getUniformLocation(program, "u_kernelWeight");
        gl.uniform1fv(kLoc, new Float32Array(node.params.kernelMatrix));
        gl.uniform1f(wLoc, node.params.kernelWeight || 1.0);
      }

      if (node.type === FilterType.MORPHOLOGY) {
        const typeLoc = gl.getUniformLocation(program, "u_morphType");
        const radLoc = gl.getUniformLocation(program, "u_radius");
        gl.uniform1i(typeLoc, node.params.morphType === MorphologyType.DILATION ? 1 : 0);
        gl.uniform1i(radLoc, node.params.morphRadius || 1);
      }

      if (node.type === FilterType.BIT_PLANE) {
        const bpLoc = gl.getUniformLocation(program, "u_bitPlane");
        gl.uniform1f(bpLoc, node.params.bitPlane || 8.0);
      }

      // NEW Phase 4: Motion Heatmap Uniforms
      if (node.type === FilterType.MOTION_HEATMAP) {
        const prevRawLoc = gl.getUniformLocation(program, "u_prevRawFrame");
        const motionThresholdLoc = gl.getUniformLocation(program, "u_motionThreshold");
        
        gl.activeTexture(gl.TEXTURE1); // Use texture unit 1
        gl.bindTexture(gl.TEXTURE_2D, this.prevRawTexture); // This contains the raw input from the PREVIOUS frame
        gl.uniform1i(prevRawLoc, 1);
        
        gl.uniform1f(motionThresholdLoc, node.params.motionThreshold || DEFAULT_MOTION_THRESHOLD);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      
      currentInputTexture = destinationTexture; // Update currentInputTexture for the next filter
      count++;
    });

    // FINAL PASS: Draw the result to the screen (Canvas)
    // The final result of the pipeline is in currentInputTexture
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);

    const program = this.getProgram(PASSTHROUGH_FRAGMENT);
    gl.useProgram(program);
    
    const posLoc = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const imgLoc = gl.getUniformLocation(program, "u_image");
    gl.uniform1i(imgLoc, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentInputTexture); // Draw the final output of the pipeline
    
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  getPixels(): Uint8Array {
      const w = this.gl.drawingBufferWidth;
      const h = this.gl.drawingBufferHeight;
      const pixels = new Uint8Array(w * h * 4);
      this.gl.readPixels(0, 0, w, h, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
      
      const flipped = new Uint8Array(w * h * 4);
      const rowBytes = w * 4;
      for (let row = 0; row < h; row++) {
          const srcRow = row * rowBytes;
          const dstRow = (h - row - 1) * rowBytes;
          flipped.set(pixels.subarray(srcRow, srcRow + rowBytes), dstRow);
      }
      return flipped;
  }
}