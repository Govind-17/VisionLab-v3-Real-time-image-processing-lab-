import { FilterNode, FilterType, MorphologyType } from '../types';
import { 
  VERTEX_SHADER, 
  PASSTHROUGH_FRAGMENT, 
  KERNEL_FRAGMENT, 
  MORPHOLOGY_FRAGMENT, 
  BIT_PLANE_FRAGMENT 
} from './shaders';

export class WebGLEngine {
  gl: WebGLRenderingContext;
  programCache: Map<string, WebGLProgram> = new Map();
  positionBuffer: WebGLBuffer;
  
  // Ping-Pong Buffers: We toggle between reading from [0] and writing to [1], then swap.
  framebuffers: [WebGLFramebuffer, WebGLFramebuffer]; 
  textures: [WebGLTexture, WebGLTexture];
  
  width: number = 0;
  height: number = 0;
  startTime: number = Date.now();

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', { 
      preserveDrawingBuffer: true,
      premultipliedAlpha: false 
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

    // Initialize FBOs placeholder
    this.framebuffers = [gl.createFramebuffer()!, gl.createFramebuffer()!];
    this.textures = [gl.createTexture()!, gl.createTexture()!];
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
    // Set up textures for ping-pong
    [0, 1].forEach(i => {
      gl.bindTexture(gl.TEXTURE_2D, this.textures[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures[i], 0);
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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

  // Load source image into initial texture
  loadImage(source: TexImageSource) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  renderPipeline(pipeline: FilterNode[]) {
    const gl = this.gl;
    let count = 0; // Tracks which texture holds the current result. 0 = textures[0], 1 = textures[1]

    // Iterate through filter chain
    pipeline.forEach(node => {
      // If node is disabled, we simply skip processing, 
      // effectively passing the current texture to the next step (or final output)
      if (!node.params.active) return;

      const sourceIdx = count % 2;
      const destIdx = (count + 1) % 2;

      // Draw into the 'Next' framebuffer
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[destIdx]);
      gl.viewport(0, 0, this.width, this.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Select Shader
      let fs = PASSTHROUGH_FRAGMENT;
      if (node.type === FilterType.KERNEL) fs = KERNEL_FRAGMENT;
      if (node.type === FilterType.MORPHOLOGY) fs = MORPHOLOGY_FRAGMENT;
      if (node.type === FilterType.BIT_PLANE) fs = BIT_PLANE_FRAGMENT;
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
      
      gl.uniform1i(imgLoc, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[sourceIdx]);
      
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

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      
      // Swap buffers
      count++;
    });

    // FINAL PASS: Draw the result to the screen (Canvas)
    // The latest result is in textures[count % 2]
    const finalSourceIdx = count % 2;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
    // gl.clear(gl.COLOR_BUFFER_BIT); // Optional, we overwrite anyway

    const program = this.getProgram(PASSTHROUGH_FRAGMENT);
    gl.useProgram(program);
    
    const posLoc = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const imgLoc = gl.getUniformLocation(program, "u_image");
    gl.uniform1i(imgLoc, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures[finalSourceIdx]);
    
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  getPixels(): Uint8Array {
      const w = this.gl.drawingBufferWidth;
      const h = this.gl.drawingBufferHeight;
      // Note: reading pixels is slow. In a production app, we would use a PBO or read only a small area for the probe.
      const pixels = new Uint8Array(w * h * 4);
      this.gl.readPixels(0, 0, w, h, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
      
      // WebGL reads pixels upside down relative to Canvas 2D ImageData.
      // We need to flip them for the Probe/Histogram to be correct relative to pointer events.
      // Optimization: We could do this only for the probed area, but for simplicity here we flip all.
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
