import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { 
  Upload, Camera, Download, Zap, Eye, RefreshCcw, BarChart, 
  ArrowRightLeft, Grid3X3, Droplet, Binary, Palette, Code, 
  Plus, Trash2, GripVertical, ChevronDown, ChevronRight, Play, Square, Layers
} from 'lucide-react';
import { KERNELS, INITIAL_CUSTOM_KERNEL, PROBE_SIZE, SAMPLE_IMAGE_URL } from './constants';
import { 
  Kernel, DisplayMode, PixelData, LabMode, 
  MorphologyType, ColorChannel, HistogramData,
  FilterNode, FilterType, FilterParams
} from './types';
import { computeHistogram } from './utils/imageProcessing'; // CPU Analysis
import { WebGLEngine } from './utils/webgl'; // GPU Processing
import { DEFAULT_CUSTOM_SHADER } from './utils/shaders';
import MatrixView from './components/MatrixView';
import Histogram from './components/Histogram';
import CustomKernelLab from './components/CustomKernelLab';

const App: React.FC = () => {
  // --- STATE: IMAGE SOURCE ---
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [imageSrc, setImageSrc] = useState<string>(SAMPLE_IMAGE_URL);
  
  // --- STATE: PIPELINE (Phase 3 Core) ---
  // We use a list of FilterNodes instead of single states
  const [pipeline, setPipeline] = useState<FilterNode[]>([
    { 
      id: 'base-kernel', 
      name: 'Gaussian Blur', 
      type: FilterType.KERNEL, 
      expanded: true,
      params: { 
        active: true,
        kernelMatrix: KERNELS['gaussianBlur3'].matrix, 
        kernelWeight: 1/KERNELS['gaussianBlur3'].factor!
      } 
    }
  ]);
  const [selectedNodeId, setSelectedNodeId] = useState<string>('base-kernel');

  // --- STATE: ANALYTICS (Phase 1 & 2) ---
  const [displayMode, setDisplayMode] = useState<DisplayMode>(DisplayMode.RGB);
  const [probeData, setProbeData] = useState<PixelData[][]>(
    Array(PROBE_SIZE).fill(Array(PROBE_SIZE).fill({ r: 0, g: 0, b: 0, a: 0 }))
  );
  const [histogramData, setHistogramData] = useState<HistogramData | null>(null);
  
  // --- REFS ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<WebGLEngine | null>(null);
  const requestRef = useRef<number>(0);
  const imageObjRef = useRef<HTMLImageElement | null>(null);
  const pixelsRef = useRef<Uint8Array | null>(null); // Cache CPU pixels for probe

  // --- HELPERS ---
  const updateNodeParams = (id: string, newParams: Partial<FilterParams>) => {
    setPipeline(prev => prev.map(node => 
      node.id === id ? { ...node, params: { ...node.params, ...newParams } } : node
    ));
  };

  const toggleNodeActive = (id: string) => {
    setPipeline(prev => prev.map(node => 
      node.id === id ? { ...node, params: { ...node.params, active: !node.params.active } } : node
    ));
  };

  const toggleNodeExpanded = (id: string) => {
    setPipeline(prev => prev.map(node => 
      node.id === id ? { ...node, expanded: !node.expanded } : node
    ));
    setSelectedNodeId(id);
  };

  const removeNode = (id: string) => {
    setPipeline(prev => prev.filter(n => n.id !== id));
    if (selectedNodeId === id) setSelectedNodeId('');
  };

  const addNode = (type: FilterType) => {
    const id = `node-${Date.now()}`;
    const newNode: FilterNode = {
      id,
      name: type === FilterType.CUSTOM_GLSL ? 'Custom Shader' : 
            type === FilterType.MORPHOLOGY ? 'Morphology' : 
            type === FilterType.BIT_PLANE ? 'Bit Plane' : 'Kernel Filter',
      type,
      expanded: true,
      params: {
        active: true,
        // Defaults
        kernelMatrix: KERNELS['identity'].matrix,
        kernelWeight: 1,
        morphType: MorphologyType.DILATION,
        morphRadius: 1,
        bitPlane: 8,
        customSource: DEFAULT_CUSTOM_SHADER
      }
    };
    setPipeline(prev => [...prev, newNode]);
    setSelectedNodeId(id);
  };

  // --- WEBCAM & IMAGE LOADING ---
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.src = imageSrc;
    img.onload = () => {
      imageObjRef.current = img;
      if (engineRef.current) {
        engineRef.current.resize(img.width, img.height);
        engineRef.current.loadImage(img);
      }
    };
  }, [imageSrc]);

  // --- RENDER LOOP ---
  const renderLoop = useCallback(() => {
    const engine = engineRef.current;
    if (!engine) return;

    // 1. Load Source (Video or Image)
    if (isWebcamActive && videoRef.current && videoRef.current.readyState >= 2) {
      const v = videoRef.current;
      if (engine.width !== v.videoWidth || engine.height !== v.videoHeight) {
        engine.resize(v.videoWidth, v.videoHeight);
      }
      engine.loadImage(v);
    } else if (!isWebcamActive && imageObjRef.current) {
      // For static images, we load once in useEffect, but if we need to ensure resize:
      const img = imageObjRef.current;
      if (engine.width !== img.width || engine.height !== img.height) {
        engine.resize(img.width, img.height);
        engine.loadImage(img);
      }
    }

    // 2. Execute WebGL Pipeline
    engine.renderPipeline(pipeline);

    // 3. Analytics Sync (ReadPixels)
    // We only read pixels if we need to update UI (Histogram or Probe)
    // To save performance, we could throttle this, but for "Lab" accuracy we do it.
    const rawPixels = engine.getPixels();
    pixelsRef.current = rawPixels;

    // Calculate Histogram
    // We create a fake ImageData to reuse the util function
    const w = engine.width;
    const h = engine.height;
    if (w > 0 && h > 0) {
      const imageData = new ImageData(new Uint8ClampedArray(rawPixels), w, h);
      setHistogramData(computeHistogram(imageData));
    }

    if (isWebcamActive) {
      requestRef.current = requestAnimationFrame(renderLoop);
    }
  }, [pipeline, isWebcamActive]);

  // --- INIT ENGINE ---
  useEffect(() => {
    if (canvasRef.current && !engineRef.current) {
      try {
        engineRef.current = new WebGLEngine(canvasRef.current);
        // Force initial render
        if (imageObjRef.current) {
            engineRef.current.resize(imageObjRef.current.width, imageObjRef.current.height);
            engineRef.current.loadImage(imageObjRef.current);
            requestAnimationFrame(renderLoop);
        }
      } catch (e) {
        console.error("WebGL Init Failed", e);
      }
    }
  }, [renderLoop]);

  // Handle Pipeline Changes (Trigger re-render if static)
  useEffect(() => {
    if (!isWebcamActive) {
      requestRef.current = requestAnimationFrame(renderLoop);
    }
  }, [pipeline, renderLoop, isWebcamActive]);

  // --- PIXEL PROBE HANDLER ---
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !pixelsRef.current || !engineRef.current) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));
    
    const w = engineRef.current.width;
    const h = engineRef.current.height;
    const pixels = pixelsRef.current;
    
    const probePixels: PixelData[][] = [];
    const half = Math.floor(PROBE_SIZE / 2);

    for (let i = 0; i < PROBE_SIZE; i++) {
      const rowArr: PixelData[] = [];
      for (let j = 0; j < PROBE_SIZE; j++) {
        const px = x + j - half;
        const py = y + i - half;
        
        if (px >= 0 && px < w && py >= 0 && py < h) {
          const idx = (py * w + px) * 4;
          rowArr.push({
            r: pixels[idx],
            g: pixels[idx+1],
            b: pixels[idx+2],
            a: pixels[idx+3]
          });
        } else {
          rowArr.push({ r:0, g:0, b:0, a:0 });
        }
      }
      probePixels.push(rowArr);
    }
    setProbeData(probePixels);

  }, []);

  // --- UI HELPERS ---
  const toggleWebcam = async () => {
    if (isWebcamActive) {
      setIsWebcamActive(false);
      const tracks = (videoRef.current?.srcObject as MediaStream)?.getTracks();
      tracks?.forEach(t => t.stop());
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setIsWebcamActive(true);
          requestRef.current = requestAnimationFrame(renderLoop);
        }
      } catch (err) {
        alert("Webcam access denied");
      }
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        if (evt.target?.result) {
          setImageSrc(evt.target.result as string);
          setIsWebcamActive(false);
        }
      };
      reader.readAsDataURL(file);
    }
    e.target.value = '';
  };

  return (
    <div className="h-screen bg-[#0f172a] text-slate-200 overflow-hidden font-sans flex flex-col">
      
      {/* HEADER */}
      <header className="h-14 shrink-0 border-b border-glassBorder bg-slate-900/50 backdrop-blur-md flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-2">
          <Eye className="text-neonCyan w-6 h-6" />
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-neonCyan to-neonIndigo bg-clip-text text-transparent">
            VisionLab v3 <span className="text-xs text-slate-500 font-mono ml-2">WebGL ACCELERATED</span>
          </h1>
        </div>
        <div className="flex items-center gap-3">
            <button onClick={() => setPipeline([])} className="p-2 rounded hover:bg-slate-800 text-slate-400">
               <RefreshCcw className="w-4 h-4" />
            </button>
            <label className="flex items-center justify-center p-2 bg-slate-800 rounded cursor-pointer hover:bg-slate-700">
                <Upload className="w-4 h-4 text-slate-400" />
                <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
            </label>
            <button onClick={toggleWebcam} className={`p-2 rounded ${isWebcamActive ? 'bg-red-500/20 text-red-400' : 'bg-slate-800 text-slate-400'}`}>
               <Camera className="w-4 h-4" />
            </button>
            <button 
              onClick={() => {
                const link = document.createElement('a');
                link.download = `vision-lab-result.png`;
                link.href = canvasRef.current?.toDataURL() || '';
                link.click();
              }}
              className="px-3 py-1.5 rounded bg-neonCyan/10 text-neonCyan text-xs flex items-center gap-2 border border-neonCyan/20"
            >
              <Download className="w-3 h-3" /> Export
            </button>
            <video ref={videoRef} className="hidden" playsInline muted />
        </div>
      </header>

      {/* MAIN CONTENT GRID */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* LEFT: PIPELINE EDITOR */}
        <aside className="w-80 bg-slate-900/80 border-r border-glassBorder flex flex-col z-20">
          <div className="p-4 border-b border-glassBorder flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Layers className="w-3 h-3 text-neonIndigo" /> Filter Chain
            </h2>
            <div className="flex gap-1">
               <button onClick={() => addNode(FilterType.KERNEL)} className="p-1 hover:bg-slate-700 rounded" title="Add Kernel"><Grid3X3 className="w-3 h-3 text-neonCyan"/></button>
               <button onClick={() => addNode(FilterType.MORPHOLOGY)} className="p-1 hover:bg-slate-700 rounded" title="Add Morphology"><Droplet className="w-3 h-3 text-neonIndigo"/></button>
               <button onClick={() => addNode(FilterType.BIT_PLANE)} className="p-1 hover:bg-slate-700 rounded" title="Add Bit Plane"><Binary className="w-3 h-3 text-green-400"/></button>
               <button onClick={() => addNode(FilterType.CUSTOM_GLSL)} className="p-1 hover:bg-slate-700 rounded" title="Add Shader"><Code className="w-3 h-3 text-pink-400"/></button>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {pipeline.map((node, index) => (
              <div key={node.id} className={`rounded-lg border transition-all ${selectedNodeId === node.id ? 'bg-slate-800 border-neonCyan/50' : 'bg-slate-900 border-slate-700'}`}>
                {/* Node Header */}
                <div className="flex items-center p-2 cursor-pointer select-none" onClick={() => toggleNodeExpanded(node.id)}>
                   <GripVertical className="w-4 h-4 text-slate-600 mr-2" />
                   <div className="flex-1">
                     <div className="text-xs font-bold text-slate-200">{node.name}</div>
                     <div className="text-[9px] text-slate-500 uppercase">{node.type}</div>
                   </div>
                   <div className="flex items-center gap-2">
                      <button onClick={(e) => { e.stopPropagation(); toggleNodeActive(node.id); }}>
                        {node.params.active ? <Play className="w-3 h-3 text-green-400 fill-current"/> : <Square className="w-3 h-3 text-slate-600 fill-current"/>}
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); removeNode(node.id); }}>
                        <Trash2 className="w-3 h-3 text-slate-600 hover:text-red-400"/>
                      </button>
                      {node.expanded ? <ChevronDown className="w-3 h-3 text-slate-500"/> : <ChevronRight className="w-3 h-3 text-slate-500"/>}
                   </div>
                </div>

                {/* Node Controls */}
                {node.expanded && (
                  <div className="p-3 border-t border-slate-700 bg-black/20">
                    
                    {/* KERNEL CONTROLS */}
                    {node.type === FilterType.KERNEL && (
                      <div className="space-y-3">
                         <select 
                           className="w-full bg-slate-900 border border-slate-700 rounded text-xs p-1 text-slate-300"
                           onChange={(e) => {
                             const k = KERNELS[e.target.value];
                             if(k) updateNodeParams(node.id, { kernelMatrix: k.matrix, kernelWeight: 1/(k.factor || 1) });
                           }}
                         >
                            <option value="">Choose Preset...</option>
                            {Object.keys(KERNELS).map(k => <option key={k} value={k}>{KERNELS[k].name}</option>)}
                         </select>
                         <CustomKernelLab active={true} onKernelChange={(m) => updateNodeParams(node.id, { kernelMatrix: m, kernelWeight: 1 })} />
                      </div>
                    )}

                    {/* MORPHOLOGY CONTROLS */}
                    {node.type === FilterType.MORPHOLOGY && (
                      <div className="space-y-3">
                        <div className="flex gap-1 bg-slate-900 p-1 rounded">
                           <button onClick={() => updateNodeParams(node.id, { morphType: MorphologyType.DILATION })} className={`flex-1 py-1 text-[10px] rounded ${node.params.morphType === MorphologyType.DILATION ? 'bg-indigo-600' : ''}`}>Dilation</button>
                           <button onClick={() => updateNodeParams(node.id, { morphType: MorphologyType.EROSION })} className={`flex-1 py-1 text-[10px] rounded ${node.params.morphType === MorphologyType.EROSION ? 'bg-indigo-600' : ''}`}>Erosion</button>
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400">Radius: {node.params.morphRadius}px</label>
                          <input type="range" min="1" max="5" value={node.params.morphRadius} onChange={(e) => updateNodeParams(node.id, { morphRadius: parseInt(e.target.value) })} className="w-full accent-indigo-500 h-1 bg-slate-700 rounded cursor-pointer"/>
                        </div>
                      </div>
                    )}

                    {/* BIT PLANE CONTROLS */}
                    {node.type === FilterType.BIT_PLANE && (
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-2xl font-mono text-green-400">{node.params.bitPlane}</span>
                          <span className="text-[10px] text-slate-500">SIGNIFICANCE</span>
                        </div>
                        <input type="range" min="1" max="8" step="1" value={node.params.bitPlane} onChange={(e) => updateNodeParams(node.id, { bitPlane: parseInt(e.target.value) })} className="w-full accent-green-400 h-1 bg-slate-700 rounded cursor-pointer"/>
                      </div>
                    )}

                    {/* GLSL EDITOR (Mini) */}
                    {node.type === FilterType.CUSTOM_GLSL && (
                       <div className="space-y-2">
                         <p className="text-[10px] text-slate-500">Edit fragment shader main():</p>
                         <textarea 
                           className="w-full h-32 bg-slate-950 text-pink-300 font-mono text-[10px] p-2 rounded border border-slate-700 focus:border-pink-500 outline-none resize-none"
                           value={node.params.customSource}
                           onChange={(e) => updateNodeParams(node.id, { customSource: e.target.value })}
                           spellCheck={false}
                         />
                       </div>
                    )}

                  </div>
                )}
              </div>
            ))}
          </div>
        </aside>

        {/* CENTER: VIEWPORT */}
        <main className="flex-1 bg-slate-950 relative flex flex-col overflow-hidden">
          <div className="flex-1 relative flex items-center justify-center p-4 bg-[url('https://www.transparenttextures.com/patterns/dark-matter.png')]">
             <canvas 
               ref={canvasRef}
               onMouseMove={handleMouseMove}
               className="shadow-2xl shadow-neonIndigo/20 rounded-lg max-w-full max-h-[85vh] cursor-crosshair border border-slate-800"
             />
             {/* Overlay Histogram */}
             <div className="absolute bottom-6 right-6 w-64 h-32 pointer-events-none opacity-80">
                <Histogram data={histogramData} />
             </div>
          </div>
        </main>

        {/* RIGHT: PROBE & DATA */}
        <aside className="w-72 bg-slate-900/95 border-l border-glassBorder flex flex-col z-20">
          <div className="p-4 border-b border-glassBorder">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2 mb-3">
                <Zap className="w-3 h-3 text-yellow-400" /> Pixel Probe
              </h2>
              <div className="flex bg-slate-800 p-1 rounded-lg">
                 {(['RGB', 'HEX', 'GRAY'] as const).map(mode => (
                   <button key={mode} onClick={() => setDisplayMode(mode as DisplayMode)} className={`flex-1 py-1 text-[10px] font-bold rounded transition-all ${displayMode === mode ? 'bg-slate-600 text-white' : 'text-slate-400'}`}>
                     {mode}
                   </button>
                 ))}
              </div>
          </div>
          <div className="flex-1 relative bg-slate-900/50 p-2 overflow-hidden">
              <MatrixView data={probeData} mode={displayMode} />
          </div>
          {/* Expanded GLSL Editor Hint or Mini-Map could go here */}
          <div className="p-4 border-t border-glassBorder bg-slate-900 text-[10px] text-slate-500">
             <p>Phase 3 Architecture</p>
             <p>WebGL 1.0 • Ping-Pong FBO • Sync ReadPixels</p>
          </div>
        </aside>

      </div>
    </div>
  );
};

export default App;