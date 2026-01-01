import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { 
  Upload, Camera, Download, Zap, Eye, RefreshCcw, BarChart, 
  ArrowRightLeft, Grid3X3, Droplet, Binary, Palette, Code, 
  Plus, Trash2, GripVertical, ChevronDown, ChevronRight, Play, Square, Layers,
  Lightbulb, ScanText, Smile, Route, Maximize2, Columns
} from 'lucide-react';
import { KERNELS, PROBE_SIZE, SAMPLE_IMAGE_URL, DEFAULT_MOTION_THRESHOLD } from './constants';
import { 
  DisplayMode, PixelData, HistogramData,
  MorphologyType, 
  FilterNode, FilterType, FilterParams,
  DetectedObject, FaceLandmarkPoint, FaceMeshResult
} from './types';
import { computeHistogram } from './utils/imageProcessing'; // CPU Analysis
import { WebGLEngine } from './utils/webgl'; // GPU Processing
import { DEFAULT_CUSTOM_SHADER } from './utils/shaders';
import MatrixView from './components/MatrixView';
import Histogram from './components/Histogram';
import CustomKernelLab from './components/CustomKernelLab';

// Dynamic imports for TensorFlow.js models
let tf: typeof import('@tensorflow/tfjs') | undefined;

// Add a global declaration for window properties to resolve TypeScript errors
declare global {
  interface Window {
    tf: typeof import('@tensorflow/tfjs');
    cocoSsd: any;
    faceLandmarksDetection: any;
  }
}

const App: React.FC = () => {
  // --- STATE: IMAGE SOURCE ---
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [imageSrc, setImageSrc] = useState<string>(SAMPLE_IMAGE_URL);
  const [isCompareMode, setIsCompareMode] = useState(true);
  
  // --- STATE: PIPELINE (Phase 3 Core) ---
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

  // --- STATE: AI SUITE (Phase 4) ---
  const [tfjsReady, setTfjsReady] = useState(false);
  const [detectObjectsEnabled, setDetectObjectsEnabled] = useState(false);
  const [faceMeshEnabled, setFaceMeshEnabled] = useState(false);
  const [motionHeatmapEnabled, setMotionHeatmapEnabled] = useState(false);
  const [objectDetections, setObjectDetections] = useState<DetectedObject[]>([]);
  const [faceLandmarks, setFaceLandmarks] = useState<FaceMeshResult | null>(null);

  // --- REFS ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null); // Main WebGL output
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null); // For TF.js overlays
  const engineRef = useRef<WebGLEngine | null>(null);
  const requestRef = useRef<number>(0);
  const imageObjRef = useRef<HTMLImageElement | null>(null);
  const pixelsRef = useRef<Uint8Array | null>(null); // Cache CPU pixels for probe
  
  // TF.js Model Refs
  const cocoSsdModelRef = useRef<any | null>(null);
  const faceMeshModelRef = useRef<any | null>(null);

  const frameCounter = useRef<number>(0); // For throttling TF.js inference

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
            type === FilterType.BIT_PLANE ? 'Bit Plane' : 
            type === FilterType.MOTION_HEATMAP ? 'Motion Heatmap' : 'Kernel Filter',
      type,
      expanded: true,
      params: {
        active: true,
        kernelMatrix: KERNELS['identity'].matrix,
        kernelWeight: 1,
        morphType: MorphologyType.DILATION,
        morphRadius: 1,
        bitPlane: 8,
        customSource: DEFAULT_CUSTOM_SHADER,
        motionThreshold: DEFAULT_MOTION_THRESHOLD
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

  // --- PHASE 4: TF.js MODEL LOADING ---
  useEffect(() => {
    const loadTfjsModels = async () => {
      try {
        await import('@tensorflow/tfjs');
        await import('@tensorflow/tfjs-backend-webgl');

        if (window.tf) {
          tf = window.tf;
          await tf.setBackend('webgl');
          await tf.ready();
          
          await import('@tensorflow-models/coco-ssd');
          const cocoSsd = window.cocoSsd;
          if (cocoSsd && typeof cocoSsd.load === 'function') {
            cocoSsdModelRef.current = await cocoSsd.load();
          }

          await import('@tensorflow-models/face-landmarks-detection');
          const faceLandmarksDetection = window.faceLandmarksDetection;
          if (faceLandmarksDetection && typeof faceLandmarksDetection.load === 'function') {
            faceMeshModelRef.current = await faceLandmarksDetection.load(
              faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
              { maxFaces: 1 }
            );
          }
          setTfjsReady(true);
        }
      } catch (error) {
        console.error("Failed to load AI suite:", error);
        setTfjsReady(false);
      }
    };
    loadTfjsModels();
  }, []);

  // --- RENDER LOOP ---
  const renderLoop = useCallback(async () => {
    const engine = engineRef.current;
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;

    if (!engine || !canvas || !overlayCanvas) return;

    let sourceElement: HTMLVideoElement | HTMLImageElement | null = null;
    if (isWebcamActive && videoRef.current && videoRef.current.readyState >= 2) {
      const v = videoRef.current;
      if (engine.width !== v.videoWidth || engine.height !== v.videoHeight) {
        engine.resize(v.videoWidth, v.videoHeight);
        overlayCanvas.width = v.videoWidth;
        overlayCanvas.height = v.videoHeight;
      }
      engine.loadImage(v);
      sourceElement = v;
    } else if (!isWebcamActive && imageObjRef.current) {
      const img = imageObjRef.current;
      if (engine.width !== img.width || engine.height !== img.height) {
        engine.resize(img.width, img.height);
        overlayCanvas.width = img.width;
        overlayCanvas.height = img.height;
        engine.loadImage(img);
      }
      sourceElement = img;
    } else {
      if (!isWebcamActive) {
        cancelAnimationFrame(requestRef.current);
        return;
      }
    }

    engine.renderPipeline(pipeline);
    const rawPixels = engine.getPixels();
    pixelsRef.current = rawPixels;

    if (engine.width > 0 && engine.height > 0) {
      const imageData = new ImageData(new Uint8ClampedArray(rawPixels), engine.width, engine.height);
      setHistogramData(computeHistogram(imageData));
    }

    const overlayCtx = overlayCanvas.getContext('2d');
    if (overlayCtx) {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      if (tfjsReady && sourceElement) {
        frameCounter.current++;
        if (frameCounter.current % 5 === 0) {
          frameCounter.current = 0;
          if (detectObjectsEnabled && cocoSsdModelRef.current) {
            try {
              const predictions = await cocoSsdModelRef.current.detect(canvas);
              setObjectDetections(predictions);
            } catch (e) {}
          }
          if (faceMeshEnabled && faceMeshModelRef.current && isWebcamActive) {
            try {
              const faces = await faceMeshModelRef.current.estimateFaces({ input: videoRef.current });
              if (faces.length > 0) {
                const face = faces[0];
                setFaceLandmarks({
                  faceContours: {
                    lips: face.mesh.slice(0, 68).map((p: any) => ({ x: p[0], y: p[1], z: p[2] })),
                    leftEye: face.annotations.leftEyeIris?.map((p: any) => ({ x: p[0], y: p[1], z: p[2] })) || [],
                    rightEye: face.annotations.rightEyeIris?.map((p: any) => ({ x: p[0], y: p[1], z: p[2] })) || [],
                    leftEyebrow: face.annotations.leftEyebrow?.map((p: any) => ({ x: p[0], y: p[1], z: p[2] })) || [],
                    rightEyebrow: face.annotations.rightEyebrow?.map((p: any) => ({ x: p[0], y: p[1], z: p[2] })) || [],
                    faceOval: face.annotations.faceOval?.map((p: any) => ({ x: p[0], y: p[1], z: p[2] })) || [],
                  }
                });
              } else { setFaceLandmarks(null); }
            } catch (e) {}
          }
        }
        overlayCtx.strokeStyle = 'cyan';
        overlayCtx.lineWidth = 2;
        overlayCtx.fillStyle = 'cyan';
        overlayCtx.font = '12px monospace';
        if (detectObjectsEnabled) {
          objectDetections.forEach(p => {
            const [x, y, w, h] = p.bbox;
            overlayCtx.strokeRect(x, y, w, h);
            overlayCtx.fillText(`${p.class} ${Math.round(p.score * 100)}%`, x, y > 15 ? y - 5 : 15);
          });
        }
        if (faceMeshEnabled && faceLandmarks) {
          overlayCtx.strokeStyle = 'lime';
          overlayCtx.lineWidth = 1;
          Object.values(faceLandmarks.faceContours).forEach((points: any) => {
            if (points.length < 2) return;
            overlayCtx.beginPath();
            overlayCtx.moveTo(points[0].x, points[0].y);
            points.slice(1).forEach((p: any) => overlayCtx.lineTo(p.x, p.y));
            overlayCtx.stroke();
          });
        }
      }
    }

    if (isWebcamActive) {
      requestRef.current = requestAnimationFrame(renderLoop);
    }
  }, [pipeline, isWebcamActive, tfjsReady, detectObjectsEnabled, faceMeshEnabled, objectDetections, faceLandmarks]);

  useEffect(() => {
    if (canvasRef.current && !engineRef.current) {
      engineRef.current = new WebGLEngine(canvasRef.current);
      if (imageObjRef.current) {
        engineRef.current.resize(imageObjRef.current.width, imageObjRef.current.height);
        if (overlayCanvasRef.current) {
          overlayCanvasRef.current.width = imageObjRef.current.width;
          overlayCanvasRef.current.height = imageObjRef.current.height;
        }
        engineRef.current.loadImage(imageObjRef.current);
        requestAnimationFrame(renderLoop);
      }
    }
    return () => cancelAnimationFrame(requestRef.current);
  }, [renderLoop]);

  useEffect(() => {
    if (!isWebcamActive) requestRef.current = requestAnimationFrame(renderLoop);
  }, [pipeline, renderLoop, isWebcamActive]);

  useEffect(() => {
    setPipeline(prev => {
      const hasMotion = prev.some(n => n.type === FilterType.MOTION_HEATMAP);
      if (motionHeatmapEnabled && !hasMotion) {
        return [...prev, {
          id: 'motion-heatmap',
          name: 'Motion Heatmap',
          type: FilterType.MOTION_HEATMAP,
          expanded: true,
          params: { active: true, motionThreshold: DEFAULT_MOTION_THRESHOLD }
        }];
      } else if (!motionHeatmapEnabled && hasMotion) {
        return prev.filter(n => n.type !== FilterType.MOTION_HEATMAP);
      }
      return prev;
    });
  }, [motionHeatmapEnabled]);

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
          rowArr.push({ r: pixels[idx], g: pixels[idx+1], b: pixels[idx+2], a: pixels[idx+3] });
        } else { rowArr.push({ r:0, g:0, b:0, a:0 }); }
      }
      probePixels.push(rowArr);
    }
    setProbeData(probePixels);
  }, []);

  const toggleWebcam = async () => {
    if (isWebcamActive) {
      setIsWebcamActive(false);
      (videoRef.current?.srcObject as MediaStream)?.getTracks().forEach(t => t.stop());
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setIsWebcamActive(true);
          requestRef.current = requestAnimationFrame(renderLoop);
        }
      } catch (err) { alert("Webcam access denied."); }
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
      <header className="h-14 shrink-0 border-b border-glassBorder bg-slate-900/50 backdrop-blur-md flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-2">
          <Eye className="text-neonCyan w-6 h-6" />
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-neonCyan to-neonIndigo bg-clip-text text-transparent">
            VisionLab v4 <span className="text-xs text-slate-500 font-mono ml-2 uppercase tracking-widest">Interactive</span>
          </h1>
        </div>
        <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsCompareMode(!isCompareMode)} 
              className={`px-3 py-1.5 rounded text-xs flex items-center gap-2 border transition-colors ${isCompareMode ? 'bg-neonIndigo/20 text-neonIndigo border-neonIndigo/40' : 'bg-slate-800 text-slate-400 border-slate-700'}`}
              title="Toggle Compare Mode"
            >
              <Columns className="w-4 h-4" /> Compare
            </button>
            <button onClick={() => setPipeline([])} className="p-2 rounded hover:bg-slate-800 text-slate-400" title="Reset Pipeline"><RefreshCcw className="w-4 h-4" /></button>
            <label className="flex items-center justify-center p-2 bg-slate-800 rounded cursor-pointer hover:bg-slate-700" title="Upload Image">
                <Upload className="w-4 h-4 text-slate-400" />
                <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
            </label>
            <button onClick={toggleWebcam} className={`p-2 rounded ${isWebcamActive ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-slate-800 text-slate-400 border border-transparent'}`} title="Toggle Webcam"><Camera className="w-4 h-4" /></button>
            <button onClick={() => {
                const link = document.createElement('a');
                link.download = `vision-lab-v4.png`;
                link.href = canvasRef.current?.toDataURL() || '';
                link.click();
              }} className="px-3 py-1.5 rounded bg-neonCyan/10 text-neonCyan text-xs flex items-center gap-2 border border-neonCyan/20"><Download className="w-3 h-3" /> Export</button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* LEFT SIDEBAR: PIPELINE */}
        <aside className="w-80 bg-slate-900/80 border-r border-glassBorder flex flex-col z-20">
          <div className="p-4 border-b border-glassBorder flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2"><Layers className="w-3 h-3 text-neonIndigo" /> Pipeline</h2>
            <div className="flex gap-1">
               <button onClick={() => addNode(FilterType.KERNEL)} className="p-1 hover:bg-slate-700 rounded" title="Add Kernel"><Grid3X3 className="w-3 h-3 text-neonCyan"/></button>
               <button onClick={() => addNode(FilterType.MORPHOLOGY)} className="p-1 hover:bg-slate-700 rounded" title="Add Morphology"><Droplet className="w-3 h-3 text-neonIndigo"/></button>
               <button onClick={() => addNode(FilterType.BIT_PLANE)} className="p-1 hover:bg-slate-700 rounded" title="Add Bit Plane"><Binary className="w-3 h-3 text-green-400"/></button>
               <button onClick={() => addNode(FilterType.CUSTOM_GLSL)} className="p-1 hover:bg-slate-700 rounded" title="Add Custom Shader"><Code className="w-3 h-3 text-pink-400"/></button>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto p-2 space-y-2 scrollbar-hide">
            {pipeline.map((node) => (
              <div key={node.id} className={`rounded-lg border transition-all ${selectedNodeId === node.id ? 'bg-slate-800 border-neonCyan/50 shadow-lg shadow-neonCyan/5' : 'bg-slate-900 border-slate-700'}`}>
                <div className="flex items-center p-2 cursor-pointer" onClick={() => toggleNodeExpanded(node.id)}>
                   <GripVertical className="w-4 h-4 text-slate-600 mr-2" />
                   <div className="flex-1">
                     <div className="text-xs font-bold text-slate-200">{node.name}</div>
                     <div className="text-[9px] text-slate-500 uppercase font-mono tracking-tighter">{node.type}</div>
                   </div>
                   <div className="flex items-center gap-2">
                      <button onClick={(e) => { e.stopPropagation(); toggleNodeActive(node.id); }}>
                        {node.params.active ? <Play className="w-3 h-3 text-green-400 fill-current"/> : <Square className="w-3 h-3 text-slate-600 fill-current"/>}
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); removeNode(node.id); }}><Trash2 className="w-3 h-3 text-slate-600 hover:text-red-400"/></button>
                      {node.expanded ? <ChevronDown className="w-3 h-3 text-slate-500"/> : <ChevronRight className="w-3 h-3 text-slate-500"/>}
                   </div>
                </div>

                {node.expanded && (
                  <div className="p-3 border-t border-slate-700 bg-black/20">
                    {node.type === FilterType.KERNEL && (
                      <div className="space-y-3">
                         <select className="w-full bg-slate-900 border border-slate-700 rounded text-xs p-1 text-slate-300" onChange={(e) => {
                             const k = KERNELS[e.target.value];
                             if(k) updateNodeParams(node.id, { kernelMatrix: k.matrix, kernelWeight: 1/(k.factor || 1) });
                           }}>
                            <option value="">Choose Preset...</option>
                            {Object.keys(KERNELS).map(k => <option key={k} value={k}>{KERNELS[k].name}</option>)}
                         </select>
                         <CustomKernelLab active={true} onKernelChange={(m) => updateNodeParams(node.id, { kernelMatrix: m, kernelWeight: 1 })} />
                      </div>
                    )}
                    {node.type === FilterType.MORPHOLOGY && (
                      <div className="space-y-3">
                        <div className="flex gap-1 bg-slate-900 p-1 rounded">
                           <button onClick={() => updateNodeParams(node.id, { morphType: MorphologyType.DILATION })} className={`flex-1 py-1 text-[10px] rounded ${node.params.morphType === MorphologyType.DILATION ? 'bg-indigo-600' : ''}`}>Dilation</button>
                           <button onClick={() => updateNodeParams(node.id, { morphType: MorphologyType.EROSION })} className={`flex-1 py-1 text-[10px] rounded ${node.params.morphType === MorphologyType.EROSION ? 'bg-indigo-600' : ''}`}>Erosion</button>
                        </div>
                        <input type="range" min="1" max="5" value={node.params.morphRadius} onChange={(e) => updateNodeParams(node.id, { morphRadius: parseInt(e.target.value) })} className="w-full accent-indigo-500 h-1 bg-slate-700 rounded cursor-pointer"/>
                      </div>
                    )}
                    {node.type === FilterType.BIT_PLANE && (
                      <input type="range" min="1" max="8" value={node.params.bitPlane} onChange={(e) => updateNodeParams(node.id, { bitPlane: parseInt(e.target.value) })} className="w-full accent-green-400 h-1 bg-slate-700 rounded cursor-pointer"/>
                    )}
                    {node.type === FilterType.CUSTOM_GLSL && (
                       <textarea className="w-full h-32 bg-slate-950 text-pink-300 font-mono text-[10px] p-2 rounded border border-slate-700 outline-none resize-none" value={node.params.customSource} onChange={(e) => updateNodeParams(node.id, { customSource: e.target.value })} />
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="p-4 border-t border-glassBorder bg-slate-900/50">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2 mb-3"><Lightbulb className="w-3 h-3 text-yellow-400" /> AI Suite</h2>
            <div className="space-y-3">
               <label className="flex items-center justify-between text-xs cursor-pointer">
                 <span>Object Detection</span>
                 <input type="checkbox" checked={detectObjectsEnabled} onChange={() => setDetectObjectsEnabled(!detectObjectsEnabled)} disabled={!tfjsReady} className="w-4 h-4 accent-neonCyan" />
               </label>
               <label className="flex items-center justify-between text-xs cursor-pointer">
                 <span>Face Mesh Tracking</span>
                 <input type="checkbox" checked={faceMeshEnabled} onChange={() => setFaceMeshEnabled(!faceMeshEnabled)} disabled={!tfjsReady || !isWebcamActive} className="w-4 h-4 accent-lime-500" />
               </label>
               <label className="flex items-center justify-between text-xs cursor-pointer">
                 <span>Motion Heatmap</span>
                 <input type="checkbox" checked={motionHeatmapEnabled} onChange={() => setMotionHeatmapEnabled(!motionHeatmapEnabled)} className="w-4 h-4 accent-teal-400" />
               </label>
            </div>
          </div>
        </aside>

        {/* CENTER VIEWPORT: DUAL MODE */}
        <main className="flex-1 bg-slate-950 relative flex flex-col p-4 gap-4 overflow-hidden">
           <div className={`flex-1 grid gap-4 ${isCompareMode ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
              
              {/* SOURCE PANEL */}
              {isCompareMode && (
                <div className="relative bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center shadow-inner group">
                  <div className="absolute top-3 left-3 z-30 px-2 py-0.5 bg-slate-800/80 rounded text-[10px] font-bold text-slate-400 border border-slate-700 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full"/> SOURCE
                  </div>
                  
                  {isWebcamActive ? (
                    <video 
                      ref={videoRef} 
                      className="max-w-full max-h-full object-contain"
                      playsInline 
                      muted 
                    />
                  ) : (
                    <img 
                      src={imageSrc} 
                      className="max-w-full max-h-full object-contain"
                      alt="Input Source"
                    />
                  )}
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/40 to-transparent pointer-events-none" />
                </div>
              )}

              {/* RESULT PANEL */}
              <div className="relative bg-slate-900 rounded-xl border border-neonCyan/20 overflow-hidden flex items-center justify-center shadow-lg group">
                <div className="absolute top-3 left-3 z-30 px-2 py-0.5 bg-neonCyan/10 rounded text-[10px] font-bold text-neonCyan border border-neonCyan/40 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neonCyan rounded-full animate-pulse"/> RESULT
                </div>
                
                <div className="relative w-full h-full flex items-center justify-center p-2">
                  <canvas 
                    ref={canvasRef} 
                    onMouseMove={handleMouseMove} 
                    className="shadow-2xl rounded-lg max-w-full max-h-full border border-slate-800 cursor-crosshair z-10" 
                  />
                  <canvas 
                    ref={overlayCanvasRef} 
                    className="absolute pointer-events-none z-20" 
                    style={{ 
                      width: canvasRef.current?.getBoundingClientRect().width, 
                      height: canvasRef.current?.getBoundingClientRect().height 
                    }} 
                  />
                </div>

                <div className="absolute bottom-6 right-6 w-48 lg:w-64 h-24 lg:h-32 pointer-events-none opacity-80 z-30">
                  <Histogram data={histogramData} />
                </div>
              </div>
           </div>
        </main>

        {/* RIGHT ASIDE: TOOLS */}
        <aside className="w-72 bg-slate-900/95 border-l border-glassBorder flex flex-col z-20">
          <div className="p-4 border-b border-glassBorder">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2 mb-3"><Zap className="w-3 h-3 text-yellow-400" /> Pixel Probe</h2>
              <div className="flex bg-slate-800 p-1 rounded-lg">
                 {(['RGB', 'HEX', 'GRAY'] as const).map(mode => (
                   <button key={mode} onClick={() => setDisplayMode(mode as DisplayMode)} className={`flex-1 py-1 text-[10px] font-bold rounded ${displayMode === mode ? 'bg-slate-600 text-white shadow' : 'text-slate-400'}`}>{mode}</button>
                 ))}
              </div>
          </div>
          <div className="flex-1 p-2 overflow-hidden flex items-center justify-center">
            <MatrixView data={probeData} mode={displayMode} />
          </div>
          <div className="p-4 border-t border-glassBorder bg-slate-900">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2 mb-2"><ScanText className="w-3 h-3 text-neonCyan" /> AI Logs</h2>
            <div className="text-[10px] font-mono text-slate-500 space-y-1 h-32 overflow-y-auto scrollbar-hide">
               {!tfjsReady && <div className="animate-pulse">Initializing AI Engine...</div>}
               {tfjsReady && <div className="text-green-500/60">TensorFlow.js WebGL Ready</div>}
               {objectDetections.length > 0 && objectDetections.map((o, i) => <div key={i} className="text-neonCyan">&gt; DETECTED: {o.class} ({Math.round(o.score*100)}%)</div>)}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default App;