"use client";

import { useState, useCallback } from "react";
import { Upload, Activity, Layers, Download, Eye, EyeOff, BarChart3, MapPin } from "lucide-react";
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';

const API_URL = "http://localhost:8000";

export default function Home() {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [previewA, setPreviewA] = useState<string | null>(null);
  const [previewB, setPreviewB] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Results
  const [maskBase64, setMaskBase64] = useState<string | null>(null);
  const [overlayBase64, setOverlayBase64] = useState<string | null>(null);
  const [heatmapBase64, setHeatmapBase64] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [summary, setSummary] = useState<string | null>(null);

  // UI toggles
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(80);
  const [activeView, setActiveView] = useState<"slider" | "mask" | "heatmap">("slider");

  // Drag state
  const [dragA, setDragA] = useState(false);
  const [dragB, setDragB] = useState(false);

  const handleFile = (file: File, target: "A" | "B") => {
    const url = URL.createObjectURL(file);
    if (target === "A") {
      setFileA(file);
      setPreviewA(url);
    } else {
      setFileB(file);
      setPreviewB(url);
    }
    // Reset results
    setMaskBase64(null);
    setOverlayBase64(null);
    setHeatmapBase64(null);
    setStats(null);
    setSummary(null);
    setError(null);
  };

  const handleDrop = useCallback((e: React.DragEvent, target: "A" | "B") => {
    e.preventDefault();
    if (target === "A") setDragA(false);
    else setDragB(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, target);
  }, []);

  const runAnalysis = async () => {
    if (!fileA || !fileB) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file_A", fileA);
      formData.append("file_B", fileB);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction failed");
      }

      const data = await res.json();

      setMaskBase64(data.mask_base64);
      setOverlayBase64(data.overlay_base64);
      setHeatmapBase64(data.heatmap_base64);
      setStats(data.stats);
      setSummary(data.summary);
    } catch (err: any) {
      setError(err.message || "Failed to connect to API. Make sure the backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const downloadMask = () => {
    if (!maskBase64) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${maskBase64}`;
    link.download = "change_mask.png";
    link.click();
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-12 flex flex-col items-center">
      
      {/* Hero Section */}
      <div className="text-center mb-12 space-y-4">
        <h2 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300 drop-shadow-sm">
          Bi-Temporal Cadastral Monitoring
        </h2>
        <p className="text-gray-400 max-w-2xl mx-auto text-lg">
          Upload historical and recent satellite patches. Our AI model will extract illegal construction, structural demolition, and urban expansion instantly.
        </p>
      </div>

      {/* Upload Zone */}
      <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Upload A */}
        <div
          className={`glass-panel p-6 flex flex-col items-center justify-center min-h-[300px] relative transition-all ${dragA ? "ring-2 ring-blue-400 scale-[1.02]" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragA(true); }}
          onDragLeave={() => setDragA(false)}
          onDrop={(e) => handleDrop(e, "A")}
        >
          {previewA ? (
            <img src={previewA} className="absolute inset-0 w-full h-full object-contain rounded-xl opacity-90 pointer-events-none" alt="Before" />
          ) : (
            <>
              <div className="w-16 h-16 rounded-full bg-blue-500/10 flex items-center justify-center mb-4">
                <Upload className="text-blue-400 w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold">Time T1 (Before)</h3>
              <p className="text-sm text-gray-400 mt-2">Drop or click to upload base imagery</p>
            </>
          )}
          <input
            id="upload-before"
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0], "A")}
            onDragOver={(e) => { e.preventDefault(); setDragA(true); }}
            onDragLeave={() => setDragA(false)}
            onDrop={(e) => handleDrop(e, "A")}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
        </div>

        {/* Upload B */}
        <div
          className={`glass-panel p-6 flex flex-col items-center justify-center min-h-[300px] relative transition-all ${dragB ? "ring-2 ring-pink-400 scale-[1.02]" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragB(true); }}
          onDragLeave={() => setDragB(false)}
          onDrop={(e) => handleDrop(e, "B")}
        >
          {previewB ? (
            <img src={previewB} className="absolute inset-0 w-full h-full object-contain rounded-xl opacity-90 pointer-events-none" alt="After" />
          ) : (
            <>
              <div className="w-16 h-16 rounded-full bg-pink-500/10 flex items-center justify-center mb-4">
                <Layers className="text-pink-400 w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold">Time T2 (After)</h3>
              <p className="text-sm text-gray-400 mt-2">Drop or click to upload recent imagery</p>
            </>
          )}
          <input
            id="upload-after"
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0], "B")}
            onDragOver={(e) => { e.preventDefault(); setDragB(true); }}
            onDragLeave={() => setDragB(false)}
            onDrop={(e) => handleDrop(e, "B")}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
        </div>
      </div>

      {/* Action Button */}
      <button
        id="run-analysis-btn"
        onClick={runAnalysis}
        disabled={!fileA || !fileB || loading}
        className="glass-panel px-10 py-4 font-bold text-lg hover:bg-primary/20 transition-all font-mono tracking-widest disabled:opacity-50 flex items-center space-x-3"
      >
        {loading ? (
          <><Activity className="w-5 h-5 animate-spin" /> <span>ANALYZING TENSORS...</span></>
        ) : (
          <span>INITIATE AI EXTRACTION</span>
        )}
      </button>

      {/* Error State */}
      {error && (
        <div className="w-full mt-6 glass-panel p-4 border-l-4 border-l-red-500 text-red-300">
          <p className="font-semibold">Error</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}

      {/* ─── RESULTS ──────────────────────────────── */}
      {stats && (
        <div className="w-full mt-16 animate-in fade-in slide-in-from-bottom-8 duration-700">
          <h3 className="text-2xl font-bold mb-2 flex items-center">
            <MapPin className="mr-3 text-red-500" /> AI Detection Results
          </h3>

          {/* Summary */}
          {summary && (
            <p className="text-gray-400 mb-6 text-sm italic">&quot;{summary}&quot;</p>
          )}

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="glass-panel p-6 border-l-4 border-l-red-500">
              <p className="text-gray-400 text-sm uppercase tracking-wider">Changed Area</p>
              <p className="text-4xl font-bold mt-2">{stats.changed_area_percentage}%</p>
            </div>
            <div className="glass-panel p-6 border-l-4 border-l-blue-500">
              <p className="text-gray-400 text-sm uppercase tracking-wider">Detected Regions</p>
              <p className="text-4xl font-bold mt-2">{stats.region_count}</p>
            </div>
            <div className="glass-panel p-6 border-l-4 border-l-green-500">
              <p className="text-gray-400 text-sm uppercase tracking-wider">Status</p>
              <p className="text-xl font-bold mt-2 text-green-400">Analysis Complete</p>
            </div>
          </div>

          {/* View Toggles */}
          <div className="flex flex-wrap gap-3 mb-4">
            <button
              onClick={() => setActiveView("slider")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeView === "slider" ? "bg-blue-500/30 text-blue-300 ring-1 ring-blue-500/50" : "bg-white/5 text-gray-400 hover:bg-white/10"}`}
            >
              Comparison Slider
            </button>
            <button
              onClick={() => setActiveView("mask")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeView === "mask" ? "bg-blue-500/30 text-blue-300 ring-1 ring-blue-500/50" : "bg-white/5 text-gray-400 hover:bg-white/10"}`}
            >
              Binary Mask
            </button>
            <button
              onClick={() => setActiveView("heatmap")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeView === "heatmap" ? "bg-blue-500/30 text-blue-300 ring-1 ring-blue-500/50" : "bg-white/5 text-gray-400 hover:bg-white/10"}`}
            >
              <BarChart3 className="inline w-4 h-4 mr-1" /> Confidence Heatmap
            </button>

            {/* Overlay toggle */}
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className="ml-auto px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-400 hover:bg-white/10 flex items-center gap-2"
            >
              {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              Overlay
            </button>

            {/* Download */}
            <button
              onClick={downloadMask}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-400 hover:bg-white/10 flex items-center gap-2"
            >
              <Download className="w-4 h-4" /> Download Mask
            </button>
          </div>

          {/* Overlay Opacity Slider */}
          {activeView === "slider" && (
            <div className="flex items-center gap-3 mb-4 text-sm text-gray-400">
              <span>Opacity</span>
              <input
                type="range"
                min={0}
                max={100}
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                className="w-48 accent-blue-500"
              />
              <span>{overlayOpacity}%</span>
            </div>
          )}

          {/* Main Result View */}
          <div className="glass-panel p-2 rounded-2xl overflow-hidden shadow-[0_0_30px_rgba(255,0,0,0.15)]">
            {activeView === "slider" && overlayBase64 && previewA && previewB && (
              <ReactCompareSlider
                itemOne={<ReactCompareSliderImage src={previewA} alt="Before" style={{ objectFit: "contain", objectPosition: "center", background: "#000" }} />}
                itemTwo={
                  <div className="relative w-full h-full" style={{ background: "#000" }}>
                    <ReactCompareSliderImage src={previewB} alt="After" style={{ objectFit: "contain", objectPosition: "center" }} />
                    <img
                      src={`data:image/png;base64,${overlayBase64}`}
                      alt="Overlay"
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ opacity: showOverlay ? overlayOpacity / 100 : 0, objectFit: "contain", objectPosition: "center" }}
                    />
                  </div>
                }
                className="h-[380px] w-full rounded-xl"
              />
            )}
            {activeView === "mask" && maskBase64 && (
              <img
                src={`data:image/png;base64,${maskBase64}`}
                alt="Change Mask"
                className="w-full rounded-xl"
                style={{ maxHeight: "380px", objectFit: "contain" }}
              />
            )}
            {activeView === "heatmap" && heatmapBase64 && (
              <img
                src={`data:image/png;base64,${heatmapBase64}`}
                alt="Confidence Heatmap"
                className="w-full rounded-xl"
                style={{ maxHeight: "380px", objectFit: "contain" }}
              />
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!stats && !loading && !error && (
        <div className="mt-16 text-center text-gray-600">
          <p className="text-lg">Upload two satellite images above and press the button to start analysis.</p>
        </div>
      )}
    </div>
  );
}
