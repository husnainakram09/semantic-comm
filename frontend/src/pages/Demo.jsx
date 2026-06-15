import { useState, useRef, useCallback } from "react";
import ChannelConfig from "../components/ChannelConfig";
import ImagePanel from "../components/ImagePanel";
import MetricsDisplay from "../components/MetricsDisplay";
import PipelineStatus from "../components/PipelineStatus";
import PipelineVisualizer from "../components/PipelineVisualizer";
import api from "../services/api";

const INITIAL_CHANNEL = { type: "awgn", snr: 10 };

export default function Demo() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [channel, setChannel] = useState(INITIAL_CHANNEL);
  const [stage, setStage] = useState("idle");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [caption, setCaption] = useState(null);
  const fileRef = useRef();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setResults(null);
    setError(null);
    setStage("idle");
    setCaption(null);
    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    setResults(null);
    setError(null);
    setStage("idle");
    setCaption(null);
    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target.result);
    reader.readAsDataURL(file);
  }, []);

  const runPipeline = async () => {
    if (!imageFile) return;
    setError(null);
    setResults(null);
    setCaption(null);

    try {
      // Step 1: Encode
      setStage("encoding");
      const formData = new FormData();
      formData.append("image", imageFile);
      const encodeRes = await api.post("/encode", formData);
      const { embedding, caption: encodedCaption } = encodeRes.data;
      setCaption(encodedCaption);

      // Step 2: Transmit
      setStage("transmitting");
      const transmitRes = await api.post("/transmit", {
        embedding,
        channel_type: channel.type,
        snr_db: channel.snr,
      });
      const { noisy_embedding, snr_db } = transmitRes.data;

      // Step 3: Decode
      setStage("decoding");
      const decodeRes = await api.post("/decode", {
        noisy_embedding,
        snr_db,
        original_caption: encodedCaption,
      });
      const reconstructed_image = decodeRes.data.reconstructed_image_b64;

      // Step 4: Metrics
      const metricsData = new FormData();
      metricsData.append("original_image", imageFile);
      metricsData.append("reconstructed_image_b64", reconstructed_image);
      metricsData.append("original_caption", encodedCaption);
      metricsData.append("recovered_caption", decodeRes.data.recovered_caption);
      metricsData.append(
        "compression_ratio",
        transmitRes.data.bandwidth_savings?.compression_ratio || 1
      );
      const metricsRes = await api.post("/metrics", metricsData);

      setResults({
        reconstructed: reconstructed_image,
        metrics: metricsRes.data,
      });
      setStage("done");
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || err.message || "Pipeline failed");
      setStage("error");
    }
  };

  const reset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    setStage("idle");
    setCaption(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const isRunning = ["encoding", "transmitting", "decoding"].includes(stage);

  return (
    <div className="min-h-screen bg-neutral-900 pb-20">
      {/* Header */}
      <div className="max-w-7xl mx-auto px-6 pt-12 pb-8 border-b border-neutral-800">
        <h1 className="font-display text-4xl font-bold text-neutral-100 mb-2">
          Interactive Demo
        </h1>
        <p className="text-neutral-500 font-body">
          Upload an image, configure the channel noise level, and watch the full
          semantic communication pipeline in action.
        </p>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* LEFT COLUMN: Upload & Config */}
          <div className="lg:col-span-1 space-y-4">
            {/* Input Image Upload */}
            <div className="terminal-border bg-neutral-800/30 rounded-xl p-6">
              <p className="font-mono text-xs text-accent-blue uppercase tracking-widest mb-4 font-semibold">
                ▸ Input Image
              </p>
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileRef.current?.click()}
                className={`relative cursor-pointer rounded-lg border-2 border-dashed transition-all duration-200 flex flex-col items-center justify-center min-h-[180px] p-4 ${
                  imagePreview
                    ? "border-accent-blue/50 bg-blue-500/5"
                    : "border-neutral-700 hover:border-accent-blue/50 hover:bg-blue-500/5"
                }`}
              >
                {imagePreview ? (
                  <img
                    src={imagePreview}
                    alt="preview"
                    className="max-h-44 rounded object-contain"
                  />
                ) : (
                  <>
                    <span className="text-4xl mb-2">⬆️</span>
                    <p className="text-neutral-400 font-mono text-xs text-center font-semibold">
                      Drop image here
                      <br />
                      or click to browse
                    </p>
                    <p className="text-neutral-600 font-mono text-xs mt-2">
                      PNG, JPG, or WebP
                    </p>
                  </>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>
              {imageFile && (
                <p className="font-mono text-xs text-neutral-500 mt-3 truncate">
                  📁 {imageFile.name} ({(imageFile.size / 1024).toFixed(1)} KB)
                </p>
              )}
            </div>

            {/* Channel Configuration */}
            <ChannelConfig
              value={channel}
              onChange={setChannel}
              disabled={isRunning}
            />

            {/* Action Buttons */}
            <div className="space-y-2">
              <button
                onClick={runPipeline}
                disabled={!imageFile || isRunning}
                className="btn-primary w-full justify-center font-semibold"
              >
                {isRunning ? (
                  <>
                    <span className="inline-block animate-spin mr-2">⚡</span>
                    Processing...
                  </>
                ) : (
                  <>▶ Run Pipeline</>
                )}
              </button>
              {(stage === "done" || stage === "error") && (
                <button onClick={reset} className="btn-ghost w-full">
                  ↻ Reset
                </button>
              )}
            </div>

            {/* Pipeline Status */}
            <PipelineStatus stage={stage} error={error} />
          </div>

          {/* CENTER COLUMN: Pipeline Visualizer */}
          <div className="lg:col-span-1">
            <PipelineVisualizer
              stage={stage}
              caption={caption}
              snrDb={channel.snr}
            />
          </div>

          {/* RIGHT COLUMN: Results & Metrics */}
          <div className="lg:col-span-1 space-y-4">
            <div className="space-y-4">
              <ImagePanel
                label="Original Image"
                tag="INPUT"
                tagColor="text-accent-blue bg-blue-500/10 border-accent-blue/30"
                src={imagePreview}
                placeholder="Upload an image to begin"
              />
              <ImagePanel
                label="Reconstructed Image"
                tag="OUTPUT"
                tagColor="text-accent-cyan bg-cyan-500/10 border-cyan-500/30"
                src={results?.reconstructed}
                placeholder={
                  isRunning ? "Reconstructing…" : "Results will appear here"
                }
                loading={stage === "decoding"}
              />
            </div>

            {results?.metrics && <MetricsDisplay metrics={results.metrics} />}
          </div>
        </div>
      </div>
    </div>
  );
}
