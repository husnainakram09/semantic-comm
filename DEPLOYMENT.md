# Deployment Guide: Semantic Communication Demo

## Overview

This document covers deploying the Semantic Communication Demo to:
1. **Hugging Face Spaces** (Backend - CPU-only free tier)
2. **Vercel** (Frontend - already configured)
3. **Local Development** (both frontend and backend)

## Backend Deployment

### Hugging Face Spaces (Recommended for Production)

**Prerequisites:**
- Hugging Face account with Spaces access
- Backend code committed to a Git repository

**Key Requirements:**
- ✅ **Port**: 7860 (HF Spaces standard)
- ✅ **Host**: 0.0.0.0 (listen on all interfaces)
- ✅ **Memory**: CPU-optimized model loading
- ✅ **Cold start**: Models pre-loaded at app startup
- ✅ **Cache**: Uses HF_HOME for persistent model caching

**Setup Steps:**

1. Create a new Space on huggingface.co:
   - Name: `semantic-comm-demo`
   - License: OpenRAIL-M (recommended for models)
   - Space SDK: Docker (or Python if preferred)

2. Push this repository to your Space (HF will auto-deploy):
   ```bash
   git push https://huggingface.co/spaces/YOUR-USERNAME/semantic-comm-demo main
   ```

3. **Environment Variables** (set in Space settings):
   ```
   DEVICE=cpu
   FLASK_ENV=production
   PORT=7860
   HOST=0.0.0.0
   HF_HOME=/tmp/hf_cache
   TOKENIZERS_PARALLELISM=false
   ```

4. **Docker Setup** (if using Docker SDK):
   Create a `Dockerfile` in the repository root:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY backend/requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY backend/ .
   ENV DEVICE=cpu
   ENV FLASK_ENV=production
   CMD ["python", "app.py"]
   ```

**Important Notes:**
- First startup may take 2-3 minutes (model downloads)
- Subsequent restarts use cached models (~10 seconds)
- Stable Diffusion (~4GB) uses CPU memory efficiently via:
  - `enable_attention_slicing()` - reduces memory
  - `enable_sequential_cpu_offload()` - trades speed for memory
- BLIP and SentenceTransformer (~400MB) auto-cached

---

### Local Development

**Backend Setup:**

1. Install Python 3.11+ and create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```bash
   # Port 7860 (default, matching HF Spaces)
   python app.py

   # OR port 5000 (for local development with old config)
   PORT=5000 python app.py
   ```

4. Test the server:
   ```bash
   curl http://localhost:7860/health
   # Should return: {"status": "ok"}
   ```

**Frontend Setup:**

1. Install Node.js 18+ dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the dev server:
   ```bash
   # Uses port 7860 for backend (default)
   npm run dev

   # OR use port 5000 for backend
   VITE_API_PORT=5000 npm run dev
   ```

3. Open http://localhost:5173 in your browser

---

## API Endpoints

All endpoints are relative to the backend server (e.g., http://localhost:7860):

### POST /encode
Upload an image → get semantic caption + embedding
```bash
curl -X POST http://localhost:7860/encode \
  -F "image=@image.jpg"
```
Returns:
```json
{
  "caption": "a cat sitting on a windowsill",
  "embedding": [0.123, -0.456, ...],
  "token_count": 12
}
```

### POST /transmit
Simulate noisy channel on embedding
```bash
curl -X POST http://localhost:7860/transmit \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.123, -0.456, ...],
    "snr_db": 10.0,
    "image_width": 512,
    "image_height": 512
  }'
```

### POST /decode
Reconstruct image from noisy embedding
```bash
curl -X POST http://localhost:7860/decode \
  -H "Content-Type: application/json" \
  -d '{
    "noisy_embedding": [0.123, -0.456, ...],
    "original_caption": "a cat sitting on a windowsill",
    "snr_db": 10.0,
    "seed": 42
  }'
```

### POST /metrics
Compute quality metrics
```bash
curl -X POST http://localhost:7860/metrics \
  -F "original_image=@original.jpg" \
  -F "reconstructed_image_b64=data:image/png;base64,..." \
  -F "original_caption=..." \
  -F "recovered_caption=..." \
  -F "compression_ratio=512"
```

### GET /health
Health check (used by HF Spaces)
```bash
curl http://localhost:7860/health
```
Returns: `{"status": "ok"}`

---

## Performance Characteristics

### Cold Start (First Request)
- **Time**: 2-3 minutes (one-time)
- **What happens**: Models download from HuggingFace Hub → initialize → cached
- **Models loaded**:
  - BLIP image captioning (Salesforce/blip-image-captioning-base) - ~350MB
  - SentenceTransformer (all-MiniLM-L6-v2) - ~70MB
  - Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5) - ~4GB

### Warm Requests (After Cold Start)
- **/encode**: 1-2 seconds (BLIP caption generation)
- **/transmit**: <10ms (pure NumPy AWGN)
- **/decode**: 15-30 seconds (SD text-to-image generation on CPU)
- **/metrics**: <100ms (PSNR/SSIM/BLEU computation)

### Memory Usage
- **Peak on CPU**: ~8-10 GB RAM (Stable Diffusion + inference)
  - Mitigated by `enable_sequential_cpu_offload()`
- **Model cache**: ~4.4GB on disk (HF_HOME)
- **HF Spaces free tier**: Limited to ~16GB disk + memory

---

## Troubleshooting

### "Connection refused" on /api endpoint
**Problem**: Frontend can't reach backend
**Solution**:
- Check backend is running: `curl http://localhost:7860/health`
- For local dev: `VITE_API_PORT=7860 npm run dev`
- For HF Spaces: frontend should call `https://your-space-url/api/...`

### "CUDA out of memory" or similar GPU errors
**Problem**: DEVICE=cuda used but GPU unavailable
**Solution**:
- Ensure `DEVICE=cpu` environment variable is set
- Backend defaults to CPU if not set

### First request hangs or times out
**Problem**: Cold start taking too long
**Solution**:
- This is expected on first startup (2-3 min)
- Check `app.initialize_models()` logs
- Subsequent requests will be much faster

### "Model download interrupted"
**Problem**: Network error during model download
**Solution**:
- Ensure HF_HOME directory is writable
- Clear cache: `rm -rf /tmp/hf_cache`
- Try again - HuggingFace Hub will resume download

---

## Frontend Deployment to Vercel

1. Push frontend code to GitHub
2. Connect repo to Vercel
3. Set environment variables (if needed):
   ```
   VITE_API_URL=https://your-hf-space-url/api
   ```
4. Deploy with `vercel deploy`

The frontend will call the backend at the configured API URL.

---

## Code Changes from Original

### app.py
- ✅ Added `initialize_models()` to pre-load models at startup
- ✅ Changed default port from 5000 → 7860
- ✅ Added environment variable support (HF_HOME, TOKENIZERS_PARALLELISM)
- ✅ Removed debug print statements
- ✅ Better error handling and logging

### encoder.py
- ✅ Added `initialize_encoder_models()` for startup loading
- ✅ Optimized NLTK and HuggingFace caching

### decoder.py
- ✅ Added `enable_attention_slicing()` for both CUDA and CPU
- ✅ Added `enable_sequential_cpu_offload()` for CPU memory efficiency
- ✅ Added `initialize_decoder_models()` for startup loading
- ✅ Better error handling and logging

### requirements.txt
- ✅ Pinned package versions for stability
- ✅ Added accelerate and safetensors for model optimization

### vite.config.js
- ✅ Updated proxy to use configurable API port
- ✅ Default to port 7860 (HF Spaces standard)

---

## Monitoring & Logs

### HuggingFace Spaces Logs
Check real-time logs in your Space dashboard or via CLI:
```bash
huggingface-cli repo logs spaces/YOUR-USERNAME/semantic-comm-demo
```

### Local Logging
Backend runs with INFO level logging showing:
- Model initialization progress
- Request routing and timing
- Error details with stack traces

Increase verbosity:
```bash
FLASK_ENV=development python app.py
```

---

## Next Steps

1. **Test locally first**: Run both frontend and backend locally
2. **Deploy backend to HF Spaces**: Use Space creation UI
3. **Update frontend API URL**: Set to your HF Space URL
4. **Deploy frontend to Vercel**: Connect GitHub repo
5. **Monitor performance**: Check logs and response times

---

For questions or issues, refer to:
- [HuggingFace Spaces Docs](https://huggingface.co/spaces)
- [Vercel Documentation](https://vercel.com/docs)
- Backend logs for detailed error messages
