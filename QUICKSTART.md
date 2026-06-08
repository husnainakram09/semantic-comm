# Quick Start Guide

Get the Semantic Communication Demo running locally in 5 minutes.

## Prerequisites

- Python 3.11+ (download from [python.org](https://www.python.org/downloads/))
- Node.js 18+ (download from [nodejs.org](https://nodejs.org/))
- Git (download from [git-scm.com](https://git-scm.com/))

## Option A: Run Everything Locally (Recommended for Development)

### Step 1: Start the Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
# Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```

You should see output like:
```
2024-01-15 10:30:45,123 [INFO] app – Device: cpu
2024-01-15 10:30:46,456 [INFO] app – Starting Flask app on 0.0.0.0:7860
2024-01-15 10:30:47,789 [INFO] app – Running on http://0.0.0.0:7860
```

⏳ **First startup takes 2-3 minutes** as models download. You'll see:
- "Pre-loading BLIP and SentenceTransformer models…"
- "Loading Stable Diffusion pipeline…"
- "✓ All models initialized successfully"

✅ Once you see "Running on", the backend is ready!

### Step 2: Start the Frontend (New Terminal)

```bash
# Navigate to frontend directory (from repo root)
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  press h + enter to show help
```

### Step 3: Open in Browser

Open **http://localhost:5173** in your browser. You should see the Semantic Communication Demo home page!

---

## Option B: Using Different Backend Port (If Port 7860 is in Use)

```bash
# Terminal 1: Start backend on port 5000
cd backend
PORT=5000 python app.py

# Terminal 2: Start frontend with custom API port
cd frontend
VITE_API_PORT=5000 npm run dev
```

---

## Testing the App

### 1. Test Health Check
```bash
curl http://localhost:7860/health
# Should return: {"status": "ok"}
```

### 2. Use the Demo Page

1. Click **"Start Demo"** on the home page
2. Click **"Upload Image"** and select an image
3. Adjust the **SNR (Signal-to-Noise Ratio)** slider
4. Click **"Run Pipeline"**
5. Watch the progress bars and see results!

### Expected Output
- **Original Image**: Your uploaded image
- **Recovered Caption**: Text description extracted by BLIP
- **Reconstructed Image**: Image generated from the noisy caption via Stable Diffusion
- **Metrics**: PSNR, SSIM, BLEU scores showing quality

---

## Troubleshooting

### "Connection refused" Error
```
Error: Failed to connect to backend
```
**Solution**: Make sure backend is running with `python app.py` in the backend directory.

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Use a different port:
```bash
PORT=8000 python app.py  # Backend on 8000
VITE_API_PORT=8000 npm run dev  # Frontend pointing to 8000
```

### "Model download interrupted"
```
ConnectionError: Unable to connect to HuggingFace Hub
```
**Solution**: 
- Check your internet connection
- Wait 30 seconds and restart the backend
- Models will resume downloading from where they left off

### Slow Image Generation
```
SD inference takes 30+ seconds
```
**This is normal** for CPU-only systems. Stable Diffusion inference is slow without a GPU. Expected times:
- CPU: 15-30 seconds per image
- GPU (CUDA): 2-5 seconds per image

---

## Next Steps

### Learn About Semantic Communication
See [README.md](README.md) for the project overview and technical details.

### Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying to:
- Hugging Face Spaces (backend)
- Vercel (frontend)
- Docker containers

### Code Changes
See [BACKEND_OPTIMIZATION.md](BACKEND_OPTIMIZATION.md) for details on what was optimized.

---

## Development Tips

### Modify Frontend Code
```bash
cd frontend
npm run dev
# Changes auto-reload in browser
```

### Modify Backend Code
```bash
cd backend
# Stop the server (Ctrl+C)
python app.py
# Changes require restart
```

### View Backend Logs
Backend logs are printed to terminal with timestamps and log levels:
```
2024-01-15 10:35:22,123 [INFO] app – Processing /encode request
2024-01-15 10:35:23,456 [WARNING] encoder – Caption is very short
2024-01-15 10:35:45,789 [ERROR] decoder – SD inference failed
```

### Reset Everything
```bash
# Clear model cache
rm -rf /tmp/hf_cache

# Delete frontend node_modules
cd frontend && rm -rf node_modules && npm install

# Delete backend venv
cd backend && rm -rf venv && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Restart both
```

---

## Performance Tips

### Speed Up First Startup
First startup downloads 4.4GB of models. To speed up testing:
1. Run once to download models (2-3 minutes)
2. Subsequently they're cached
3. Restart is now fast (~10 seconds)

### Local GPU Acceleration
If you have an NVIDIA GPU:
```bash
# In backend directory
DEVICE=cuda python app.py
# Image generation now takes 2-5 seconds instead of 15-30
```

### Disable Hot Reload (for stability)
```bash
# Frontend
npm run build  # Build production version
npx serve -l 5173 dist

# Reduces CPU usage if app is running in background
```

---

## File Structure Reference

```
semantic-comm-demo/
├── backend/                     # Flask API server
│   ├── app.py                  # Main Flask app
│   ├── encoder.py              # BLIP image captioning
│   ├── decoder.py              # Stable Diffusion image generation
│   ├── channel.py              # AWGN channel simulation
│   ├── metrics.py              # Quality metrics (PSNR, SSIM, BLEU)
│   └── requirements.txt         # Python dependencies
│
├── frontend/                    # React + Vite
│   ├── src/
│   │   ├── pages/              # Home and Demo pages
│   │   ├── components/         # UI components
│   │   ├── services/           # API client
│   │   ├── hooks/              # Custom React hooks
│   │   └── index.css           # Global styles
│   ├── vite.config.js          # Vite configuration
│   └── package.json            # Node dependencies
│
├── README.md                    # Project overview
├── DEPLOYMENT.md               # Deployment guide
└── BACKEND_OPTIMIZATION.md     # Technical optimization details
```

---

## Getting Help

1. **Check logs**: Terminal output often has helpful error messages
2. **Read error message carefully**: Usually indicates the problem
3. **Try the troubleshooting section above**
4. **Check DEPLOYMENT.md**: Common issues documented there
5. **Check BACKEND_OPTIMIZATION.md**: Technical details on implementation

---

## Common Questions

**Q: Why is generation so slow?**
A: Stable Diffusion inference on CPU takes 15-30 seconds. Use `DEVICE=cuda` if you have a GPU.

**Q: Can I use this without internet?**
A: Yes, after first startup. Models are cached locally in `/tmp/hf_cache`.

**Q: What's the minimum RAM needed?**
A: ~8GB for CPU inference. 4GB+ for development without running inference.

**Q: Can I use an older Python version?**
A: Python 3.10+ should work, but 3.11+ is recommended and tested.

**Q: How do I monitor the backend?**
A: Check the terminal where you ran `python app.py`. All requests and errors are logged there.

---

Enjoy exploring semantic communication! 🚀
