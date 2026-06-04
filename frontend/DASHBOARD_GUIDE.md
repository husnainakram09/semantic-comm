# Semantic Communication Demo Dashboard

A beautiful, production-grade React dashboard for demonstrating semantic image communication over noisy wireless channels.

## 🎨 Design & Aesthetics

### Color Palette
- **Deep Navy Background**: `#0a0f1e` (neutral-900)
- **Electric Blue Accents**: `#3b82f6` (accent-blue) - primary interaction colors
- **Subtle Cyan**: `#06b6d4` (accent-cyan) - secondary accents
- **Clean Whites**: Neutral-100 to Neutral-200 for text
- **Neutral Grays**: Gradient for secondary UI elements

### Typography
- **Headings**: Space Grotesk (Google Fonts) - modern, tech-forward aesthetic
- **Body**: Inter (Google Fonts) - clean, readable sans-serif
- **Monospace**: JetBrains Mono - code and numeric displays

### Visual Effects
- Scanline overlay (subtle)
- Animated gradient orbs on hero section
- Grid pattern background
- Smooth transitions and hover effects
- Glow effects on accent elements
- Pulse animations on loading states

## 📁 Project Structure

```
frontend/
├── src/
│   ├── App.jsx                    # Main app with routing & navbar
│   ├── index.css                  # Global styles with Google Fonts
│   ├── main.jsx                   # React entry point
│   ├── pages/
│   │   ├── Home.jsx               # Landing page with hero & features
│   │   └── Demo.jsx               # Interactive demo with 3-column layout
│   ├── components/
│   │   ├── ChannelConfig.jsx      # Channel type & SNR slider
│   │   ├── ImagePanel.jsx         # Image display with loading state
│   │   ├── MetricsDisplay.jsx     # Quality metrics cards with charts
│   │   ├── PipelineStatus.jsx     # Step-by-step pipeline progress
│   │   └── PipelineVisualizer.jsx # Visual flow of the pipeline
│   ├── services/
│   │   └── api.js                 # Axios API wrapper for Flask endpoints
│   └── hooks/
│       └── usePipeline.js         # Custom hook for pipeline state management
├── tailwind.config.js             # Tailwind configuration with custom colors
├── vite.config.js                 # Vite config with API proxy
├── postcss.config.js              # PostCSS/Tailwind config
├── package.json                   # Dependencies
└── index.html                     # HTML entry point
```

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm/yarn
- Python backend running on http://localhost:5000
- Flask API with endpoints: `/api/encode`, `/api/transmit`, `/api/decode`, `/api/metrics`

### Installation & Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server (runs on http://localhost:5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm preview
```

The Vite dev server automatically proxies `/api/*` requests to `http://localhost:5000`.

## 📋 Feature Breakdown

### HomePage (Landing Page)
- **Hero Section**: Eye-catching title with gradient text and animated background
- **Feature Cards**: Three main benefits of semantic communication
- **Pipeline Overview**: Four-step process cards with descriptions
- **Tech Stack**: Display of technologies used
- **CTA Section**: Call-to-action button to try the demo

### DemoPage (Interactive Dashboard)
Three-column layout for hands-on experimentation:

#### Left Column: Controls
- **Image Upload**: Drag-and-drop zone with preview
  - Supports PNG, JPG, WebP
  - Shows file size and name
- **Channel Configuration**:
  - Three channel types: AWGN, Rayleigh, Ideal
  - SNR slider: -5 to 30 dB
  - Real-time channel condition indicator (Poor/Moderate/Good)
- **Action Buttons**:
  - "Run Pipeline" button (disabled until image is uploaded)
  - "Reset" button (appears after processing)
- **Pipeline Status**:
  - Real-time progress indicators for each stage
  - Error display with human-readable messages

#### Center Column: Pipeline Visualizer
- **Visual Flow**: Three-step pipeline with animated connections
  - Encoding → Transmitting → Decoding
- **Interactive Indicators**:
  - Active step highlights with pulse animation
  - Completed steps show checkmarks
  - Color-coded by step (blue→orange→cyan)
- **Semantic Caption**: Display extracted text description from image
- **Channel Noise Display**: Visual SNR bar chart

#### Right Column: Results & Metrics
- **Image Comparison**: 
  - Original image (input)
  - Reconstructed image (output after noisy channel)
- **Quality Metrics** (animated cards):
  - PSNR (dB) - Peak Signal-to-Noise Ratio
  - SSIM - Structural Similarity Index
  - BLEU - Semantic Similarity Score
  - Color-coded indicators (green for good, orange for moderate)
  - Animated progress bars for each metric

## 🔌 API Integration

### API Service (`src/services/api.js`)

Provides clean wrapper functions around Flask endpoints:

```javascript
// Encode image to semantic features
await encodeImage(formData)

// Transmit through noisy channel
await transmitThroughChannel(latentCode, channelType, snrDb)

// Decode noisy latent back to image
await decodeImage(noisyLatent)

// Calculate quality metrics
await calculateMetrics(originalImage, reconstructedImage)

// Full pipeline in one call
await runFullPipeline(formData, channelType, snrDb)
```

### Expected Flask Endpoints

```
POST /api/encode
- Input: FormData with 'image' file
- Output: { latent_code: [...], caption: "..." }

POST /api/transmit
- Input: { latent_code: [...], channel_type: "...", snr_db: 10 }
- Output: { noisy_latent: [...] }

POST /api/decode
- Input: { noisy_latent: [...] }
- Output: { reconstructed_image: "data:image/..." }

POST /api/metrics
- Input: { original_image: "...", reconstructed_image: "..." }
- Output: { psnr: 25.5, ssim: 0.87, bleu: 0.75, channel_info: {...} }
```

## 🎣 Custom Hooks

### `usePipeline()` Hook

Manages complete pipeline state and orchestration:

```javascript
const {
  stage,              // Current pipeline stage
  results,            // Results from completed pipeline
  error,              // Error message if any
  caption,            // Extracted semantic caption
  progress,           // Progress percentage (0-100)
  isProcessing,       // Boolean: currently processing?
  runPipeline,        // Function to start pipeline
  reset,              // Function to reset state
} = usePipeline()

// Usage
await runPipeline(imageFile, snrDb, channelType)
```

## 🎬 Component Details

### ChannelConfig
- Buttons to select channel type (AWGN, Rayleigh, Ideal)
- Range slider for SNR adjustment
- Real-time channel status display
- Disabled state when pipeline is running

### ImagePanel
- Displays images with optional loading spinner
- Fallback placeholder text
- Supports dragging/dropping for upload
- Loading animation with pulsing bars

### PipelineStatus
- Shows current pipeline stage
- Checkmarks for completed steps
- Animated dots for active step
- Error display with warning icon

### PipelineVisualizer
- Circular step indicators
- Animated arrows between steps
- Caption display when extracted
- SNR visualization with color-coded bars
- Status footer message

### MetricsDisplay
- Three metric cards (PSNR, SSIM, BLEU)
- Color-coded quality badges
- Animated progress bars
- Channel information summary

## 🎨 Styling Details

### Tailwind Configuration
Extended with:
- Custom colors (neutral, accent-blue, accent-cyan, noise)
- Custom animations (wave, pulse-slow, scan)
- Custom components (.terminal-border, .btn-primary, .metric-card)
- Glass-morphism effects with backdrop-blur

### CSS Features
- Scanline overlay for retro feel
- Custom scrollbar styling
- Responsive grid layouts
- Smooth transitions on all interactive elements
- Hover states for all buttons

## 📱 Responsive Design

- **Desktop (lg)**: 3-column layout for full dashboard experience
- **Tablet (md)**: Adapts to 2 columns where needed
- **Mobile**: Single column, stacked layout
- All components scale gracefully

## ⚡ Performance Optimizations

- Lazy loading of images
- Memoized components where appropriate
- Efficient re-renders with proper key props
- Minimal bundle with tree-shaking
- Vite for fast dev/build

## 🔒 Production Checklist

- [ ] Verify all Flask endpoints are working
- [ ] Test with various image sizes (small, medium, large)
- [ ] Test on different browsers (Chrome, Firefox, Safari)
- [ ] Test responsive design on mobile
- [ ] Configure CORS if needed (frontend on different port)
- [ ] Update API base URL for production
- [ ] Add error logging/monitoring
- [ ] Test error scenarios (network failures, invalid images)
- [ ] Optimize images and assets
- [ ] Run `npm run build` for production

## 🚀 Deployment

```bash
# Build for production
npm run build

# Output goes to dist/ directory
# Deploy dist/ contents to web server

# Ensure API proxy is configured for your production backend
# Update vite.config.js proxy target if needed
```

## 🛠️ Development Notes

### Debugging
- Check browser console for client errors
- Check network tab to verify API calls
- Use React DevTools for component inspection
- Verify Flask backend is running on port 5000

### Common Issues
- **CORS errors**: Ensure Flask backend has CORS enabled
- **API proxy not working**: Check vite.config.js server.proxy settings
- **Images not loading**: Verify image paths and API responses
- **Styles not loading**: Clear browser cache, check Tailwind build

## 📚 Resources

- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [Vite Documentation](https://vitejs.dev)
- [Axios Documentation](https://axios-http.com)
- [Space Grotesk Font](https://fonts.google.com/specimen/Space+Grotesk)
- [Inter Font](https://fonts.google.com/specimen/Inter)

## 📄 License

This project is part of the Semantic Communication Demo research toolkit.
