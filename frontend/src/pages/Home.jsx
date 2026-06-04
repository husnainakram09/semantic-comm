import { Link } from "react-router-dom";

const PIPELINE_STEPS = [
  {
    id: "01",
    label: "Semantic Encoding",
    color: "text-accent-blue",
    bg: "bg-blue-500/10 border-accent-blue/30",
    icon: "⟨⟩",
    desc: "Extract semantic features from the input image using AI. Compress meaning, not pixels.",
  },
  {
    id: "02",
    label: "Noisy Channel",
    color: "text-orange-400",
    bg: "bg-orange-500/10 border-orange-500/30",
    icon: "≋",
    desc: "Simulate wireless transmission with configurable SNR. Test robustness to channel noise.",
  },
  {
    id: "03",
    label: "AI Reconstruction",
    color: "text-accent-cyan",
    bg: "bg-cyan-500/10 border-cyan-500/30",
    icon: "◈",
    desc: "Recover the original image from noisy latent codes. See how AI preserves semantic content.",
  },
  {
    id: "04",
    label: "Quality Metrics",
    color: "text-purple-400",
    bg: "bg-purple-500/10 border-purple-500/30",
    icon: "∿",
    desc: "Measure PSNR, SSIM, and BLEU. Quantify how well meaning survived the noisy channel.",
  },
];

const FEATURE_CARDS = [
  {
    title: "Semantic Encoding",
    description:
      "Compress images by encoding only their semantic meaning, not raw pixels. Uses state-of-the-art vision transformers.",
    icon: "🧠",
  },
  {
    title: "Noisy Channel Simulation",
    description:
      "Add configurable noise to test robustness. Simulate real wireless conditions with SNR control.",
    icon: "📡",
  },
  {
    title: "AI Reconstruction",
    description:
      "Diffusion models rebuild images from noisy semantic codes. Watch AI magic recover lost details.",
    icon: "✨",
  },
];

function AnimatedWaveBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Animated gradient orb */}
      <div className="absolute top-20 right-1/4 w-96 h-96 rounded-full bg-blue-500/10 blur-3xl animate-pulse-slow" />
      <div
        className="absolute bottom-32 left-1/3 w-80 h-80 rounded-full bg-cyan-500/10 blur-3xl animate-pulse-slow"
        style={{ animationDelay: "1s" }}
      />

      {/* Grid pattern */}
      <svg className="absolute inset-0 w-full h-full opacity-20">
        <defs>
          <pattern
            id="grid"
            width="50"
            height="50"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 50 0 L 0 0 0 50"
              fill="none"
              stroke="rgba(59, 130, 246, 0.1)"
              strokeWidth="0.5"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative max-w-7xl mx-auto px-6 pt-24 pb-32 text-center overflow-hidden">
        <AnimatedWaveBackground />
        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 label-tag bg-blue-500/10 text-accent-blue border border-accent-blue/30 mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-blue animate-pulse-slow inline-block" />
            Semantic Communication Research Demo
          </div>

          <h1 className="font-display text-6xl md:text-8xl font-bold tracking-tight text-neutral-100 leading-[1.05] mb-6">
            Transmit{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-blue via-blue-400 to-accent-cyan">
              meaning,
            </span>
            <br />
            not just bits.
          </h1>

          <p className="text-neutral-400 font-body text-lg md:text-xl max-w-3xl mx-auto leading-relaxed mb-12">
            A hands-on demonstration of semantic image communication — encode
            semantic features, simulate a noisy wireless channel, reconstruct
            with AI, and measure perceptual quality in real time. See how AI
            transmits meaning even when data is corrupted.
          </p>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link to="/demo" className="btn-primary group">
              <span className="inline-flex items-center gap-2">
                Launch Interactive Demo
                <svg
                  className="w-4 h-4 group-hover:translate-x-1 transition-transform"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7l5 5m0 0l-5 5m5-5H6"
                  />
                </svg>
              </span>
            </Link>
            <a
              href="https://arxiv.org/abs/2212.00500"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-ghost"
            >
              Read the Research Paper
            </a>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-6 py-20 border-t border-neutral-800">
        <div className="text-center mb-16">
          <p className="font-mono text-accent-blue text-sm tracking-widest uppercase mb-4">
            How It Works
          </p>
          <h2 className="font-display text-4xl font-bold text-neutral-100 mb-4">
            Three-Stage Pipeline
          </h2>
          <p className="text-neutral-500 max-w-2xl mx-auto">
            Watch as your image travels through an AI-powered semantic
            communication system
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {FEATURE_CARDS.map((card, idx) => (
            <div
              key={idx}
              className="group terminal-border bg-neutral-800/30 hover:bg-neutral-800/50 hover:border-accent-blue/40 p-8 rounded-xl transition-all duration-300"
            >
              <div className="text-4xl mb-4">{card.icon}</div>
              <h3 className="font-display text-lg font-semibold text-neutral-100 mb-3 group-hover:text-accent-blue transition-colors">
                {card.title}
              </h3>
              <p className="text-neutral-400 text-sm leading-relaxed">
                {card.description}
              </p>
            </div>
          ))}
        </div>

        {/* Pipeline Steps */}
        <div className="mt-20">
          <p className="font-mono text-neutral-500 text-sm tracking-widest uppercase mb-8">
            Detailed Pipeline
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {PIPELINE_STEPS.map((step, idx) => (
              <div
                key={step.id}
                className={`terminal-border ${step.bg} p-6 rounded-xl hover:scale-105 transition-transform duration-200`}
              >
                <div className="flex items-start justify-between mb-4">
                  <span className={`text-3xl ${step.color}`}>{step.icon}</span>
                  <span className="font-mono text-xs text-neutral-600">
                    {step.id}
                  </span>
                </div>
                <h3
                  className={`font-display font-semibold text-base mb-2 ${step.color}`}
                >
                  {step.label}
                </h3>
                <p className="text-neutral-400 text-sm leading-relaxed">
                  {step.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="max-w-7xl mx-auto px-6 py-20 border-t border-neutral-800">
        <p className="font-mono text-accent-blue text-xs tracking-widest uppercase mb-8">
          Technology Stack
        </p>
        <div className="flex flex-wrap gap-3">
          {[
            "React 18 + Vite",
            "Tailwind CSS",
            "Flask Backend",
            "PyTorch",
            "HuggingFace BLIP",
            "Stable Diffusion",
            "scikit-image",
            "NLTK",
          ].map((tech) => (
            <span
              key={tech}
              className="font-mono text-xs px-3 py-2 rounded-lg bg-neutral-800 text-neutral-300 border border-neutral-700 hover:border-accent-blue/50 transition-colors"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative max-w-7xl mx-auto px-6 py-20">
        <div className="terminal-border bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-accent-blue/30 p-12 rounded-xl text-center">
          <h2 className="font-display text-3xl font-bold text-neutral-100 mb-4">
            Ready to see AI handle noisy communication?
          </h2>
          <p className="text-neutral-400 mb-8 max-w-2xl mx-auto">
            Upload an image, configure the channel noise, and watch as semantic
            encoding preserves your image through a noisy wireless channel.
          </p>
          <Link to="/demo" className="btn-primary inline-block">
            Try the Interactive Demo Now →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <section className="border-t border-neutral-800 mt-20 py-12">
        <div className="max-w-7xl mx-auto px-6 text-center text-neutral-600 text-sm">
          <p>Semantic Communication Demo • Research & Development Tool</p>
        </div>
      </section>
    </div>
  );
}
