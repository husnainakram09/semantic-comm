function MetricBar({ value, max, color }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="w-full bg-neutral-700/40 rounded-full h-2 mt-2 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

const METRIC_DEFS = [
  {
    key: 'psnr',
    label: 'PSNR',
    unit: 'dB',
    desc: 'Peak Signal-to-Noise Ratio',
    max: 50,
    color: 'bg-accent-blue',
    good: 30,
  },
  {
    key: 'ssim',
    label: 'SSIM',
    unit: '',
    desc: 'Structural Similarity Index',
    max: 1,
    color: 'bg-accent-cyan',
    good: 0.8,
    decimals: 4,
  },
  {
    key: 'bleu',
    label: 'BLEU',
    unit: '',
    desc: 'Semantic Similarity Score',
    max: 1,
    color: 'bg-purple-500',
    good: 0.7,
    decimals: 4,
  },
]

export default function MetricsDisplay({ metrics }) {
  return (
    <div className="terminal-border bg-neutral-800/30 rounded-xl p-6">
      <p className="font-mono text-xs text-accent-blue uppercase tracking-widest mb-6 font-semibold">
        ▸ Quality Metrics
      </p>
      <div className="grid grid-cols-3 gap-4">
        {METRIC_DEFS.map((def) => {
          const val = metrics[def.key]
          const formatted =
            val != null
              ? val.toFixed(def.decimals ?? 2)
              : '—'
          const isGood = val != null && val >= def.good

          return (
            <div key={def.key} className="metric-card bg-neutral-900/50">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="font-mono text-xs text-neutral-500 mb-1">{def.desc}</p>
                  <p className="font-display text-xl font-bold text-neutral-100">
                    {formatted}
                    {def.unit && (
                      <span className="text-xs font-mono text-neutral-600 ml-1">{def.unit}</span>
                    )}
                  </p>
                </div>
                <span
                  className={`label-tag border text-xs ${
                    isGood
                      ? 'text-green-400 bg-green-500/10 border-green-500/20'
                      : 'text-orange-400 bg-orange-500/10 border-orange-500/20'
                  }`}
                >
                  {def.label}
                </span>
              </div>
              <MetricBar value={val ?? 0} max={def.max} color={def.color} />
            </div>
          )
        })}
      </div>

      {metrics.channel_info && (
        <div className="mt-6 p-4 rounded-lg bg-neutral-900/50 border border-neutral-700">
          <p className="font-mono text-xs text-neutral-500">
            <span className="text-neutral-400">Channel:</span>{' '}
            {metrics.channel_info.type?.toUpperCase() ?? 'AWGN'}
            {metrics.channel_info.snr_db != null && (
              <> &nbsp;·&nbsp; <span className="text-accent-blue font-semibold">{metrics.channel_info.snr_db} dB SNR</span></>
            )}
          </p>
        </div>
      )}
    </div>
  )
}
