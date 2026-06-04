const CHANNEL_TYPES = [
  { id: 'awgn',    label: 'AWGN',    desc: 'Additive White Gaussian Noise' },
  { id: 'rayleigh', label: 'Rayleigh', desc: 'Rayleigh fading channel' },
  { id: 'none',   label: 'Ideal',   desc: 'No noise (baseline)' },
]

function getChannelStatus(snr) {
  if (snr < 5) return { label: 'Poor', color: 'text-red-400' }
  if (snr < 15) return { label: 'Moderate', color: 'text-orange-400' }
  return { label: 'Good', color: 'text-green-400' }
}

export default function ChannelConfig({ value, onChange, disabled }) {
  const channelStatus = getChannelStatus(value.snr)

  return (
    <div className="terminal-border bg-neutral-800/30 rounded-xl p-6">
      <p className="font-mono text-xs text-accent-blue uppercase tracking-widest mb-4 font-semibold">
        ▸ Channel Configuration
      </p>

      <div className="space-y-2 mb-6">
        {CHANNEL_TYPES.map((ch) => (
          <button
            key={ch.id}
            onClick={() => onChange({ ...value, type: ch.id })}
            disabled={disabled}
            className={`w-full text-left px-3 py-2.5 rounded-lg border transition-all duration-150 ${  
              value.type === ch.id
                ? 'border-accent-blue/60 bg-blue-500/10 text-neutral-100'
                : 'border-neutral-700 bg-neutral-900/40 text-neutral-400 hover:border-neutral-600'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span className="font-mono text-sm font-semibold block">{ch.label}</span>
            <span className="font-body text-xs text-neutral-500">{ch.desc}</span>
          </button>
        ))}
      </div>

      {value.type !== 'none' && (
        <div>
          <div className="flex justify-between items-center mb-3">
            <label className="font-mono text-xs text-neutral-400 font-semibold">Signal-to-Noise Ratio</label>
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm text-accent-blue font-bold">
                {value.snr} dB
              </span>
              <span className={`label-tag text-xs ${channelStatus.color}`}>
                {channelStatus.label}
              </span>
            </div>
          </div>
          <input
            type="range"
            min={-5}
            max={30}
            step={1}
            value={value.snr}
            onChange={(e) => onChange({ ...value, snr: Number(e.target.value) })}
            disabled={disabled}
            className="w-full accent-accent-blue cursor-pointer"
          />
          <div className="flex justify-between font-mono text-xs text-neutral-600 mt-2">
            <span>−5 dB (very noisy)</span>
            <span>30 dB (very clean)</span>
          </div>
        </div>
      )}
    </div>
  )
}
