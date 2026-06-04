const STEPS = [
  { id: 'encoding',    label: 'Encoding',    icon: '⟨⟩' },
  { id: 'transmitting', label: 'Transmitting', icon: '≋' },
  { id: 'decoding',   label: 'Decoding',    icon: '◈' },
]

export default function PipelineStatus({ stage, error }) {
  if (stage === 'idle') return null

  return (
    <div className="terminal-border bg-neutral-800/30 rounded-xl p-5">
      <p className="font-mono text-xs text-accent-blue uppercase tracking-widest mb-4 font-semibold">
        ▸ Status
      </p>
      <div className="space-y-2">
        {STEPS.map((step) => {
          const order = ['encoding', 'transmitting', 'decoding']
          const stepIdx = order.indexOf(step.id)
          const stageIdx = order.indexOf(stage)
          const done = stageIdx > stepIdx || stage === 'done'
          const active = stage === step.id

          return (
            <div
              key={step.id}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-300 ${
                active  ? 'bg-blue-500/10 border border-accent-blue/30' :
                done    ? 'bg-neutral-700/20' :
                          'opacity-40'
              }`}
            >
              <span className={`font-mono text-base ${
                active ? 'text-accent-blue animate-pulse' : 
                done ? 'text-neutral-500' : 
                'text-neutral-700'
              }`}>
                {done && !active ? '✓' : step.icon}
              </span>
              <span className={`font-mono text-sm ${
                active ? 'text-accent-blue font-semibold' : 
                done ? 'text-neutral-500' : 
                'text-neutral-700'
              }`}>
                {step.label}
              </span>
              {active && (
                <div className="ml-auto flex gap-1">
                  {[0,1,2].map(i => (
                    <div key={i} className="w-1 h-1 rounded-full bg-accent-blue animate-bounce"
                      style={{ animationDelay: `${i*0.1}s` }} />
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {stage === 'done' && (
        <p className="font-mono text-xs text-accent-cyan mt-4 text-center font-semibold">
          ✓ Pipeline complete
        </p>
      )}

      {stage === 'error' && error && (
        <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <p className="font-mono text-xs text-red-400">⚠ {error}</p>
        </div>
      )}
    </div>
  )
}
