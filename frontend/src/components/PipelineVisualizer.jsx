const PIPELINE_FLOW = [
  {
    step: 'encoding',
    title: 'Semantic Encoding',
    description: 'Extract semantic meaning',
    icon: '⟨⟩',
    color: 'from-accent-blue to-blue-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-accent-blue/30',
  },
  {
    step: 'transmitting',
    title: 'Noisy Channel',
    description: 'Add channel noise',
    icon: '≋',
    color: 'from-orange-400 to-orange-500',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
  },
  {
    step: 'decoding',
    title: 'AI Reconstruction',
    description: 'Recover original image',
    icon: '◈',
    color: 'from-accent-cyan to-cyan-500',
    bgColor: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/30',
  },
]

function PipelineStep({ step, isActive, isComplete, flow }) {
  return (
    <div className="flex flex-col items-center relative z-10">
      {/* Circle */}
      <div
        className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold transition-all duration-300 border-2 ${
          isActive
            ? `border-${flow.color.split('-')[1]}-500 bg-gradient-to-br ${flow.color} text-neutral-900 shadow-lg scale-110 animate-pulse`
            : isComplete
            ? 'border-neutral-600 bg-neutral-700 text-neutral-300'
            : 'border-neutral-700 bg-neutral-800 text-neutral-600'
        }`}
      >
        {isComplete && !isActive ? '✓' : flow.icon}
      </div>

      {/* Title */}
      <h3
        className={`mt-3 font-display font-semibold text-sm text-center transition-colors ${
          isActive
            ? 'text-neutral-100'
            : isComplete
            ? 'text-neutral-500'
            : 'text-neutral-700'
        }`}
      >
        {flow.title}
      </h3>

      {/* Description */}
      <p
        className={`text-xs text-center mt-1 transition-colors ${
          isActive
            ? 'text-accent-blue'
            : isComplete
            ? 'text-neutral-600'
            : 'text-neutral-700'
        }`}
      >
        {flow.description}
      </p>
    </div>
  )
}

function Arrow({ isActive }) {
  return (
    <div className="flex-1 flex items-center justify-center px-2 relative">
      <div className={`h-1 w-full transition-all duration-300 ${
        isActive ? 'bg-accent-blue' : 'bg-neutral-700'
      }`} />
      <div className={`absolute right-0 w-0 h-0 border-l-4 border-l-transparent border-r-0 border-t-2 border-t-transparent border-b-2 border-b-transparent transition-all duration-300 ${
        isActive ? 'border-l-accent-blue' : 'border-l-neutral-700'
      }`} style={{ width: 0, height: 0, borderLeftWidth: '6px', borderTopWidth: '4px', borderBottomWidth: '4px' }} />
    </div>
  )
}

export default function PipelineVisualizer({ stage, caption, snrDb }) {
  const getStageIndex = (s) => {
    const stages = ['encoding', 'transmitting', 'decoding']
    return stages.indexOf(s)
  }

  const currentIdx = getStageIndex(stage)
  const isProcessing = ['encoding', 'transmitting', 'decoding'].includes(stage)

  return (
    <div className="terminal-border bg-neutral-800/30 rounded-xl p-6 h-full flex flex-col justify-between">
      <div>
        <p className="font-mono text-xs text-accent-blue uppercase tracking-widest mb-8 font-semibold">
          ▸ Pipeline Flow
        </p>

        {/* Main Pipeline Visualization */}
        <div className="space-y-8 mb-8">
          {/* Pipeline Steps */}
          <div className="flex items-center gap-2">
            {PIPELINE_FLOW.map((flow, idx) => (
              <div key={flow.step} className="flex-1 flex items-center gap-2">
                <PipelineStep
                  step={flow.step}
                  flow={flow}
                  isActive={stage === flow.step}
                  isComplete={idx < currentIdx || stage === 'done'}
                />
                {idx < PIPELINE_FLOW.length - 1 && (
                  <Arrow isActive={idx < currentIdx || (stage === 'done' && idx < PIPELINE_FLOW.length - 1)} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Semantic Caption Display */}
        {caption && (
          <div className={`terminal-border rounded-lg p-4 transition-all duration-300 ${
            stage === 'transmitting' || stage === 'decoding' || stage === 'done'
              ? 'bg-blue-500/10 border-accent-blue/30'
              : 'bg-neutral-900/50 border-neutral-700'
          }`}>
            <p className="font-mono text-xs text-neutral-500 mb-2 uppercase tracking-widest">
              Extracted Semantic Caption
            </p>
            <p className="font-body text-sm text-neutral-200 italic">
              "{caption}"
            </p>
          </div>
        )}

        {/* Channel Noise Indicator */}
        {snrDb !== undefined && (
          <div className="mt-6 p-4 rounded-lg bg-neutral-900/50 border border-neutral-700">
            <div className="flex items-center justify-between mb-2">
              <p className="font-mono text-xs text-neutral-500 uppercase tracking-widest">
                Channel SNR
              </p>
              <p className="font-mono text-sm font-semibold text-accent-blue">
                {snrDb} dB
              </p>
            </div>
            <div className="flex gap-1">
              {Array.from({ length: 10 }).map((_, i) => (
                <div
                  key={i}
                  className={`flex-1 h-1 rounded-full transition-all duration-300 ${
                    snrDb > 5 + i * 2.5
                      ? i < 3
                        ? 'bg-red-500'
                        : i < 6
                        ? 'bg-orange-500'
                        : 'bg-green-500'
                      : 'bg-neutral-700'
                  }`}
                />
              ))}
            </div>
            <p className="text-xs text-neutral-600 mt-2">
              {snrDb < 5 ? '🔴 Very Noisy' : snrDb < 15 ? '🟠 Moderate Noise' : '🟢 Clean Channel'}
            </p>
          </div>
        )}
      </div>

      {/* Status Footer */}
      <div className="mt-6 pt-4 border-t border-neutral-700">
        <p className="font-mono text-xs text-neutral-600 text-center">
          {!isProcessing ? (
            'Ready to process image'
          ) : stage === 'encoding' ? (
            'Extracting semantic features...'
          ) : stage === 'transmitting' ? (
            'Simulating noisy channel...'
          ) : stage === 'decoding' ? (
            'Reconstructing image...'
          ) : stage === 'done' ? (
            '✓ Complete'
          ) : (
            'Idle'
          )}
        </p>
      </div>
    </div>
  )
}
