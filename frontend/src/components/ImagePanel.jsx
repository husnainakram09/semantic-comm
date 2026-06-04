export default function ImagePanel({ label, tag, tagColor, src, placeholder, loading }) {
  return (
    <div className="terminal-border bg-neutral-800/30 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-neutral-700/50 flex items-center justify-between">
        <span className="font-display text-sm font-semibold text-neutral-100">{label}</span>
        <span className={`label-tag border text-xs ${tagColor}`}>{tag}</span>
      </div>
      <div className="flex items-center justify-center min-h-[240px] p-4 relative bg-neutral-900/50">
        {loading && (
          <div className="absolute inset-0 bg-neutral-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-10 gap-3">
            <div className="flex gap-1.5">
              {[0, 1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="w-1.5 h-6 bg-accent-blue rounded-full animate-pulse"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
            <span className="font-mono text-xs text-accent-blue font-semibold">Reconstructing…</span>
          </div>
        )}
        {src ? (
          <img
            src={src}
            alt={label}
            className="max-h-56 w-full object-contain rounded"
          />
        ) : (
          <p className="font-mono text-xs text-neutral-600 text-center px-4">{placeholder}</p>
        )}
      </div>
    </div>
  )
}
