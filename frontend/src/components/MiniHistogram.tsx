interface MiniHistogramProps {
  counts: number[]
  binEdges: number[]
  title: string
}

// Plain-SVG histogram — no charting library dependency for a simple bar chart.
export function MiniHistogram({ counts, binEdges, title }: MiniHistogramProps) {
  const width = 420
  const height = 220
  const padding = 30
  const max = Math.max(...counts, 1)
  const barWidth = (width - padding * 2) / counts.length

  return (
    <div className="card" style={{ padding: '1rem' }}>
      <div className="caption" style={{ marginBottom: '0.5rem', fontWeight: 600 }}>
        {title}
      </div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
        {counts.map((c, i) => {
          const barHeight = (c / max) * (height - padding * 2)
          return (
            <rect
              key={i}
              x={padding + i * barWidth}
              y={height - padding - barHeight}
              width={Math.max(barWidth - 1, 1)}
              height={barHeight}
              fill="var(--primary)"
              opacity={0.85}
            />
          )
        })}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--border)" />
        <text x={padding} y={height - 8} fontSize="10" fill="var(--text-caption)">
          {binEdges[0].toFixed(2)}
        </text>
        <text x={width - padding} y={height - 8} fontSize="10" fill="var(--text-caption)" textAnchor="end">
          {binEdges[binEdges.length - 1].toFixed(2)}
        </text>
      </svg>
    </div>
  )
}
