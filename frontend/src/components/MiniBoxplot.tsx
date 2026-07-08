interface MiniBoxplotProps {
  min: number
  q1: number
  median: number
  q3: number
  max: number
  title: string
}

// Plain-SVG horizontal box plot — no charting library dependency.
export function MiniBoxplot({ min, q1, median, q3, max, title }: MiniBoxplotProps) {
  const width = 420
  const height = 220
  const padding = 40
  const scale = (v: number) => padding + ((v - min) / (max - min || 1)) * (width - padding * 2)
  const midY = height / 2
  const boxHeight = 50

  return (
    <div className="card" style={{ padding: '1rem' }}>
      <div className="caption" style={{ marginBottom: '0.5rem', fontWeight: 600 }}>
        {title}
      </div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
        <line x1={scale(min)} y1={midY} x2={scale(q1)} y2={midY} stroke="var(--primary)" strokeWidth={2} />
        <line x1={scale(q3)} y1={midY} x2={scale(max)} y2={midY} stroke="var(--primary)" strokeWidth={2} />
        <line x1={scale(min)} y1={midY - 10} x2={scale(min)} y2={midY + 10} stroke="var(--primary)" strokeWidth={2} />
        <line x1={scale(max)} y1={midY - 10} x2={scale(max)} y2={midY + 10} stroke="var(--primary)" strokeWidth={2} />
        <rect
          x={scale(q1)}
          y={midY - boxHeight / 2}
          width={scale(q3) - scale(q1)}
          height={boxHeight}
          fill="var(--info-bg)"
          stroke="var(--primary)"
          strokeWidth={2}
        />
        <line x1={scale(median)} y1={midY - boxHeight / 2} x2={scale(median)} y2={midY + boxHeight / 2} stroke="var(--primary)" strokeWidth={2} />
        {[
          ['min', min],
          ['q1', q1],
          ['median', median],
          ['q3', q3],
          ['max', max],
        ].map(([label, v]) => (
          <text key={label as string} x={scale(v as number)} y={midY + boxHeight / 2 + 18} fontSize="9" fill="var(--text-caption)" textAnchor="middle">
            {(v as number).toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  )
}
