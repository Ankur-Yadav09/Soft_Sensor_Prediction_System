interface Point {
  x: number
  y: number
}

interface ScatterChartProps {
  points: Point[]
  title: string
  xLabel: string
  yLabel: string
  height?: number
}

// Plain-SVG scatter plot with a y=x reference line (actual vs predicted) —
// no charting library dependency, consistent with MiniHistogram/LineChart.
export function ScatterChart({ points, title, xLabel, yLabel, height = 260 }: ScatterChartProps) {
  const width = 480
  const padding = { top: 20, right: 20, bottom: 40, left: 50 }
  const plotW = width - padding.left - padding.right
  const plotH = height - padding.top - padding.bottom

  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const min = Math.min(...xs, ...ys)
  const max = Math.max(...xs, ...ys)
  const range = max - min || 1

  const scale = (v: number) => (v - min) / range
  const toX = (v: number) => padding.left + scale(v) * plotW
  const toY = (v: number) => padding.top + plotH - scale(v) * plotH

  return (
    <div className="card" style={{ padding: '1rem' }}>
      <div className="caption" style={{ marginBottom: '0.5rem', fontWeight: 600 }}>
        {title}
      </div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotH} stroke="var(--border)" />
        <line
          x1={padding.left}
          y1={padding.top + plotH}
          x2={padding.left + plotW}
          y2={padding.top + plotH}
          stroke="var(--border)"
        />
        <line x1={toX(min)} y1={toY(min)} x2={toX(max)} y2={toY(max)} stroke="var(--text-caption)" strokeDasharray="4 3" />
        {points.map((p, i) => (
          <circle key={i} cx={toX(p.x)} cy={toY(p.y)} r={2.5} fill="var(--primary)" opacity={0.6} />
        ))}
        <text x={padding.left + plotW / 2} y={height - 4} fontSize="10" fill="var(--text-caption)" textAnchor="middle">
          {xLabel}
        </text>
        <text
          x={12}
          y={padding.top + plotH / 2}
          fontSize="10"
          fill="var(--text-caption)"
          textAnchor="middle"
          transform={`rotate(-90, 12, ${padding.top + plotH / 2})`}
        >
          {yLabel}
        </text>
      </svg>
    </div>
  )
}
