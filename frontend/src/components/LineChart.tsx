interface Series {
  label: string
  color: string
  data: number[]
}

interface LineChartProps {
  series: Series[]
  title: string
  xLabel?: string
  yLabel?: string
  height?: number
}

// Plain-SVG multi-series line chart — no charting library dependency,
// consistent with MiniHistogram/MiniBoxplot.
export function LineChart({ series, title, xLabel, yLabel, height = 260 }: LineChartProps) {
  const width = 480
  const padding = { top: 20, right: 20, bottom: 40, left: 50 }
  const plotW = width - padding.left - padding.right
  const plotH = height - padding.top - padding.bottom

  const allValues = series.flatMap((s) => s.data)
  const maxLen = Math.max(...series.map((s) => s.data.length), 1)
  const yMin = Math.min(...allValues, 0)
  const yMax = Math.max(...allValues, 1)
  const yRange = yMax - yMin || 1

  const scaleX = (i: number) => padding.left + (i / Math.max(maxLen - 1, 1)) * plotW
  const scaleY = (v: number) => padding.top + plotH - ((v - yMin) / yRange) * plotH

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
        <text x={padding.left - 8} y={padding.top + 4} fontSize="9" fill="var(--text-caption)" textAnchor="end">
          {yMax.toFixed(2)}
        </text>
        <text x={padding.left - 8} y={padding.top + plotH} fontSize="9" fill="var(--text-caption)" textAnchor="end">
          {yMin.toFixed(2)}
        </text>
        {series.map((s) => (
          <polyline
            key={s.label}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            points={s.data.map((v, i) => `${scaleX(i)},${scaleY(v)}`).join(' ')}
          />
        ))}
        {xLabel && (
          <text x={padding.left + plotW / 2} y={height - 4} fontSize="10" fill="var(--text-caption)" textAnchor="middle">
            {xLabel}
          </text>
        )}
        {yLabel && (
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
        )}
      </svg>
      <div style={{ display: 'flex', gap: '1rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        {series.map((s) => (
          <span key={s.label} className="caption" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: 10, height: 10, background: s.color, display: 'inline-block', borderRadius: 2 }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  )
}
