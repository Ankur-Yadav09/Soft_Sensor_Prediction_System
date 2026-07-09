import { useState } from 'react'

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

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '—'
  const abs = Math.abs(v)
  if (abs === 0) return '0'
  if (abs >= 100) return v.toFixed(0)
  if (abs >= 1) return v.toFixed(2)
  return v.toFixed(4)
}

// Plain-SVG multi-series line chart — no charting library dependency,
// consistent with MiniHistogram/MiniBoxplot. Gridlines + axis ticks + a
// hover crosshair make it readable without guessing values from two labels.
export function LineChart({ series, title, xLabel, yLabel, height = 260 }: LineChartProps) {
  const width = 480
  const padding = { top: 20, right: 20, bottom: 40, left: 50 }
  const plotW = width - padding.left - padding.right
  const plotH = height - padding.top - padding.bottom
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const allValues = series.flatMap((s) => s.data)
  const maxLen = Math.max(...series.map((s) => s.data.length), 1)
  const yMin = Math.min(...allValues, 0)
  const yMax = allValues.length > 0 ? Math.max(...allValues) : 1
  const yRange = yMax - yMin || 1

  const scaleX = (i: number) => padding.left + (i / Math.max(maxLen - 1, 1)) * plotW
  const scaleY = (v: number) => padding.top + plotH - ((v - yMin) / yRange) * plotH

  const Y_TICKS = 4
  const yTicks = Array.from({ length: Y_TICKS + 1 }, (_, i) => yMin + (yRange * i) / Y_TICKS)

  const xTickCount = Math.min(6, maxLen)
  const xTickIndices = Array.from(
    new Set(
      Array.from({ length: xTickCount }, (_, i) =>
        Math.round((i * (maxLen - 1)) / Math.max(xTickCount - 1, 1)),
      ),
    ),
  )

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const userX = ((e.clientX - rect.left) / rect.width) * width
    const idx = Math.round(((userX - padding.left) / plotW) * (maxLen - 1))
    setHoverIdx(Math.min(Math.max(idx, 0), maxLen - 1))
  }

  return (
    <div className="card" style={{ padding: '1rem' }}>
      <div className="caption" style={{ marginBottom: '0.5rem', fontWeight: 600 }}>
        {title}
      </div>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ cursor: 'crosshair' }}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={padding.left}
              y1={scaleY(t)}
              x2={padding.left + plotW}
              y2={scaleY(t)}
              stroke="var(--border)"
              strokeDasharray="2,4"
            />
            <text x={padding.left - 8} y={scaleY(t) + 3} fontSize="9" fill="var(--text-caption)" textAnchor="end">
              {formatNum(t)}
            </text>
          </g>
        ))}
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotH} stroke="var(--border)" />
        <line
          x1={padding.left}
          y1={padding.top + plotH}
          x2={padding.left + plotW}
          y2={padding.top + plotH}
          stroke="var(--border)"
        />
        {xTickIndices.map((idx) => (
          <text
            key={idx}
            x={scaleX(idx)}
            y={padding.top + plotH + 12}
            fontSize="9"
            fill="var(--text-caption)"
            textAnchor="middle"
          >
            {idx + 1}
          </text>
        ))}
        {series.map((s) => (
          <polyline
            key={s.label}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeLinejoin="round"
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
        {hoverIdx !== null && (
          <g>
            <line
              x1={scaleX(hoverIdx)}
              y1={padding.top}
              x2={scaleX(hoverIdx)}
              y2={padding.top + plotH}
              stroke="var(--text-caption)"
              strokeDasharray="3,3"
            />
            {series.map(
              (s) =>
                s.data[hoverIdx] !== undefined && (
                  <circle
                    key={s.label}
                    cx={scaleX(hoverIdx)}
                    cy={scaleY(s.data[hoverIdx])}
                    r={3}
                    fill={s.color}
                    stroke="var(--bg-page)"
                    strokeWidth={1}
                  />
                ),
            )}
            <g transform={`translate(${Math.min(scaleX(hoverIdx) + 10, width - 132)}, ${padding.top + 4})`}>
              <rect
                width={128}
                height={16 + series.length * 14}
                rx={4}
                fill="var(--bg-page)"
                stroke="var(--border)"
              />
              <text x={8} y={14} fontSize="9" fontWeight={700} fill="var(--text-main)">
                Epoch {hoverIdx + 1}
              </text>
              {series.map(
                (s, i) =>
                  s.data[hoverIdx] !== undefined && (
                    <text key={s.label} x={8} y={28 + i * 14} fontSize="9" fill={s.color}>
                      {s.label}: {formatNum(s.data[hoverIdx])}
                    </text>
                  ),
              )}
            </g>
          </g>
        )}
      </svg>
      <div style={{ display: 'flex', gap: '1rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        {series.map((s) => (
          <span key={s.label} className="caption" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: 10, height: 10, background: s.color, display: 'inline-block', borderRadius: 2 }} />
            {s.label}
            {s.data.length > 0 && (
              <strong style={{ color: 'var(--text-main)' }}>{formatNum(s.data[s.data.length - 1])}</strong>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}
