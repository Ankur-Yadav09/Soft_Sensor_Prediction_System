function gradeEmoji(r2: number): string {
  if (r2 >= 0.85) return '🟢'
  if (r2 >= 0.75) return '🟡'
  return '🔴'
}

interface KpiCardProps {
  name: string
  r2: number
  mae: number
}

export function KpiCard({ name, r2, mae }: KpiCardProps) {
  return (
    <div className="status-card" style={{ flex: 1, minWidth: 180 }}>
      <div className="caption" style={{ marginBottom: '0.4rem' }}>
        {name}
      </div>
      <div className="metric-value">
        {gradeEmoji(r2)} R² {r2.toFixed(4)}
      </div>
      <div className="caption" style={{ marginTop: '0.25rem' }}>
        MAE {mae.toFixed(4)}
      </div>
    </div>
  )
}
