interface StatusCardProps {
  label: string
  value: string
  sublabel?: string
  tone?: 'neutral' | 'warning' | 'info' | 'success'
}

const TONE_STYLES: Record<string, { background: string; color: string }> = {
  neutral: { background: 'var(--bg-page)', color: 'var(--text-main)' },
  warning: { background: 'var(--warning-bg)', color: 'var(--warning-text)' },
  info: { background: 'var(--info-bg)', color: 'var(--info-text)' },
  success: { background: 'var(--success-bg)', color: 'var(--success-text)' },
}

export function StatusCard({ label, value, sublabel, tone = 'neutral' }: StatusCardProps) {
  const style = TONE_STYLES[tone]
  return (
    <div
      className="status-card"
      style={{ flex: 1, minWidth: 180, background: style.background, borderColor: 'transparent' }}
    >
      <div className="caption" style={{ marginBottom: '0.4rem', color: tone === 'neutral' ? undefined : style.color }}>
        {label}
      </div>
      <div className="metric-value" style={tone === 'neutral' ? undefined : { color: style.color }}>
        {value}
      </div>
      {sublabel && (
        <div className="caption" style={{ marginTop: '0.25rem', color: tone === 'neutral' ? undefined : style.color }}>
          {sublabel}
        </div>
      )}
    </div>
  )
}
