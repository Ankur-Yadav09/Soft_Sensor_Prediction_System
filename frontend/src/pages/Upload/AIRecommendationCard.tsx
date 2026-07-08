interface AIRecommendationCardProps {
  datasetName: string
}

export function AIRecommendationCard({ datasetName }: AIRecommendationCardProps) {
  return (
    <div
      className="card"
      style={{
        padding: '1.25rem 1.5rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '1rem',
        background: 'var(--success-bg)',
        border: '1px solid rgba(15, 95, 66, 0.18)',
      }}
    >
      <div
        style={{
          width: 40,
          height: 40,
          flexShrink: 0,
          borderRadius: '50%',
          background: 'var(--success-text)',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '1.15rem',
        }}
      >
        🤖
      </div>
      <div>
        <p style={{ margin: 0, fontWeight: 700, color: 'var(--success-text)' }}>
          ✅ Dataset loaded successfully — <span style={{ fontWeight: 400 }}>{datasetName}</span>
        </p>
        <p className="caption" style={{ margin: '0.3rem 0 0' }}>
          Next step: Continue to Data Health Assessment to check for missing values, outliers, and data quality issues.
        </p>
      </div>
    </div>
  )
}
