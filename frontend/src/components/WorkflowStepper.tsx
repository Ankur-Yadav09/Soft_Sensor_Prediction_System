// Reusable 5-phase workflow stepper, matching the platform's story: Connect
// Data -> Data Health -> AI Feature Discovery -> Build Soft Sensor -> Live
// Prediction. Each phase maps 1:1 to an existing page (Upload / Preprocess /
// Feature Selection / Train / Predict) — this is the same phase language
// used in the sidebar reference the user shared earlier, deliberately kept
// as a standalone reusable component so later pages can adopt it too.
const PHASES = [
  { key: 'connect', label: 'Connect Data', subtitle: 'Select dataset', icon: '🔌' },
  { key: 'health', label: 'Data Health', subtitle: 'Quality check', icon: '❤️' },
  { key: 'discovery', label: 'AI Feature Discovery', subtitle: 'Variable selection', icon: '🔍' },
  { key: 'build', label: 'Build Soft Sensor', subtitle: 'Train model', icon: '🧠' },
  { key: 'predict', label: 'Live Prediction', subtitle: 'Go live', icon: '📡' },
] as const

export type WorkflowPhaseKey = (typeof PHASES)[number]['key']

interface WorkflowStepperProps {
  current: WorkflowPhaseKey
}

export function WorkflowStepper({ current }: WorkflowStepperProps) {
  const currentIndex = PHASES.findIndex((p) => p.key === current)

  return (
    <div className="card" style={{ padding: '1.1rem 1.5rem', overflowX: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', minWidth: 640 }}>
        {PHASES.map((phase, i) => {
          const isCurrent = i === currentIndex
          const isPast = i < currentIndex
          return (
            <div key={phase.key} style={{ display: 'flex', alignItems: 'center', flex: i === PHASES.length - 1 ? 'none' : 1 }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem', minWidth: 110 }}>
                <div
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.1rem',
                    background: isCurrent
                      ? 'linear-gradient(135deg, #4da6ff 0%, #2563eb 100%)'
                      : isPast
                        ? 'var(--success-bg)'
                        : 'var(--control-bg)',
                    color: isCurrent ? 'white' : isPast ? 'var(--success-text)' : 'var(--text-caption)',
                    boxShadow: isCurrent ? '0 3px 8px rgba(37, 99, 235, 0.35)' : 'none',
                    border: isCurrent ? 'none' : '1px solid var(--border)',
                  }}
                >
                  {isPast ? '✓' : isCurrent ? phase.icon : i + 1}
                </div>
                <span
                  style={{
                    fontSize: '0.78rem',
                    textAlign: 'center',
                    fontWeight: isCurrent ? 700 : 500,
                    color: isCurrent ? 'var(--text-main)' : 'var(--text-caption)',
                  }}
                >
                  {phase.label}
                </span>
                <span
                  style={{
                    fontSize: '0.7rem',
                    textAlign: 'center',
                    color: 'var(--text-caption)',
                  }}
                >
                  {phase.subtitle}
                </span>
              </div>
              {i < PHASES.length - 1 && (
                <div
                  style={{
                    flex: 1,
                    height: 2,
                    background: isPast ? 'var(--success-text)' : 'var(--border)',
                    marginBottom: '1.4rem',
                  }}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
