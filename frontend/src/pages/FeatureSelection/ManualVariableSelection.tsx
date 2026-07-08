import { SectionBanner } from '../../components/SectionBanner'

interface ManualVariableSelectionProps {
  numericCols: string[]
  xCols: Set<string>
  yCols: Set<string>
  onToggleX: (col: string) => void
  onToggleY: (col: string) => void
  onSelectAllX: (all: boolean) => void
  onSelectAllY: (all: boolean) => void
}

export function ManualVariableSelection({
  numericCols,
  xCols,
  yCols,
  onToggleX,
  onToggleY,
  onSelectAllX,
  onSelectAllY,
}: ManualVariableSelectionProps) {
  const yOptions = numericCols.filter((c) => !xCols.has(c))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <SectionBanner
        icon="🎛️"
        title="Manual Variable Selection"
        subtitle="Review and adjust the auto-selected X input features and Y target variables."
      />
      <div style={{ display: 'flex', gap: '2.5rem', flexWrap: 'wrap' }}>
        <div>
          <strong>Input Features (X)</strong>
          <p className="caption">{xCols.size} feature(s) selected — check/uncheck to adjust.</p>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
            <input type="checkbox" checked={xCols.size === numericCols.length} onChange={(e) => onSelectAllX(e.target.checked)} />
            Select All X
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', maxHeight: 260, overflowY: 'auto' }}>
            {numericCols.map((c) => (
              <label key={c} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <input type="checkbox" checked={xCols.has(c)} onChange={() => onToggleX(c)} />
                {c}
              </label>
            ))}
          </div>
        </div>
        <div>
          <strong>Target Variables (Y)</strong>
          <p className="caption">{yCols.size} target(s) selected.</p>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
            <input
              type="checkbox"
              checked={yOptions.length > 0 && yCols.size === yOptions.length}
              onChange={(e) => onSelectAllY(e.target.checked)}
            />
            Select All Y
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', maxHeight: 260, overflowY: 'auto' }}>
            {yOptions.map((c) => (
              <label key={c} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <input type="checkbox" checked={yCols.has(c)} onChange={() => onToggleY(c)} />
                {c}
              </label>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
