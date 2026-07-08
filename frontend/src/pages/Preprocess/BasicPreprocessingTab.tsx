import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { applyBasicCleaning } from '../../api/preprocess'
import { Callout } from '../../components/Callout'
import { SectionBanner } from '../../components/SectionBanner'
import { Tabs } from '../../components/Tabs'
import type { FeatureStat } from '../../api/types'
import type { CleaningResponse } from '../../api/preprocess'

const IMPUTE_METHODS = ['None', 'Mean', 'Median', 'Mode', 'Forward Fill', 'Backward Fill', 'Custom Value']
const OUTLIER_METHODS = [
  'None',
  'IQR Capping',
  'Z-Score Capping',
  'Winsorization',
  'Capping/Flooring (custom IQR multiplier)',
  'Remove Outliers (IQR)',
  'Remove Outliers (Z-Score)',
]

interface BasicPreprocessingTabProps {
  datasetName: string
  numericCols: string[]
  stats: FeatureStat[]
  onCleaned: (result: CleaningResponse) => void
}

function MultiToggle({
  options,
  selected,
  onToggle,
}: {
  options: string[]
  selected: Set<string>
  onToggle: (v: string) => void
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
      {options.map((o) => (
        <button
          key={o}
          className={`chip${selected.has(o) ? ' active' : ''}`}
          onClick={() => onToggle(o)}
          style={{ padding: '0.25rem 0.6rem', fontSize: '0.78rem' }}
        >
          {o}
        </button>
      ))}
    </div>
  )
}

export function BasicPreprocessingTab({ datasetName, numericCols, stats, onCleaned }: BasicPreprocessingTabProps) {
  const queryClient = useQueryClient()

  // --- Remove Records ---
  const [removeMissingRows, setRemoveMissingRows] = useState(false)
  const [removeDuplicates, setRemoveDuplicates] = useState(false)
  const [removeMissingCols, setRemoveMissingCols] = useState(false)
  const [missingColThreshold, setMissingColThreshold] = useState(50)
  const [removeConstantCols, setRemoveConstantCols] = useState(false)
  const [removeNzvCols, setRemoveNzvCols] = useState(false)
  const [nzvThreshold, setNzvThreshold] = useState(0.01)

  // --- Missing Values ---
  const [imputeCols, setImputeCols] = useState<Set<string>>(new Set())
  const [imputeMethod, setImputeMethod] = useState('None')
  const [customFillValue, setCustomFillValue] = useState(0)

  // --- Outlier Treatment ---
  const [outlierMethod, setOutlierMethod] = useState('None')
  const [outlierCols, setOutlierCols] = useState<Set<string>>(new Set(numericCols))
  const [zscoreThreshold, setZscoreThreshold] = useState(3.0)
  const [winsorLo, setWinsorLo] = useState(2.5)
  const [winsorHi, setWinsorHi] = useState(97.5)
  const [capMultiplier, setCapMultiplier] = useState(1.5)

  // --- Domain Filters ---
  const [filterTags, setFilterTags] = useState<Set<string>>(new Set())
  const [domainBounds, setDomainBounds] = useState<Record<string, { min: number; max: number }>>({})

  const statsByFeature = Object.fromEntries(stats.map((s) => [s.Feature, s]))

  function toggleSet(set: Set<string>, setter: (s: Set<string>) => void, v: string) {
    const next = new Set(set)
    if (next.has(v)) next.delete(v)
    else next.add(v)
    setter(next)
  }

  function toggleFilterTag(tag: string) {
    const next = new Set(filterTags)
    if (next.has(tag)) {
      next.delete(tag)
    } else {
      next.add(tag)
      const s = statsByFeature[tag]
      if (s && !domainBounds[tag]) {
        setDomainBounds((prev) => ({ ...prev, [tag]: { min: s.Min ?? 0, max: s.Max ?? 0 } }))
      }
    }
    setFilterTags(next)
  }

  const cleanMutation = useMutation({
    mutationFn: applyBasicCleaning,
    onSuccess: (res) => {
      onCleaned(res)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    },
  })

  const activeSteps: string[] = []
  if (removeMissingRows) activeSteps.push('Remove missing rows')
  if (removeDuplicates) activeSteps.push('Remove duplicates')
  if (removeMissingCols) activeSteps.push(`Remove columns (missing ≥ ${missingColThreshold}%)`)
  if (removeConstantCols) activeSteps.push('Remove constant columns')
  if (removeNzvCols) activeSteps.push(`Remove NZV columns (std < ${nzvThreshold})`)
  if (imputeMethod !== 'None') activeSteps.push(`Impute: ${imputeMethod}`)
  if (outlierMethod !== 'None') activeSteps.push(`Outliers: ${outlierMethod}`)
  if (filterTags.size > 0) activeSteps.push(`Domain filters: ${filterTags.size} tag(s)`)

  function apply() {
    const domain_filters =
      filterTags.size > 0
        ? Object.fromEntries([...filterTags].map((t) => [t, domainBounds[t]]))
        : undefined
    cleanMutation.mutate({
      dataset_name: datasetName,
      remove_missing_rows: removeMissingRows,
      remove_duplicates: removeDuplicates,
      remove_missing_cols: removeMissingCols,
      missing_col_threshold: missingColThreshold,
      remove_constant_cols: removeConstantCols,
      remove_nzv_cols: removeNzvCols,
      nzv_threshold: nzvThreshold,
      impute_method: imputeMethod,
      impute_cols: imputeCols.size > 0 ? [...imputeCols] : undefined,
      custom_fill_value: customFillValue,
      outlier_method: outlierMethod,
      outlier_cols: outlierMethod !== 'None' ? [...outlierCols] : undefined,
      zscore_threshold: zscoreThreshold,
      winsor_lo: winsorLo,
      winsor_hi: winsorHi,
      cap_multiplier: capMultiplier,
      domain_filters,
    })
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <SectionBanner
        icon="⚙️"
        title="Basic Preprocessing"
        subtitle="Remove records, impute missing values, handle outliers, and apply domain filters. Click 'Apply Cleaning' to save a new cleaned dataset."
      />

      <Tabs
        tabs={[
          {
            label: '🗑️ Remove Records',
            content: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', gap: '2rem' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <input type="checkbox" checked={removeMissingRows} onChange={(e) => setRemoveMissingRows(e.target.checked)} />
                    Remove rows with any missing values
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <input type="checkbox" checked={removeDuplicates} onChange={(e) => setRemoveDuplicates(e.target.checked)} />
                    Remove duplicate records
                  </label>
                </div>
                <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <input type="checkbox" checked={removeMissingCols} onChange={(e) => setRemoveMissingCols(e.target.checked)} />
                    Remove columns with missing values ≥
                  </label>
                  <input
                    type="number"
                    value={missingColThreshold}
                    onChange={(e) => setMissingColThreshold(Number(e.target.value))}
                    style={{ width: 70 }}
                    disabled={!removeMissingCols}
                  />
                  %
                </div>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <input type="checkbox" checked={removeConstantCols} onChange={(e) => setRemoveConstantCols(e.target.checked)} />
                  Remove constant columns (std = 0)
                </label>
                <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <input type="checkbox" checked={removeNzvCols} onChange={(e) => setRemoveNzvCols(e.target.checked)} />
                    Remove near-zero variance columns (std &lt;
                  </label>
                  <input
                    type="number"
                    step={0.001}
                    value={nzvThreshold}
                    onChange={(e) => setNzvThreshold(Number(e.target.value))}
                    style={{ width: 90 }}
                    disabled={!removeNzvCols}
                  />
                  )
                </div>
              </div>
            ),
          },
          {
            label: '🔧 Missing Values',
            content: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div className="caption">Apply to columns (leave empty for all columns with missing values)</div>
                <MultiToggle options={numericCols} selected={imputeCols} onToggle={(v) => toggleSet(imputeCols, setImputeCols, v)} />
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', marginTop: '0.5rem' }}>
                  <label>
                    <div className="caption">Imputation Method</div>
                    <select value={imputeMethod} onChange={(e) => setImputeMethod(e.target.value)}>
                      {IMPUTE_METHODS.map((m) => (
                        <option key={m}>{m}</option>
                      ))}
                    </select>
                  </label>
                  {imputeMethod === 'Custom Value' && (
                    <label>
                      <div className="caption">Fill Value</div>
                      <input type="number" value={customFillValue} onChange={(e) => setCustomFillValue(Number(e.target.value))} style={{ width: 100 }} />
                    </label>
                  )}
                </div>
              </div>
            ),
          },
          {
            label: '📊 Outlier Treatment',
            content: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
                  <label>
                    <div className="caption">Outlier Treatment Method</div>
                    <select value={outlierMethod} onChange={(e) => setOutlierMethod(e.target.value)} style={{ minWidth: 280 }}>
                      {OUTLIER_METHODS.map((m) => (
                        <option key={m}>{m}</option>
                      ))}
                    </select>
                  </label>
                  {outlierMethod === 'Z-Score Capping' || outlierMethod === 'Remove Outliers (Z-Score)' ? (
                    <label>
                      <div className="caption">Z-Score threshold</div>
                      <input type="number" value={zscoreThreshold} onChange={(e) => setZscoreThreshold(Number(e.target.value))} style={{ width: 90 }} />
                    </label>
                  ) : null}
                  {outlierMethod === 'Winsorization' && (
                    <>
                      <label>
                        <div className="caption">Lower %</div>
                        <input type="number" value={winsorLo} onChange={(e) => setWinsorLo(Number(e.target.value))} style={{ width: 90 }} />
                      </label>
                      <label>
                        <div className="caption">Upper %</div>
                        <input type="number" value={winsorHi} onChange={(e) => setWinsorHi(Number(e.target.value))} style={{ width: 90 }} />
                      </label>
                    </>
                  )}
                  {outlierMethod === 'Capping/Flooring (custom IQR multiplier)' && (
                    <label>
                      <div className="caption">IQR multiplier</div>
                      <input type="number" value={capMultiplier} onChange={(e) => setCapMultiplier(Number(e.target.value))} style={{ width: 90 }} />
                    </label>
                  )}
                </div>
                <div className="caption">Apply to columns</div>
                <MultiToggle options={numericCols} selected={outlierCols} onToggle={(v) => toggleSet(outlierCols, setOutlierCols, v)} />
              </div>
            ),
          },
          {
            label: '📐 Domain Filters',
            content: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div className="caption">Select features to filter</div>
                <MultiToggle options={numericCols} selected={filterTags} onToggle={toggleFilterTag} />
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.5rem' }}>
                  {[...filterTags].map((tag) => (
                    <div key={tag} className="card" style={{ padding: '0.75rem 1rem' }}>
                      <div style={{ fontWeight: 700, color: 'var(--primary)' }}>{tag}</div>
                      <div className="caption">
                        Data range: {statsByFeature[tag]?.Min?.toPrecision(4)} — {statsByFeature[tag]?.Max?.toPrecision(4)}
                      </div>
                      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.4rem' }}>
                        <label>
                          <div className="caption">Min</div>
                          <input
                            type="number"
                            value={domainBounds[tag]?.min ?? 0}
                            onChange={(e) =>
                              setDomainBounds((prev) => ({ ...prev, [tag]: { ...prev[tag], min: Number(e.target.value) } }))
                            }
                            style={{ width: 90 }}
                          />
                        </label>
                        <label>
                          <div className="caption">Max</div>
                          <input
                            type="number"
                            value={domainBounds[tag]?.max ?? 0}
                            onChange={(e) =>
                              setDomainBounds((prev) => ({ ...prev, [tag]: { ...prev[tag], max: Number(e.target.value) } }))
                            }
                            style={{ width: 90 }}
                          />
                        </label>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ),
          },
        ]}
      />

      <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
        {activeSteps.length > 0 ? (
          <p className="caption">Configured steps: {activeSteps.join(' → ')}</p>
        ) : (
          <p className="caption">No preprocessing steps configured.</p>
        )}
        <button onClick={apply} disabled={cleanMutation.isPending}>
          {cleanMutation.isPending ? 'Applying…' : '✅ Apply Cleaning'}
        </button>

        {cleanMutation.isError && (
          <div style={{ marginTop: '0.75rem' }}>
            <Callout variant="error">Failed to apply cleaning.</Callout>
          </div>
        )}

        {cleanMutation.data && (
          <div style={{ marginTop: '0.75rem' }}>
            <Callout variant="success">
              Cleaning complete. Records: {cleanMutation.data.before_rows} → {cleanMutation.data.after_rows}. Saved as{' '}
              <code>{cleanMutation.data.new_dataset_name}</code>.
            </Callout>
            <ul className="caption">
              {(cleanMutation.data.action_log ?? []).map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
