import { useMutation, useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { listDatasets } from '../../api/datasets'
import { submitFeatureSelection } from '../../api/featureSelection'
import { getFeatureStats } from '../../api/preprocess'
import { Callout } from '../../components/Callout'
import { MultiSelectDropdown } from '../../components/MultiSelectDropdown'
import { SectionBanner } from '../../components/SectionBanner'
import { StepHeading } from '../../components/StepHeading'
import { Tabs } from '../../components/Tabs'
import { WorkflowStepper } from '../../components/WorkflowStepper'
import { useJobPolling } from '../../hooks/useJobPolling'
import { useActiveDataset } from '../../state/ActiveDatasetContext'
import { FeatureSelectionResults } from './FeatureSelectionResults'
import { FinalApply } from './FinalApply'
import { ManualVariableSelection } from './ManualVariableSelection'
import { METHOD_CATEGORIES, METHOD_IDS, METHOD_LABELS } from './methodMeta'
import type { FeatureSelectionResult } from '../../api/types'

type Pathway = 'configure' | 'automated'

export function FeatureSelectionPage() {
  const { activeDataset: datasetName, setActiveDataset: setDatasetName } = useActiveDataset()
  const [yCols, setYCols] = useState<Set<string>>(new Set())
  const [pathway, setPathway] = useState<Pathway>('automated')

  // Configure-pathway settings
  const [topK, setTopK] = useState(10)
  const [corrThreshold, setCorrThreshold] = useState(0.85)
  const [vifThreshold, setVifThreshold] = useState(10.0)
  const [enabledMethods, setEnabledMethods] = useState<Set<string>>(new Set(METHOD_IDS))
  const [perTarget, setPerTarget] = useState(false)

  const [jobId, setJobId] = useState<string | null>(null)
  const [xCols, setXCols] = useState<Set<string>>(new Set())

  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })
  const statsQuery = useQuery({
    queryKey: ['preprocess-stats', datasetName],
    queryFn: () => getFeatureStats(datasetName),
    enabled: !!datasetName,
  })
  const numericCols = useMemo(
    () => (statsQuery.data ?? []).filter((s) => s.Mean !== null).map((s) => s.Feature),
    [statsQuery.data],
  )
  const candidateX = numericCols.filter((c) => !yCols.has(c))

  const submitMutation = useMutation({
    mutationFn: submitFeatureSelection,
    onSuccess: (id) => setJobId(id),
  })

  const jobQuery = useJobPolling(jobId)
  const result = jobQuery.data?.status === 'done' ? (jobQuery.data.result as FeatureSelectionResult) : null

  function toggleY(col: string) {
    const next = new Set(yCols)
    if (next.has(col)) next.delete(col)
    else next.add(col)
    setYCols(next)
    setJobId(null)
  }

  function toggleMethod(mid: string) {
    const next = new Set(enabledMethods)
    if (next.has(mid)) next.delete(mid)
    else next.add(mid)
    setEnabledMethods(next)
  }

  function runConfigure() {
    submitMutation.mutate({
      dataset_name: datasetName,
      y_cols: [...yCols],
      x_cols: candidateX,
      top_k: topK,
      corr_threshold: corrThreshold,
      vif_threshold: vifThreshold,
      per_target: perTarget,
    })
  }

  function runAutomated() {
    const autoTopK = Math.min(Math.max(10, Math.floor(candidateX.length / 5)), 30)
    submitMutation.mutate({
      dataset_name: datasetName,
      y_cols: [...yCols],
      x_cols: candidateX,
      top_k: autoTopK,
      corr_threshold: 0.85,
      vif_threshold: 10.0,
      per_target: perTarget,
    })
  }

  function applyQuickPick(features: string[]) {
    setXCols(new Set(features))
  }

  const methodsByCategory = useMemo(() => {
    const map: Record<string, string[]> = {}
    for (const mid of METHOD_IDS) {
      const cat = METHOD_CATEGORIES[mid]
      map[cat] = [...(map[cat] ?? []), mid]
    }
    return map
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <h1>Feature Selection</h1>

      <WorkflowStepper current="discovery" />

      <div>
        <StepHeading step={1} title="Select Target (Y) Variable" />
        <SectionBanner
          icon="🎯"
          title="Target Variable Selection"
          subtitle="Choose one or more columns to predict. These become Y; all others are candidate X features."
        />
      </div>

      <div className="card" style={{ padding: '1.5rem' }}>
        <label>
          <div className="caption">Dataset</div>
          <select
            value={datasetName}
            onChange={(e) => {
              setDatasetName(e.target.value)
              setYCols(new Set())
              setJobId(null)
            }}
            style={{ minWidth: 320 }}
          >
            <option value="">Select…</option>
            {(datasetsQuery.data ?? []).map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
              </option>
            ))}
          </select>
          {datasetName && (
            <p className="caption" style={{ marginTop: '0.4rem' }}>
              Carried over from your last step. Pick a different dataset above if needed.
            </p>
          )}
        </label>

        {datasetName && (
          <div style={{ marginTop: '1rem' }}>
            <div className="caption" style={{ marginBottom: '0.4rem' }}>
              Target Variable(s)
            </div>
            <MultiSelectDropdown
              options={numericCols}
              selected={yCols}
              onChange={(next) => {
                setYCols(next)
                setJobId(null)
              }}
            />
          </div>
        )}

        {datasetName && yCols.size === 0 && (
          <div style={{ marginTop: '1rem' }}>
            <Callout variant="info">Select at least one target (Y) variable to proceed.</Callout>
          </div>
        )}
      </div>

      {datasetName && yCols.size > 0 && (
        <>
          <p className="caption">
            {candidateX.length} candidate X features · {yCols.size} target(s): <code>{[...yCols].join(', ')}</code>
          </p>

          <div>
            <h2 style={{ fontSize: '1.05rem', marginBottom: '0.75rem' }}>Choose Feature Selection Mode</h2>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button className={`chip${pathway === 'automated' ? ' active' : ''}`} onClick={() => setPathway('automated')}>
                ⚡ Automated Feature Selection
              </button>
              <button className={`chip${pathway === 'configure' ? ' active' : ''}`} onClick={() => setPathway('configure')}>
                🔧 Configure Feature Selection
              </button>
            </div>
          </div>

          {pathway === 'automated' ? (
            <div className="card" style={{ padding: '1.5rem' }}>
              <StepHeading step={2} title="Automated Feature Selection" />
              <SectionBanner
                icon="⚡"
                title="Automated Feature Selection"
                subtitle="Runs all available methods with best-default parameters in one click."
              />
              <p className="caption" style={{ margin: '1rem 0' }}>
                Will run {METHOD_IDS.length} independent scoring method(s) · Top-K auto-scaled to feature count ·
                Collinearity threshold = 0.85 · VIF threshold = 10.0
              </p>
              <button disabled={submitMutation.isPending} onClick={runAutomated}>
                {submitMutation.isPending ? 'Running…' : '⚡ Run Automated Feature Selection'}
              </button>
            </div>
          ) : (
            <div className="card" style={{ padding: '1.5rem' }}>
              <StepHeading step={3} title="Configure Feature Selection" />
              <Tabs
                tabs={[
                  {
                    label: '⚙️ Configure Analysis',
                    content: (
                      <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                        <label>
                          <div className="caption">Top-K features per method</div>
                          <input
                            type="number"
                            min={2}
                            max={Math.min(25, candidateX.length)}
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            style={{ width: 80 }}
                          />
                        </label>
                        <label>
                          <div className="caption">X–X collinearity flag threshold</div>
                          <input
                            type="number"
                            min={0.5}
                            max={0.99}
                            step={0.05}
                            value={corrThreshold}
                            onChange={(e) => setCorrThreshold(Number(e.target.value))}
                            style={{ width: 90 }}
                          />
                        </label>
                        <label>
                          <div className="caption">VIF threshold</div>
                          <input
                            type="number"
                            min={2}
                            max={50}
                            step={1}
                            value={vifThreshold}
                            onChange={(e) => setVifThreshold(Number(e.target.value))}
                            style={{ width: 90 }}
                          />
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '1.4rem' }}>
                          <input type="checkbox" checked={perTarget} onChange={(e) => setPerTarget(e.target.checked)} />
                          <span className="caption">Per-target mode (recommended for multi-Y)</span>
                        </label>
                      </div>
                    ),
                  },
                  {
                    label: '📋 Methods Selection',
                    content: (
                      <div style={{ display: 'flex', gap: '2.5rem', flexWrap: 'wrap' }}>
                        {Object.entries(methodsByCategory).map(([cat, mids]) => (
                          <div key={cat}>
                            <strong>{cat}</strong>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', marginTop: '0.4rem' }}>
                              {mids.map((mid) => (
                                <label key={mid} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                  <input type="checkbox" checked={enabledMethods.has(mid)} onChange={() => toggleMethod(mid)} />
                                  {METHOD_LABELS[mid]}
                                </label>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    ),
                  },
                ]}
              />
              <p className="caption" style={{ marginTop: '1rem' }}>
                {enabledMethods.size} of {METHOD_IDS.length} core scoring method(s) selected.
              </p>
              <button
                style={{ marginTop: '1rem' }}
                disabled={enabledMethods.size === 0 || submitMutation.isPending}
                onClick={runConfigure}
              >
                {submitMutation.isPending ? 'Running…' : '🔍 Run Intelligent Feature Selection'}
              </button>
            </div>
          )}

          {jobId && jobQuery.data && !jobQuery.data.done && (
            <Callout variant="info">
              ⏳ {jobQuery.data.progress.message ?? 'Running…'} — this may take 20-90 seconds.
            </Callout>
          )}
          {jobQuery.data?.status === 'error' && <Callout variant="error">{jobQuery.data.error}</Callout>}

          {result && (
            <>
              <div>
                <StepHeading step={4} title="Analysis Results" />
                <FeatureSelectionResults result={result} />
              </div>

              <div className="card" style={{ padding: '1.5rem' }}>
                <p className="caption">
                  <strong>Highly Recommended + Recommended X ({result.recommended_features.length}):</strong>{' '}
                  <code>{result.recommended_features.join(', ') || 'None'}</code>
                </p>
                <p className="caption">
                  <strong>Consider X ({result.optional_features.length}):</strong>{' '}
                  <code>{result.optional_features.join(', ') || 'None'}</code>
                </p>
                <div style={{ display: 'flex', gap: '0.6rem', marginTop: '0.75rem' }}>
                  <button onClick={() => applyQuickPick(result.recommended_features)}>
                    ✅ Use Recommended ({result.recommended_features.length})
                  </button>
                  <button
                    onClick={() => applyQuickPick([...result.recommended_features, ...result.optional_features])}
                  >
                    ⭐ Use Rec + Optional ({result.recommended_features.length + result.optional_features.length})
                  </button>
                  <button onClick={() => setXCols(new Set())}>🗑️ Clear Selection</button>
                </div>
              </div>

              <ManualVariableSelection
                numericCols={numericCols}
                xCols={xCols}
                yCols={yCols}
                onToggleX={(c) => {
                  const next = new Set(xCols)
                  if (next.has(c)) next.delete(c)
                  else next.add(c)
                  setXCols(next)
                }}
                onToggleY={toggleY}
                onSelectAllX={(all) => setXCols(all ? new Set(numericCols) : new Set())}
                onSelectAllY={(all) => setYCols(all ? new Set(candidateX) : new Set())}
              />

              <FinalApply datasetName={datasetName} xCols={[...xCols]} yCols={[...yCols]} />
            </>
          )}
        </>
      )}
    </div>
  )
}
