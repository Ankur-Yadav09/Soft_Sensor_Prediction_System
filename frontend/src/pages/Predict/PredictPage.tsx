import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { listDatasets } from '../../api/datasets'
import { getOverview } from '../../api/overview'
import { listProjects } from '../../api/preprocess'
import { runPredict } from '../../api/predict'
import { Callout } from '../../components/Callout'
import { StepHeading } from '../../components/StepHeading'
import { WorkflowStepper } from '../../components/WorkflowStepper'
import { PredictCharts } from './PredictCharts'

function downloadCsv(rows: Record<string, unknown>[], filename: string) {
  if (rows.length === 0) return
  const cols = Object.keys(rows[0])
  const lines = [
    cols.join(','),
    ...rows.map((r) => cols.map((c) => JSON.stringify(r[c] ?? '')).join(',')),
  ]
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function PredictPage() {
  const [modelName, setModelName] = useState('')
  const [source, setSource] = useState<'project_test' | 'dataset'>('project_test')
  const [projectId, setProjectId] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [rowStart, setRowStart] = useState<number | ''>('')
  const [rowEnd, setRowEnd] = useState<number | ''>('')

  const overviewQuery = useQuery({ queryKey: ['overview'], queryFn: getOverview })
  const projectsQuery = useQuery({ queryKey: ['projects'], queryFn: listProjects })
  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })

  const predictMutation = useMutation({ mutationFn: runPredict })

  const canRun =
    !!modelName &&
    (source === 'project_test' ? !!projectId : !!datasetName) &&
    !predictMutation.isPending

  function run() {
    predictMutation.mutate({
      model_name: modelName,
      source,
      project_id: source === 'project_test' ? projectId : undefined,
      dataset_name: source === 'dataset' ? datasetName : undefined,
      row_start: source === 'dataset' && rowStart !== '' ? Number(rowStart) : undefined,
      row_end: source === 'dataset' && rowEnd !== '' ? Number(rowEnd) : undefined,
    })
  }

  const result = predictMutation.data
  const previewRows = result?.rows.slice(0, 50) ?? []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <h1>Predict</h1>

      <WorkflowStepper current="predict" />

      <div className="card" style={{ padding: '1.5rem' }}>
        <StepHeading step={1} title="Choose a Model" />
        <select value={modelName} onChange={(e) => setModelName(e.target.value)} style={{ minWidth: 320 }}>
          <option value="">Select a saved model…</option>
          {(overviewQuery.data?.saved_models ?? []).map((m) => (
            <option key={m.name} value={m.name}>
              {m.name} ({m.algorithm ?? 'unknown'})
            </option>
          ))}
        </select>
      </div>

      <div className="card" style={{ padding: '1.5rem' }}>
        <StepHeading step={2} title="Choose Data Source" />
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <button
            className={`chip${source === 'project_test' ? ' active' : ''}`}
            onClick={() => setSource('project_test')}
          >
            Project Test Split
          </button>
          <button className={`chip${source === 'dataset' ? ' active' : ''}`} onClick={() => setSource('dataset')}>
            Stored Dataset
          </button>
        </div>

        {source === 'project_test' ? (
          <select value={projectId} onChange={(e) => setProjectId(e.target.value)} style={{ minWidth: 320 }}>
            <option value="">Select a project…</option>
            {(projectsQuery.data ?? []).map((p) => (
              <option key={p.project_id} value={p.project_id}>
                {p.project_id} — {p.dataset_name} ({p.n_test} test rows)
              </option>
            ))}
          </select>
        ) : (
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <label>
              <div className="caption">Dataset</div>
              <select value={datasetName} onChange={(e) => setDatasetName(e.target.value)} style={{ minWidth: 260 }}>
                <option value="">Select…</option>
                {(datasetsQuery.data ?? []).map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <div className="caption">Row start (optional)</div>
              <input
                type="number"
                value={rowStart}
                onChange={(e) => setRowStart(e.target.value === '' ? '' : Number(e.target.value))}
                style={{ width: 100 }}
              />
            </label>
            <label>
              <div className="caption">Row end (optional)</div>
              <input
                type="number"
                value={rowEnd}
                onChange={(e) => setRowEnd(e.target.value === '' ? '' : Number(e.target.value))}
                style={{ width: 100 }}
              />
            </label>
          </div>
        )}

        <button style={{ marginTop: '1.25rem' }} disabled={!canRun} onClick={run}>
          {predictMutation.isPending ? 'Running…' : '🔮 Run Prediction'}
        </button>

        {predictMutation.isError && (
          <div style={{ marginTop: '0.75rem' }}>
            <Callout variant="error">
              {(predictMutation.error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                'Prediction failed.'}
            </Callout>
          </div>
        )}
      </div>

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {result.metrics && (
            <div>
              <h2 style={{ fontSize: '1.05rem', marginBottom: '0.75rem' }}>Metrics</h2>
              <div style={{ overflowX: 'auto' }}>
                <table>
                  <thead>
                    <tr>
                      {Object.keys(result.metrics[0]).map((k) => (
                        <th key={k}>{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.metrics.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((v, j) => (
                          <td key={j}>{typeof v === 'number' ? v.toFixed(4) : String(v)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <PredictCharts result={result} />

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h2 style={{ fontSize: '1.05rem' }}>
                Predictions ({result.rows.length} rows{result.rows.length > 50 ? ', showing first 50' : ''})
              </h2>
              <button onClick={() => downloadCsv(result.rows, `predictions_${modelName}.csv`)}>⬇️ Export CSV</button>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table>
                <thead>
                  <tr>
                    {Object.keys(previewRows[0] ?? {}).map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((v, j) => (
                        <td key={j}>{typeof v === 'number' ? v.toFixed(4) : String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
