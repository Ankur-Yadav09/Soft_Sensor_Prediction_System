import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getDatasetPreview, listDatasets } from '../../api/datasets'
import { getFeatureStats } from '../../api/preprocess'
import { Callout } from '../../components/Callout'
import { StepHeading } from '../../components/StepHeading'
import { Tabs } from '../../components/Tabs'
import { WorkflowStepper } from '../../components/WorkflowStepper'
import { useActiveDataset } from '../../state/ActiveDatasetContext'
import { AutomatedPreprocessingTab } from './AutomatedPreprocessingTab'
import { BasicPreprocessingTab } from './BasicPreprocessingTab'
import { DataUnderstandingTab } from './DataUnderstandingTab'
import type { CleaningResponse } from '../../api/preprocess'

export function PreprocessPage() {
  const navigate = useNavigate()
  const { activeDataset: datasetName, setActiveDataset: setDatasetName } = useActiveDataset()
  const [lastCleaned, setLastCleaned] = useState<CleaningResponse | null>(null)

  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })
  const statsQuery = useQuery({
    queryKey: ['preprocess-stats', datasetName],
    queryFn: () => getFeatureStats(datasetName),
    enabled: !!datasetName,
  })
  const previewQuery = useQuery({
    queryKey: ['datasets', datasetName, 'preview'],
    queryFn: () => getDatasetPreview(datasetName),
    enabled: !!datasetName,
  })

  const numericCols = useMemo(
    () => (statsQuery.data ?? []).filter((s) => s.Mean !== null).map((s) => s.Feature),
    [statsQuery.data],
  )

  function onCleaned(result: CleaningResponse) {
    setLastCleaned(result)
    setDatasetName(result.new_dataset_name)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <h1>Data Preprocessing</h1>

      <WorkflowStepper current="health" />

      <div>
        <StepHeading step={1} title="Load Dataset" />
        <label className="caption" style={{ display: 'block', marginBottom: '0.4rem' }}>
          Select Active Dataset
        </label>
        <select value={datasetName} onChange={(e) => setDatasetName(e.target.value)} style={{ minWidth: 320 }}>
          <option value="">Select a dataset…</option>
          {(datasetsQuery.data ?? []).map((d) => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.rows}×{d.cols})
            </option>
          ))}
        </select>
        {datasetName && !lastCleaned && (
          <p className="caption" style={{ marginTop: '0.4rem' }}>
            Carried over from Connect Process Data. Pick a different dataset above if needed.
          </p>
        )}
        {lastCleaned && (
          <div style={{ marginTop: '0.75rem' }}>
            <Callout variant="success">
              Now working on the cleaned dataset <code>{lastCleaned.new_dataset_name}</code> — records:{' '}
              {lastCleaned.before_rows} → {lastCleaned.after_rows}.
            </Callout>
            {(lastCleaned.action_log ?? lastCleaned.step_log) && (
              <ul className="caption" style={{ marginTop: '0.5rem' }}>
                {(lastCleaned.action_log ?? lastCleaned.step_log)!.map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {!datasetName && (
        <Callout variant="warning">Please select a dataset above (connect one on the Connect Process Data page first).</Callout>
      )}

      {datasetName && numericCols.length > 0 && (
        <>
          <div style={{ borderTop: '1px solid var(--border)' }} />
          <div>
            <StepHeading step={2} title="Configure Preprocessing" />
            <Tabs
              tabs={[
                {
                  label: '🔍 Data Understanding',
                  content: <DataUnderstandingTab datasetName={datasetName} numericCols={numericCols} />,
                },
                {
                  label: '🤖 Automated Preprocessing',
                  content: <AutomatedPreprocessingTab datasetName={datasetName} onCleaned={onCleaned} />,
                },
                {
                  label: '⚙️ Basic Preprocessing',
                  content: (
                    <BasicPreprocessingTab
                      datasetName={datasetName}
                      numericCols={numericCols}
                      stats={statsQuery.data ?? []}
                      onCleaned={onCleaned}
                    />
                  ),
                },
              ]}
            />
          </div>

          <div style={{ borderTop: '1px solid var(--border)' }} />

          <div>
            <h2 style={{ fontSize: '1.05rem', marginBottom: '0.75rem' }}>📋 Dataset Preview &amp; Download</h2>
            {previewQuery.data && (
              <>
                <div style={{ display: 'flex', gap: '2rem', marginBottom: '1rem' }}>
                  <Stat label="Rows" value={String(previewQuery.data.shape[0])} />
                  <Stat label="Columns" value={String(previewQuery.data.shape[1])} />
                  <Stat label="Numeric Columns" value={String(numericCols.length)} />
                </div>
                <div className="scroll-rows-5" style={{ overflowX: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        {previewQuery.data.columns.map((c) => (
                          <th key={c}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewQuery.data.head.map((row, i) => (
                        <tr key={i}>
                          {previewQuery.data!.columns.map((c) => (
                            <td key={c}>{String(row[c] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="caption" style={{ marginTop: '0.5rem' }}>
                  Proceed to the <strong>Feature Selection</strong> page to choose X/Y columns, run feature
                  selection, and finalize the train/test split.
                </p>
                <button style={{ marginTop: '0.75rem' }} onClick={() => navigate('/feature-selection')}>
                  Continue to Feature Selection →
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="caption">{label}</div>
      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary)' }}>{value}</div>
    </div>
  )
}
