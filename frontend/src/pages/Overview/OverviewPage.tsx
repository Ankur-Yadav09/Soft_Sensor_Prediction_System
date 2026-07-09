import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { getModelPerformance, getOverview } from '../../api/overview'
import { Callout } from '../../components/Callout'
import { DataTable } from '../../components/DataTable'
import { KpiCard } from '../../components/KpiCard'
import { StatusCard } from '../../components/StatusCard'
import { WorkflowGuide } from '../../components/WorkflowGuide'
import type { SavedModelSummary } from '../../api/types'

export function OverviewPage() {
  const overviewQuery = useQuery({ queryKey: ['overview'], queryFn: getOverview })

  const mostRecentModel = useMemo(() => {
    const models = overviewQuery.data?.saved_models ?? []
    if (models.length === 0) return null
    return [...models].sort((a, b) => (a.saved_at < b.saved_at ? 1 : -1))[0].name
  }, [overviewQuery.data])

  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const activeModel = selectedModel ?? mostRecentModel

  const performanceQuery = useQuery({
    queryKey: ['model-performance', activeModel],
    queryFn: () => getModelPerformance(activeModel as string),
    enabled: !!activeModel,
  })

  if (overviewQuery.isLoading) return <p className="caption">Loading overview…</p>
  if (overviewQuery.isError) return <p className="caption">Failed to load overview.</p>

  const { datasets, saved_models: savedModels } = overviewQuery.data!

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div>
        <h1>🏭 SoftSense AI — Industrial Intelligence</h1>
        <p className="caption">Industrial AI Platform for Soft Sensor Development and Process Optimization</p>
      </div>

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <StatusCard
          label="Datasets Stored"
          value={String(datasets.length)}
          tone={datasets.length > 0 ? 'info' : 'warning'}
          sublabel={datasets.length === 0 ? 'Upload data to begin' : undefined}
        />
        <StatusCard
          label="Models Trained"
          value={String(savedModels.length)}
          tone={savedModels.length > 0 ? 'info' : 'warning'}
          sublabel={savedModels.length === 0 ? 'Train a model first' : undefined}
        />
        <StatusCard
          label="Best Avg R² (all models)"
          tone={savedModels.some((m) => m.avg_r2 != null) ? 'success' : 'warning'}
          value={
            savedModels.some((m) => m.avg_r2 != null)
              ? Math.max(...savedModels.map((m) => m.avg_r2 ?? -Infinity)).toFixed(4)
              : '—'
          }
        />
      </div>

      <section>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.15rem' }}>📊 Model Performance</h2>
          {savedModels.length > 0 && (
            <select value={activeModel ?? ''} onChange={(e) => setSelectedModel(e.target.value)}>
              {[...savedModels]
                .sort((a, b) => (a.saved_at < b.saved_at ? 1 : -1))
                .map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
            </select>
          )}
        </div>

        {savedModels.length === 0 && (
          <Callout variant="info">Upload data, preprocess, and train a model to see performance metrics here.</Callout>
        )}

        {performanceQuery.isLoading && activeModel && (
          <p className="caption">Computing performance for {activeModel}…</p>
        )}
        {performanceQuery.isError && (
          <p className="caption">Could not compute performance for {activeModel}.</p>
        )}

        {performanceQuery.data && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              {performanceQuery.data.per_target.map((t) => (
                <KpiCard key={t.name} name={t.name} r2={t.r2} mae={t.mae} />
              ))}
            </div>
            <p>
              <strong>Overall Average R²:</strong> <code>{performanceQuery.data.avg_r2.toFixed(4)}</code> —{' '}
              <strong>
                {performanceQuery.data.grade} {performanceQuery.data.emoji}
              </strong>
            </p>
            <div style={{ display: 'flex', gap: '3rem', flexWrap: 'wrap' }}>
              <div>
                <strong>Input Features (X)</strong>
                <ul>
                  {performanceQuery.data.x_cols.map((c) => (
                    <li key={c}>
                      <code>{c}</code>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <strong>Target Features (Y)</strong>
                <ul>
                  {performanceQuery.data.y_cols.map((c) => (
                    <li key={c}>
                      <code>{c}</code>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </section>

      <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ flex: 1, minWidth: 340 }}>
          <h2 style={{ fontSize: '1.15rem', marginBottom: '0.75rem' }}>📦 Datasets in Database</h2>
          <DataTable
            keyFn={(d) => d.name}
            maxVisibleRows={5}
            emptyMessage="No datasets stored yet."
            rows={datasets}
            columns={[
              { header: 'Name', render: (d) => d.name },
              { header: 'Uploaded', render: (d) => d.uploaded_at },
              { header: 'Rows', render: (d) => d.rows },
              { header: 'Cols', render: (d) => d.cols },
            ]}
          />
        </div>
        <div style={{ flex: 1, minWidth: 340 }}>
          <h2 style={{ fontSize: '1.15rem', marginBottom: '0.75rem' }}>💾 Saved Models</h2>
          <DataTable<SavedModelSummary>
            keyFn={(m) => m.name}
            emptyMessage="No models saved yet."
            rows={savedModels}
            columns={[
              { header: 'Name', render: (m) => m.name },
              { header: 'Algorithm', render: (m) => m.algorithm ?? '—' },
              { header: 'Saved At', render: (m) => m.saved_at },
              { header: 'Avg R²', render: (m) => (m.avg_r2 != null ? m.avg_r2.toFixed(4) : '—') },
            ]}
          />
        </div>
      </div>

      <WorkflowGuide />
    </div>
  )
}
