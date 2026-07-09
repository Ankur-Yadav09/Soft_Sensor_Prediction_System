import { useState } from 'react'
import { Callout } from '../../components/Callout'
import { DataTable } from '../../components/DataTable'
import { Tabs } from '../../components/Tabs'
import { RECOMMENDATION_COLOR } from './methodMeta'
import { RankingMatrix } from './RankingMatrix'
import type { FeatureConsensusRow, FeatureSelectionResult } from '../../api/types'

const REC_CATEGORIES = ['Highly Recommended', 'Recommended', 'Consider', 'Weak Feature']

// The reasoning text from src.feature_selection.auto_selector's
// _generate_reasoning() is markdown (a table + bullet list). Rendering it as
// literal text with just bold/newline substitution left pipe-table syntax
// visible, so this converts contiguous "| a | b |" line blocks into a real
// HTML table and leaves the rest as bold/line-break text.
function renderReasoningMarkdown(text: string): string {
  const lines = text.split('\n')
  const out: string[] = []
  let tableLines: string[] = []

  const flushTable = () => {
    if (tableLines.length === 0) return
    const rows = tableLines.filter((l) => !/^\|?\s*-{2,}/.test(l.replace(/\|/g, '')))
    const htmlRows = rows.map((l) => {
      const cells = l.split('|').map((c) => c.trim()).filter((c) => c.length > 0)
      return `<tr>${cells.map((c) => `<td>${c}</td>`).join('')}</tr>`
    })
    out.push(`<table>${htmlRows.join('')}</table>`)
    tableLines = []
  }

  for (const line of lines) {
    if (line.trim().startsWith('|')) {
      tableLines.push(line)
    } else {
      flushTable()
      out.push(line)
    }
  }
  flushTable()

  return out
    .join('\n')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
}

function OverviewTab({ result }: { result: FeatureSelectionResult }) {
  const info = result.dataset_info
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        <Stat label="Total Rows" value={String(info.n_rows)} />
        <Stat label="Clean Features" value={String(info.n_clean_features)} />
        <Stat label="Missing % (X)" value={`${info.missing_pct_x.toFixed(1)}%`} />
        <Stat label="Missing % (Y)" value={`${info.missing_pct_y.toFixed(1)}%`} />
      </div>

      {info.constant_features.length > 0 && (
        <Callout variant="warning">
          {info.constant_features.length} constant feature(s) removed (zero variance):{' '}
          <code>{info.constant_features.join(', ')}</code>
        </Callout>
      )}
      {info.vif_skipped && (
        <Callout variant="warning">
          VIF computation skipped — dataset has more than 80 features. The Highly Recommended hard gate (VIF
          &lt; 10) is inactive.
        </Callout>
      )}
      {info.permutation_skipped && (
        <Callout variant="warning">
          Permutation Importance skipped — dataset has more than 100 features. Its weight has been
          redistributed across the remaining methods.
        </Callout>
      )}

      <div>
        <h3 style={{ fontSize: '0.95rem', marginBottom: '0.6rem' }}>Method Execution Summary</h3>
        <DataTable
          keyFn={(m) => m.method_id}
          rows={result.method_results}
          columns={[
            { header: 'Method', render: (m) => m.name },
            { header: 'Category', render: (m) => m.category },
            { header: 'Status', render: (m) => (m.success ? '✅ Success' : '❌ Failed') },
            { header: 'Selected', render: (m) => m.selected_features.length },
            { header: 'Notes', render: (m) => m.notes },
          ]}
        />
      </div>

      <div>
        <h3 style={{ fontSize: '0.95rem', marginBottom: '0.6rem' }}>VIF (Multicollinearity)</h3>
        <DataTable
          keyFn={(v) => v.Feature}
          rows={result.vif}
          emptyMessage="No VIF data."
          columns={[
            { header: 'Feature', render: (v) => v.Feature },
            { header: 'VIF', render: (v) => (v.VIF != null ? v.VIF.toFixed(2) : '—') },
            { header: 'Level', render: (v) => v.VIF_Level },
          ]}
        />
      </div>
    </div>
  )
}

function ConsensusTab({ result }: { result: FeatureSelectionResult }) {
  return (
    <div>
      <p className="caption" style={{ marginBottom: '0.75rem' }}>
        <strong>Final Score</strong> = 30% × Selection Frequency (damped) + 50% × Predictive Strength + 20% ×
        Stability Score
      </p>
      <DataTable<FeatureConsensusRow>
        keyFn={(r) => r.Feature}
        rows={result.consensus}
        columns={[
          { header: 'Rank', render: (r) => r.Rank },
          { header: 'Feature', render: (r) => <code>{r.Feature}</code> },
          { header: 'Sel. Freq %', render: (r) => r.SelectionFreq.toFixed(1) },
          { header: 'Predictive Strength', render: (r) => r.PredictiveStrength.toFixed(1) },
          { header: 'Stability', render: (r) => r.StabilityScore.toFixed(1) },
          { header: 'Final Score', render: (r) => r.FinalScore.toFixed(1) },
          { header: 'Corr w/ Target', render: (r) => (r.CorrWithTarget != null ? r.CorrWithTarget.toFixed(3) : '—') },
          { header: 'VIF', render: (r) => (r.VIF != null ? r.VIF.toFixed(2) : '—') },
          {
            header: 'Recommendation',
            render: (r) => (
              <span style={{ color: RECOMMENDATION_COLOR[r.Recommendation] ?? 'var(--text-main)', fontWeight: 600 }}>
                {r.Recommendation}
              </span>
            ),
          },
        ]}
      />
    </div>
  )
}

function RecommendationsTab({ result }: { result: FeatureSelectionResult }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  function toggle(feat: string) {
    const next = new Set(expanded)
    if (next.has(feat)) next.delete(feat)
    else next.add(feat)
    setExpanded(next)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {REC_CATEGORIES.map((cat) => {
        const rows = result.consensus.filter((r) => r.Recommendation === cat)
        if (rows.length === 0) return null
        return (
          <div key={cat}>
            <h3 style={{ fontSize: '1rem', color: RECOMMENDATION_COLOR[cat], marginBottom: '0.6rem' }}>
              {cat} ({rows.length})
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {rows.map((r) => (
                <div key={r.Feature} className="card" style={{ padding: '0.9rem 1.1rem' }}>
                  <button
                    onClick={() => toggle(r.Feature)}
                    style={{
                      background: 'transparent',
                      boxShadow: 'none',
                      padding: 0,
                      width: '100%',
                      textAlign: 'left',
                      color: 'var(--text-main)',
                      fontWeight: 700,
                    }}
                  >
                    {expanded.has(r.Feature) ? '▾' : '▸'} {r.Feature} — Final Score: {r.FinalScore.toFixed(0)}
                  </button>
                  {expanded.has(r.Feature) && (
                    <div style={{ marginTop: '0.75rem' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '0.75rem' }}>
                        <Stat label="Final Score" value={r.FinalScore.toFixed(0)} />
                        <Stat label="Predictive Strength" value={r.PredictiveStrength.toFixed(1)} />
                        <Stat label="Stability" value={r.StabilityScore.toFixed(1)} />
                        <Stat label="VIF" value={r.VIF != null ? r.VIF.toFixed(1) : '—'} />
                      </div>
                      <div
                        className="caption reasoning"
                        style={{ lineHeight: 1.6 }}
                        dangerouslySetInnerHTML={{
                          __html: renderReasoningMarkdown(
                            result.per_feature_reasoning[r.Feature] ?? 'No reasoning available.',
                          ),
                        }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

type MethodDetailRow = Record<string, string | number>

function MethodDetailsTab({ result }: { result: FeatureSelectionResult }) {
  const [openMethod, setOpenMethod] = useState<string | null>(null)
  const corrByFeature = new Map(result.corr_with_target.map((rec) => [String(rec.Feature), rec]))
  const corrYCols = Object.keys(result.corr_with_target[0] ?? {}).filter((k) => k !== 'Feature')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <p className="caption">
        Methods that train per Y target show individual raw scores alongside the averaged raw score and
        normalized score. Methods that reduce to a single averaged Y show only Avg Raw and Norm Score.
      </p>
      {result.method_results.map((m) => {
        const isCorr = m.method_id === 'target_correlation' && corrYCols.length > 0
        const sampleFeat = m.selected_features[0]
        const yColNames = sampleFeat ? Object.keys(m.per_target_scores[sampleFeat] ?? {}) : []
        const hasPerY = !isCorr && yColNames.length > 0

        const rows: MethodDetailRow[] = m.selected_features.map((feat, i) => {
          const row: MethodDetailRow = { Rank: i + 1, Feature: feat }
          if (isCorr) {
            const rec = corrByFeature.get(feat)
            for (const yc of corrYCols) {
              const val = rec ? Number(rec[yc]) : NaN
              row[yc] = Number.isFinite(val) ? val.toFixed(4) : '—'
            }
            row['Avg |r|'] = m.raw_scores[feat]?.toFixed(5) ?? '—'
          } else {
            if (hasPerY) {
              for (const yc of yColNames) {
                row[`${yc} Raw`] = m.per_target_scores[feat]?.[yc]?.toFixed(5) ?? '—'
              }
            }
            row['Avg Raw'] = m.raw_scores[feat]?.toFixed(5) ?? '—'
          }
          row['Norm Score'] = m.all_scores[feat]?.toFixed(4) ?? '—'
          return row
        })

        const scoreCols = isCorr
          ? [...corrYCols.map((yc) => ({ header: yc, render: (r: MethodDetailRow) => r[yc] })), { header: 'Avg |r|', render: (r: MethodDetailRow) => r['Avg |r|'] }]
          : [
              ...(hasPerY ? yColNames.map((yc) => ({ header: `${yc} Raw`, render: (r: MethodDetailRow) => r[`${yc} Raw`] })) : []),
              { header: 'Avg Raw', render: (r: MethodDetailRow) => r['Avg Raw'] },
            ]

        return (
          <div key={m.method_id} className="card" style={{ padding: '0.9rem 1.1rem' }}>
            <button
              onClick={() => setOpenMethod(openMethod === m.method_id ? null : m.method_id)}
              style={{ background: 'transparent', boxShadow: 'none', padding: 0, width: '100%', textAlign: 'left', color: 'var(--text-main)', fontWeight: 700 }}
            >
              {openMethod === m.method_id ? '▾' : '▸'} {m.success ? '✅' : '❌'} {m.name} — {m.category} —{' '}
              {m.selected_features.length} features {m.notes && `(${m.notes})`}
            </button>
            {openMethod === m.method_id && (
              <div style={{ marginTop: '0.6rem' }}>
                {!m.success ? (
                  <Callout variant="error">{m.notes}</Callout>
                ) : rows.length === 0 ? (
                  <p className="caption">No features selected by this method.</p>
                ) : (
                  <>
                    <DataTable<MethodDetailRow>
                      keyFn={(r) => String(r.Feature)}
                      maxVisibleRows={8}
                      rows={rows}
                      columns={[
                        { header: 'Rank', render: (r) => r.Rank },
                        { header: 'Feature', render: (r) => <code>{r.Feature}</code> },
                        ...scoreCols,
                        { header: 'Norm Score', render: (r) => r['Norm Score'] },
                      ]}
                    />
                    <p className="caption" style={{ marginTop: '0.5rem' }}>
                      {isCorr
                        ? 'Y columns show signed Pearson r (+/– indicates direction). Avg |r| = mean absolute correlation across targets (used in scoring). Norm Score = normalised 0–1 (used in Final Score).'
                        : hasPerY
                          ? `Each Y column shows the raw score for that target run. Avg Raw = mean raw score across ${yColNames.length} target(s). Norm Score = normalised 0–1 (used in Final Score).`
                          : 'Avg Raw = raw score used to rank features for this method. Norm Score = normalised 0–1 (used in Final Score).'}
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function PerTargetTab({ result }: { result: FeatureSelectionResult }) {
  if (!result.per_target_summary) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {Object.entries(result.per_target_summary).map(([target, summary]) => (
        <div key={target} className="card" style={{ padding: '0.9rem 1.1rem' }}>
          <strong>{target}</strong>
          <p className="caption">Recommended: {summary.recommended_features.join(', ') || 'none'}</p>
          <p className="caption">Consider: {summary.optional_features.join(', ') || 'none'}</p>
        </div>
      ))}
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

export function FeatureSelectionResults({ result }: { result: FeatureSelectionResult }) {
  const nHighly = result.consensus.filter((r) => r.Recommendation === 'Highly Recommended').length
  const nRec = result.consensus.filter((r) => r.Recommendation === 'Recommended').length
  const nConsider = result.consensus.filter((r) => r.Recommendation === 'Consider').length
  const nWeak = result.consensus.filter((r) => r.Recommendation === 'Weak Feature').length

  const tabs = [
    { label: '📊 Overview', content: <OverviewTab result={result} /> },
    { label: '🏆 Consensus Rankings', content: <ConsensusTab result={result} /> },
    { label: '📈 Visualizations', content: <RankingMatrix result={result} /> },
    { label: '🎯 Recommendations', content: <RecommendationsTab result={result} /> },
    { label: '🔬 Method Details', content: <MethodDetailsTab result={result} /> },
  ]
  if (result.mode === 'per_target') {
    tabs.push({ label: '📋 Per-Target Summary', content: <PerTargetTab result={result} /> })
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        <Stat label="🟢 Highly Recommended" value={String(nHighly)} />
        <Stat label="🔵 Recommended" value={String(nRec)} />
        <Stat label="🟡 Consider" value={String(nConsider)} />
        <Stat label="🔴 Weak Feature" value={String(nWeak)} />
      </div>
      <Tabs tabs={tabs} />
    </div>
  )
}
