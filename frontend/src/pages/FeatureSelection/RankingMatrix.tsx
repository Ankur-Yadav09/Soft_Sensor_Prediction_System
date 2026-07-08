import { useMemo, useState } from 'react'
import type { FeatureSelectionResult } from '../../api/types'

const METHOD_ORDER: [string, string][] = [
  ['target_correlation', 'Correlation'],
  ['mutual_information', 'Mut. Info'],
  ['mrmr', 'mRMR'],
  ['permutation_importance', 'Permutation'],
  ['elasticnet', 'ElasticNet'],
]
const SELECTION_METHODS = new Set(['elasticnet'])
const SORT_OPTIONS = ['Final Score', 'Predictive Strength', 'Selection Frequency', 'Average Rank']
const REC_CATEGORIES = ['Highly Recommended', 'Recommended', 'Consider', 'Weak Feature']

const SORT_COL_MAP: Record<string, string> = {
  'Final Score': 'FinalScore',
  'Predictive Strength': 'PredictiveStrength',
  'Selection Frequency': 'SelectionFreq',
  'Average Rank': 'AvgRank',
}

// Matches the real app's colour scale exactly (red -> amber -> light green -> dark green).
const STOPS: [number, [number, number, number]][] = [
  [0.0, [239, 68, 68]],
  [0.3, [245, 158, 11]],
  [0.65, [134, 239, 172]],
  [1.0, [22, 163, 74]],
]
function goodnessColor(g: number): string {
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [p0, c0] = STOPS[i]
    const [p1, c1] = STOPS[i + 1]
    if (g >= p0 && g <= p1) {
      const t = (g - p0) / (p1 - p0 || 1)
      const rgb = c0.map((v, idx) => Math.round(v + (c1[idx] - v) * t))
      return `rgb(${rgb.join(',')})`
    }
  }
  return `rgb(${STOPS[STOPS.length - 1][1].join(',')})`
}

export function RankingMatrix({ result }: { result: FeatureSelectionResult }) {
  const [sortBy, setSortBy] = useState('Final Score')
  const [filterCats, setFilterCats] = useState<Set<string>>(new Set(REC_CATEGORIES))

  function toggleFilter(cat: string) {
    const next = new Set(filterCats)
    if (next.has(cat)) next.delete(cat)
    else next.add(cat)
    setFilterCats(next)
  }

  const activeMethods = METHOD_ORDER.filter(([mid]) =>
    result.method_results.some((m) => m.method_id === mid && m.success),
  )

  const rankMaps = useMemo(() => {
    const maps: Record<string, { rankOf: Record<string, number>; total: number }> = {}
    for (const [mid] of activeMethods) {
      const m = result.method_results.find((r) => r.method_id === mid)
      if (!m) continue
      const sorted = Object.entries(m.raw_scores).sort((a, b) => b[1] - a[1])
      const rankOf: Record<string, number> = {}
      sorted.forEach(([feat], i) => (rankOf[feat] = i + 1))
      maps[mid] = { rankOf, total: sorted.length }
    }
    return maps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  const rows = result.consensus
    .filter((r) => filterCats.has(r.Recommendation))
    .slice()
    .sort((a, b) => {
      const col = SORT_COL_MAP[sortBy] as keyof typeof a
      const av = Number(a[col] ?? 0)
      const bv = Number(b[col] ?? 0)
      return sortBy === 'Average Rank' ? av - bv : bv - av
    })

  if (rows.length === 0 || activeMethods.length === 0) {
    return <p className="caption">No data available for the ranking matrix.</p>
  }

  return (
    <div>
      <p className="caption" style={{ marginBottom: '1rem' }}>
        Each cell shows a feature's rank within that method (1 = top ranked). ElasticNet shows ✓ (selected) or ✗
        (not selected). Colour: 🟢 top-ranked → 🔴 lower-ranked.
      </p>

      <div style={{ display: 'flex', gap: '2rem', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <label>
          <div className="caption">Sort features by</div>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            {SORT_OPTIONS.map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </label>
        <div>
          <div className="caption">Filter by recommendation</div>
          <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.3rem' }}>
            {REC_CATEGORIES.map((cat) => (
              <button
                key={cat}
                className={`chip${filterCats.has(cat) ? ' active' : ''}`}
                onClick={() => toggleFilter(cat)}
                style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              {activeMethods.map(([, label]) => (
                <th key={label} style={{ textAlign: 'center' }}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.Feature}>
                <td>
                  <code>{r.Feature}</code>
                </td>
                {activeMethods.map(([mid]) => {
                  const m = result.method_results.find((mr) => mr.method_id === mid)!
                  if (SELECTION_METHODS.has(mid)) {
                    const selected = m.selected_features.includes(r.Feature)
                    return (
                      <td
                        key={mid}
                        style={{
                          textAlign: 'center',
                          background: goodnessColor(selected ? 1 : 0),
                          color: 'white',
                          fontWeight: 700,
                        }}
                      >
                        {selected ? '✓' : '✗'}
                      </td>
                    )
                  }
                  const map = rankMaps[mid]
                  const rank = map?.rankOf[r.Feature]
                  if (!rank) {
                    return <td key={mid} style={{ textAlign: 'center' }}></td>
                  }
                  const goodness = (map.total - rank) / Math.max(map.total - 1, 1)
                  return (
                    <td
                      key={mid}
                      style={{ textAlign: 'center', background: goodnessColor(goodness), color: 'white', fontWeight: 700 }}
                    >
                      {rank}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="caption" style={{ marginTop: '0.75rem' }}>
        <span style={{ color: '#16a34a', fontWeight: 700 }}>Dark green</span> = top ranks ·{' '}
        <span style={{ color: '#86efac', fontWeight: 700 }}>Light green</span> = good ·{' '}
        <span style={{ color: '#f59e0b', fontWeight: 700 }}>Amber</span> = lower ·{' '}
        <span style={{ color: '#ef4444', fontWeight: 700 }}>Red</span> = poor / not selected
      </p>
    </div>
  )
}
