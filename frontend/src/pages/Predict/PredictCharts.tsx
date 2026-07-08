import { LineChart } from '../../components/LineChart'
import { MiniHistogram } from '../../components/MiniHistogram'
import { ScatterChart } from '../../components/ScatterChart'
import { computeHistogram } from '../../utils/histogram'
import type { PredictResult } from '../../api/types'

// Mirrors src/ui/components.py's render_actual_vs_predicted_lines /
// render_scatter_plots / render_residual_histograms — one row of 3 charts
// per target, built with the same plain-SVG components used elsewhere
// rather than pulling in a charting library.
export function PredictCharts({ result }: { result: PredictResult }) {
  if (!result.has_actuals) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {result.y_cols.map((col) => {
        const actual = result.rows.map((r) => Number(r[`${col}_actual`]))
        const predicted = result.rows.map((r) => Number(r[`${col}_predicted`]))
        const errors = result.rows.map((r) => Number(r[`${col}_error`]))
        const hist = computeHistogram(errors, 25)

        return (
          <div key={col}>
            <h3 style={{ fontSize: '0.95rem', marginBottom: '0.75rem' }}>{col}</h3>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <LineChart
                title="Actual vs Predicted"
                xLabel="Row"
                yLabel={col}
                series={[
                  { label: 'Actual', color: '#2563eb', data: actual },
                  { label: 'Predicted', color: '#f59e0b', data: predicted },
                ]}
              />
              <ScatterChart
                title="Scatter: Actual vs Predicted"
                xLabel="Actual"
                yLabel="Predicted"
                points={actual.map((a, i) => ({ x: a, y: predicted[i] }))}
              />
              <MiniHistogram title="Residual Distribution" counts={hist.counts} binEdges={hist.binEdges} />
            </div>
          </div>
        )
      })}
    </div>
  )
}
