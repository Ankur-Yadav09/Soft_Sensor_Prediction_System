import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { getFeatureDetail } from '../../api/preprocess'
import { Callout } from '../../components/Callout'
import { MiniBoxplot } from '../../components/MiniBoxplot'
import { MiniHistogram } from '../../components/MiniHistogram'
import { SectionBanner } from '../../components/SectionBanner'

interface DataUnderstandingTabProps {
  datasetName: string
  numericCols: string[]
}

export function DataUnderstandingTab({ datasetName, numericCols }: DataUnderstandingTabProps) {
  const [column, setColumn] = useState(numericCols[0] ?? '')

  const detailQuery = useQuery({
    queryKey: ['feature-detail', datasetName, column],
    queryFn: () => getFeatureDetail(datasetName, column),
    enabled: !!column,
  })

  const d = detailQuery.data

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <SectionBanner
        icon="🔍"
        title="Data Understanding"
        subtitle="Explore any feature in detail — statistics, distribution and outlier profile."
      />

      <label>
        <div className="caption">Select a feature to analyse</div>
        <select value={column} onChange={(e) => setColumn(e.target.value)} style={{ minWidth: 320 }}>
          {numericCols.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </label>

      {detailQuery.isLoading && <p className="caption">Loading…</p>}

      {d && d.empty && <Callout variant="warning">All values are missing for this feature.</Callout>}

      {d && !d.empty && (
        <>
          <p className="caption">
            Data Type: <code>{d.dtype}</code> · Distribution: <code>{d.distribution_label}</code>
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '1rem' }}>
            <Stat label="Total Records" value={String(d.n_total)} />
            <Stat label="Missing" value={`${d.n_missing} (${((d.n_missing / d.n_total) * 100).toFixed(1)}%)`} />
            <Stat label="Unique Values" value={String(d.n_unique)} />
            <Stat label="Duplicate Rows" value={String(d.n_duplicate_rows)} />
            <Stat label="Outliers (IQR)" value={String(d.n_outliers_iqr)} />
            <Stat label="Skewness" value={d.skew!.toFixed(3)} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '1rem' }}>
            <Stat label="Min" value={d.min!.toPrecision(4)} />
            <Stat label="Max" value={d.max!.toPrecision(4)} />
            <Stat label="Mean" value={d.mean!.toPrecision(4)} />
            <Stat label="Median" value={d.median!.toPrecision(4)} />
            <Stat label="Std Dev" value={d.std!.toPrecision(4)} />
            <Stat label="Kurtosis" value={d.kurtosis!.toFixed(3)} />
          </div>

          <Callout variant="info">
            <strong>{d.column}</strong> — {d.distribution_label?.toLowerCase()} distribution (skew ={' '}
            {d.skew!.toFixed(3)}, kurtosis = {d.kurtosis!.toFixed(3)}). Range: {d.min!.toPrecision(4)} →{' '}
            {d.max!.toPrecision(4)}, mean ± σ = {d.mean!.toPrecision(4)} ± {d.std!.toPrecision(4)}.
            {d.n_missing! > 0 && ` ${d.n_missing} missing values (${((d.n_missing! / d.n_total) * 100).toFixed(1)}%).`}
            {d.n_outliers_iqr! > 0 && ` ${d.n_outliers_iqr} potential outliers (IQR method).`}
            {Math.abs(d.skew!) > 1 && ' High skewness — consider transformation.'}
          </Callout>

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            {d.histogram && (
              <MiniHistogram counts={d.histogram.counts} binEdges={d.histogram.bin_edges} title={`Distribution — ${d.column}`} />
            )}
            {d.boxplot && (
              <MiniBoxplot {...d.boxplot} title={`Box Plot — ${d.column}`} />
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
