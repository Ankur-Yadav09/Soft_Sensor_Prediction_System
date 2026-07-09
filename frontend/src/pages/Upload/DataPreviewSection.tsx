import { useQuery } from '@tanstack/react-query'
import { getDatasetPreview } from '../../api/datasets'

interface DataPreviewSectionProps {
  datasetName: string
}

export function DataPreviewSection({ datasetName }: DataPreviewSectionProps) {
  const previewQuery = useQuery({
    queryKey: ['datasets', datasetName, 'preview'],
    queryFn: () => getDatasetPreview(datasetName),
  })

  return (
    <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h2 style={{ fontSize: '1.05rem' }}>👁️ Data Preview — {datasetName}</h2>
        {previewQuery.data && (
          <span className="caption">
            Showing first {Math.min(10, previewQuery.data.head.length)} of {previewQuery.data.shape[0].toLocaleString()} rows ·{' '}
            {previewQuery.data.shape[1]} columns
          </span>
        )}
      </div>

      {previewQuery.isLoading && <p className="caption">Loading preview…</p>}
      {previewQuery.isError && <p className="caption">Could not load preview for {datasetName}.</p>}

      {previewQuery.data && (
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
              {previewQuery.data.head.slice(0, 10).map((row, i) => (
                <tr key={i}>
                  {previewQuery.data!.columns.map((c) => (
                    <td key={c}>{String(row[c] ?? '')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
