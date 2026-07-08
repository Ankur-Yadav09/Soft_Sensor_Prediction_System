import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { deleteDataset, getDatasetPreview, listDatasets, uploadDataset } from '../../api/datasets'
import { Callout } from '../../components/Callout'
import { DataTable } from '../../components/DataTable'
import type { DatasetSummary } from '../../api/types'

export function UploadPage() {
  const queryClient = useQueryClient()
  const [file, setFile] = useState<File | null>(null)
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [message, setMessage] = useState<{ kind: 'success' | 'error'; text: string } | null>(null)

  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })

  const uploadMutation = useMutation({
    mutationFn: uploadDataset,
    onSuccess: (summary) => {
      setMessage({ kind: 'success', text: `Data saved to database as ${summary.name}!` })
      setSelectedName(summary.name)
      setFile(null)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    },
    onError: (err: unknown) => {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setMessage({ kind: 'error', text: detail ?? 'Upload failed.' })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: (_void, deletedName) => {
      setMessage({ kind: 'success', text: `Deleted ${deletedName} from database.` })
      if (selectedName === deletedName) setSelectedName(null)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    },
  })

  const previewQuery = useQuery({
    queryKey: ['datasets', selectedName, 'preview'],
    queryFn: () => getDatasetPreview(selectedName as string),
    enabled: !!selectedName,
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <h1>Upload Data</h1>

      {message && <Callout variant={message.kind}>{message.text}</Callout>}

      <div className="card" style={{ padding: '1.5rem' }}>
        <h2 style={{ fontSize: '1.05rem', marginBottom: '0.75rem' }}>Upload New File</h2>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <input
            type="file"
            accept=".xlsx,.xls,.csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button
            disabled={!file || uploadMutation.isPending}
            onClick={() => file && uploadMutation.mutate(file)}
          >
            {uploadMutation.isPending ? 'Uploading…' : 'Upload'}
          </button>
        </div>
      </div>

      <div>
        <h2 style={{ fontSize: '1.05rem', marginBottom: '0.75rem' }}>📦 Database Inventory</h2>
        {datasetsQuery.isLoading && <p className="caption">Loading datasets…</p>}
        {datasetsQuery.data && (
          <DataTable<DatasetSummary>
            keyFn={(d) => d.name}
            emptyMessage="No datasets in database yet. Upload a file to get started."
            rows={datasetsQuery.data}
            columns={[
              {
                header: 'Name',
                render: (d) => (
                  <button
                    onClick={() => setSelectedName(d.name)}
                    style={{
                      background: 'transparent',
                      boxShadow: 'none',
                      padding: 0,
                      color: d.name === selectedName ? 'var(--primary)' : 'var(--text-main)',
                      fontWeight: d.name === selectedName ? 700 : 400,
                    }}
                  >
                    {d.name}
                  </button>
                ),
              },
              { header: 'Uploaded On', render: (d) => d.uploaded_at },
              { header: 'Rows', render: (d) => d.rows },
              { header: 'Columns', render: (d) => d.cols },
              {
                header: '',
                render: (d) => (
                  <button
                    onClick={() => deleteMutation.mutate(d.name)}
                    disabled={deleteMutation.isPending}
                    style={{ padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}
                  >
                    🗑️ Delete
                  </button>
                ),
              },
            ]}
          />
        )}
      </div>

      {selectedName && (
        <div>
          <h2 style={{ fontSize: '1.05rem', marginBottom: '0.75rem' }}>Current Data Overview</h2>
          {previewQuery.isLoading && <p className="caption">Loading preview…</p>}
          {previewQuery.isError && <p className="caption">Could not load preview for {selectedName}.</p>}
          {previewQuery.data && (
            <>
              <p className="caption">
                <strong>Shape:</strong> ({previewQuery.data.shape[0]}, {previewQuery.data.shape[1]})
              </p>
              <div style={{ overflowX: 'auto' }}>
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
            </>
          )}
        </div>
      )}
    </div>
  )
}
