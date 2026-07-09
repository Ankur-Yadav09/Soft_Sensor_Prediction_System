import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { deleteDataset, listDatasets } from '../../api/datasets'
import { DataTable } from '../../components/DataTable'
import type { DatasetSummary } from '../../api/types'

interface UseExistingDatasetTabProps {
  selectedName: string | null
  onSelect: (name: string) => void
}

const STATUS_COLOR: Record<string, string> = {
  Ready: 'var(--success-text)',
  Error: 'var(--error-text)',
}

export function UseExistingDatasetTab({ selectedName, onSelect }: UseExistingDatasetTabProps) {
  const [search, setSearch] = useState('')
  const queryClient = useQueryClient()

  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })

  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    },
  })

  const filtered = useMemo(() => {
    const rows = datasetsQuery.data ?? []
    if (!search.trim()) return rows
    const q = search.toLowerCase()
    return rows.filter(
      (d) =>
        d.name.toLowerCase().includes(q) ||
        (d.plant ?? '').toLowerCase().includes(q) ||
        (d.unit ?? '').toLowerCase().includes(q),
    )
  }, [datasetsQuery.data, search])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <input
        type="text"
        placeholder="🔍 Search by dataset name, plant, or unit…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{ maxWidth: 420 }}
      />

      <DataTable<DatasetSummary>
        keyFn={(d) => d.name}
        maxVisibleRows={5}
        emptyMessage={
          datasetsQuery.isLoading
            ? 'Loading datasets…'
            : 'No datasets found. Switch to "Upload New Dataset" to connect your first data source.'
        }
        rows={filtered}
        columns={[
          {
            header: 'Dataset Name',
            render: (d) => (
              <span style={{ fontWeight: d.name === selectedName ? 700 : 400, color: d.name === selectedName ? 'var(--primary)' : undefined }}>
                {d.name}
              </span>
            ),
          },
          { header: 'Plant', render: (d) => d.plant ?? '—' },
          { header: 'System/Unit', render: (d) => d.unit ?? '—' },
          { header: 'Upload Date', render: (d) => d.uploaded_at },
          { header: 'Records', render: (d) => d.rows.toLocaleString() },
          { header: 'Features', render: (d) => d.cols },
          {
            header: 'Status',
            render: (d) => (
              <span style={{ color: STATUS_COLOR[d.status ?? 'Ready'] ?? 'var(--text-main)', fontWeight: 600 }}>
                ● {d.status ?? 'Ready'}
              </span>
            ),
          },
          {
            header: 'Actions',
            render: (d) => (
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button style={{ padding: '0.3rem 0.7rem', fontSize: '0.78rem' }} onClick={() => onSelect(d.name)}>
                  Use
                </button>
                <button
                  className="chip"
                  style={{ padding: '0.3rem 0.7rem', fontSize: '0.78rem' }}
                  onClick={() => onSelect(d.name)}
                >
                  View Details
                </button>
                <button
                  className="chip"
                  style={{ padding: '0.3rem 0.7rem', fontSize: '0.78rem', color: 'var(--error-text)' }}
                  disabled={deleteMutation.isPending}
                  onClick={() => deleteMutation.mutate(d.name)}
                >
                  🗑️
                </button>
              </div>
            ),
          },
        ]}
      />
    </div>
  )
}
