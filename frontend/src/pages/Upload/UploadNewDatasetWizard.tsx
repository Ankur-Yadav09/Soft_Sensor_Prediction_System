import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useRef, useState } from 'react'
import { listDatasets, uploadDataset } from '../../api/datasets'
import { Callout } from '../../components/Callout'
import type { DatasetSummary } from '../../api/types'

interface UploadNewDatasetWizardProps {
  onUploaded: (summary: DatasetSummary) => void
}

const ACCEPTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
const MAX_FILE_MB = 200

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function WizardStepBadge({ n, active, done }: { n: number; active: boolean; done: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <span
        className="step-badge"
        style={{
          background: active ? 'var(--primary)' : done ? 'var(--success-text)' : 'var(--control-bg)',
          color: active || done ? 'white' : 'var(--text-caption)',
        }}
      >
        {done ? '✓' : n}
      </span>
    </div>
  )
}

export function UploadNewDatasetWizard({ onUploaded }: UploadNewDatasetWizardProps) {
  const queryClient = useQueryClient()
  const [step, setStep] = useState<1 | 2 | 3>(1)
  const [plant, setPlant] = useState('')
  const [unit, setUnit] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [datasetName, setDatasetName] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [progress, setProgress] = useState(0)
  const [fileError, setFileError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })

  const { plantOptions, unitOptions } = useMemo(() => {
    const rows = datasetsQuery.data ?? []
    const plants = new Set<string>()
    const units = new Set<string>()
    rows.forEach((d) => {
      if (d.plant) plants.add(d.plant)
      if (d.unit) units.add(d.unit)
    })
    return { plantOptions: Array.from(plants), unitOptions: Array.from(units) }
  }, [datasetsQuery.data])

  const uploadMutation = useMutation({
    mutationFn: () =>
      uploadDataset(file as File, {
        datasetName: datasetName.trim() || undefined,
        plant: plant.trim() || undefined,
        unit: unit.trim() || undefined,
        onProgress: setProgress,
      }),
    onSuccess: (summary) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
      onUploaded(summary)
    },
  })

  const uploadError = (() => {
    const err = uploadMutation.error as { response?: { data?: { detail?: string } } } | undefined
    return err?.response?.data?.detail ?? (uploadMutation.isError ? 'Upload failed. Please try again.' : null)
  })()

  function validateAndSetFile(f: File) {
    setFileError(null)
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setFileError(`Unsupported file type "${ext}". Please use CSV or Excel (.csv, .xlsx, .xls).`)
      return
    }
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setFileError(`File is too large (max ${MAX_FILE_MB} MB).`)
      return
    }
    setFile(f)
    setDatasetName(f.name.replace(/\.[^/.]+$/, ''))
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragActive(false)
    const dropped = e.dataTransfer.files?.[0]
    if (dropped) validateAndSetFile(dropped)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Mini step indicator for the 3-step wizard */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {[1, 2, 3].map((n) => (
          <div key={n} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <WizardStepBadge n={n} active={step === n} done={step > n} />
            {n < 3 && <div style={{ width: 32, height: 2, background: step > n ? 'var(--success-text)' : 'var(--border)' }} />}
          </div>
        ))}
        <span className="caption" style={{ marginLeft: '0.5rem' }}>
          {step === 1 && 'Select plant & system/unit'}
          {step === 2 && 'Upload your file'}
          {step === 3 && 'Name & confirm'}
        </span>
      </div>

      {step === 1 && (
        <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="step-heading">
            <span className="step-badge">1</span>
            <h2>Where does this data come from?</h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', maxWidth: 640 }}>
            <div>
              <label htmlFor="plant-input">Plant</label>
              <input
                id="plant-input"
                type="text"
                list="plant-options"
                placeholder="e.g. North Refinery"
                value={plant}
                onChange={(e) => setPlant(e.target.value)}
                style={{ width: '100%', marginTop: '0.35rem' }}
              />
              <datalist id="plant-options">
                {plantOptions.map((p) => (
                  <option key={p} value={p} />
                ))}
              </datalist>
            </div>
            <div>
              <label htmlFor="unit-input">System / Unit</label>
              <input
                id="unit-input"
                type="text"
                list="unit-options"
                placeholder="e.g. Distillation Column 2"
                value={unit}
                onChange={(e) => setUnit(e.target.value)}
                style={{ width: '100%', marginTop: '0.35rem' }}
              />
              <datalist id="unit-options">
                {unitOptions.map((u) => (
                  <option key={u} value={u} />
                ))}
              </datalist>
            </div>
          </div>
          <p className="caption">Tagging plant and unit helps you find this dataset again later. You can leave these blank.</p>
          <div>
            <button onClick={() => setStep(2)}>Continue →</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="step-heading">
            <span className="step-badge">2</span>
            <h2>Upload your process data file</h2>
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragActive(true)
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragActive ? 'var(--primary)' : 'var(--border)'}`,
              borderRadius: 14,
              padding: '2.5rem 1.5rem',
              textAlign: 'center',
              cursor: 'pointer',
              background: dragActive ? 'var(--info-bg)' : 'var(--bg-subtle)',
              transition: 'all 0.15s ease',
            }}
          >
            <div style={{ fontSize: '2.2rem', marginBottom: '0.5rem' }}>📁</div>
            {file ? (
              <>
                <p style={{ fontWeight: 700, margin: 0 }}>{file.name}</p>
                <p className="caption" style={{ margin: 0 }}>{formatBytes(file.size)} · click or drop to replace</p>
              </>
            ) : (
              <>
                <p style={{ fontWeight: 700, margin: 0 }}>Drag & drop your file here</p>
                <p className="caption" style={{ margin: 0 }}>or click to browse — CSV, XLSX, XLS (max {MAX_FILE_MB} MB)</p>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(',')}
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) validateAndSetFile(f)
              }}
            />
          </div>

          {fileError && <Callout variant="error">{fileError}</Callout>}

          <div className="caption" style={{ background: 'var(--bg-subtle)', borderRadius: 10, padding: '0.85rem 1rem' }}>
            💡 <strong>Format tips:</strong> first row should be column headers, one row per timestamp/sample, numeric
            sensor readings in each column.
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="chip" onClick={() => setStep(1)}>← Back</button>
            <button disabled={!file} onClick={() => setStep(3)}>Continue →</button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="step-heading">
            <span className="step-badge">3</span>
            <h2>Name your dataset</h2>
          </div>

          <div style={{ maxWidth: 420 }}>
            <label htmlFor="dataset-name-input">Dataset Name</label>
            <input
              id="dataset-name-input"
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              style={{ width: '100%', marginTop: '0.35rem' }}
            />
          </div>

          <details style={{ fontSize: '0.87rem' }}>
            <summary style={{ cursor: 'pointer', color: 'var(--primary)', fontWeight: 600 }}>📋 File Format Guide</summary>
            <ul className="caption" style={{ marginTop: '0.5rem', paddingLeft: '1.2rem', lineHeight: 1.7 }}>
              <li>Supported formats: <code>.csv</code>, <code>.xlsx</code>, <code>.xls</code></li>
              <li>First row must contain unique column headers</li>
              <li>Sensor tags (X columns) and KPI targets (Y columns) as numeric columns</li>
              <li>Missing values may be left blank — handled in the Data Health step</li>
            </ul>
          </details>

          {uploadMutation.isPending && (
            <div>
              <div style={{ height: 8, borderRadius: 999, background: 'var(--control-bg)', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${progress}%`,
                    background: 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)',
                    transition: 'width 0.15s ease',
                  }}
                />
              </div>
              <p className="caption" style={{ marginTop: '0.4rem' }}>Uploading… {progress}%</p>
            </div>
          )}

          {uploadError && <Callout variant="error">{uploadError}</Callout>}

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="chip" onClick={() => setStep(2)} disabled={uploadMutation.isPending}>← Back</button>
            <button
              disabled={!file || !datasetName.trim() || uploadMutation.isPending}
              onClick={() => uploadMutation.mutate()}
            >
              {uploadMutation.isPending ? 'Uploading…' : 'Upload Dataset'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
